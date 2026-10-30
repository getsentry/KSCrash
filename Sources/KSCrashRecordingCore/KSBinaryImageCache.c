//
// KSBinaryImageCache.c
//
// Created by Gleb Linnik on 2025-04-20.
//
// Copyright (c) 2012 Karl Stenerud. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall remain in place
// in this source code.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "KSBinaryImageCache.h"

#include <dlfcn.h>
#include <mach-o/dyld_images.h>
#include <mach-o/getsect.h>
#include <mach-o/loader.h>
#include <mach/mach.h>
#include <mach/task.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "KSLogger.h"
#include "KSMemory.h"
#include "KSPlatformSpecificDefines.h"

// MARK: - Image Address Range Cache

#define KSBIC_MAX_CACHE_ENTRIES 2048

// The effective capacity, which is the compile-time one except when a test lowers it. The
// cache-full branch is unreachable otherwise: saturating it needs 2048 distinct images and a
// test host has a few hundred. Atomic because the readers below run in crash handlers, where a
// test thread's write would otherwise be a data race; relaxed, since it orders nothing but
// itself, and lock-free so the crash-time read stays async-signal-safe.
static _Atomic uint32_t g_maxCacheEntries = KSBIC_MAX_CACHE_ENTRIES;
#define KSBIC_MAX_SEGMENTS_PER_IMAGE 16

static inline uint32_t maxCacheEntries(void) { return atomic_load_explicit(&g_maxCacheEntries, memory_order_relaxed); }

/**
 * Cached segment range for fast address-in-segment checks.
 */
typedef struct {
    uintptr_t start;  // Segment start address (with slide applied)
    uintptr_t end;    // Segment end address (exclusive, with slide applied)
} KSSegmentRange;

/**
 * Cached image address range for fast lookups.
 * Stores pre-computed segment ranges for fast address validation (O(segments), typically 4-6 segments).
 * Also caches unwind section pointers for async-signal-safe access during crash handling.
 */
typedef struct {
    uintptr_t startAddress;  // Min segment address (for quick rejection)
    uintptr_t endAddress;    // Max segment address (for quick rejection)
    uintptr_t slide;         // Pre-computed ASLR slide
    uintptr_t segmentBase;   // Pre-computed segment base for symbol lookups (vmaddr - fileoff for __LINKEDIT)
    const struct mach_header *_Nullable header;
    const char *_Nullable name;
    KSSegmentRange segments[KSBIC_MAX_SEGMENTS_PER_IMAGE];  // Actual segment ranges
    uint8_t segmentCount;                                   // Number of valid segments

    // Pointer to LC_UUID data in the Mach-O header (16 bytes, memory-mapped).
    // Valid for the lifetime of the loaded image.
    const uint8_t *_Nullable uuid;

    // Cached unwind section data (populated at image load time for async-signal-safety)
    KSBinaryImageUnwindInfo unwindInfo;
} KSBinaryImageRange;

typedef struct {
    KSBinaryImageRange entries[KSBIC_MAX_CACHE_ENTRIES];
    uint32_t count;
} KSBinaryImageRangeCache;

// Static cache storage (pre-allocated for async-signal-safety)
static KSBinaryImageRangeCache g_cache_storage = { .count = 0 };

// Atomic pointer to the cache. NULL means cache is in use by another caller.
static _Atomic(KSBinaryImageRangeCache *) g_cache_ptr = NULL;

// Check if an address falls within any of the cached segment ranges.
// This is O(segments) but segments is typically 4-6, so very fast.
static inline bool addressInCachedSegments(const KSBinaryImageRange *entry, uintptr_t address)
{
    for (uint8_t i = 0; i < entry->segmentCount; i++) {
        if (address >= entry->segments[i].start && address < entry->segments[i].end) {
            return true;
        }
    }
    return false;
}

// Binary search to find the rightmost entry with startAddress <= address.
// Returns -1 if no such entry exists.
// The cache must be sorted by startAddress in ascending order.
static inline int32_t binarySearchCache(const KSBinaryImageRangeCache *cache, uintptr_t address)
{
    if (cache->count == 0) {
        return -1;
    }

    int32_t left = 0;
    int32_t right = (int32_t)cache->count - 1;
    int32_t result = -1;

    while (left <= right) {
        int32_t mid = left + (right - left) / 2;
        if (cache->entries[mid].startAddress <= address) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

// Insert an entry into the cache maintaining sorted order by startAddress.
// Uses binary search to find insertion point, then shifts entries in-place.
// Avoid libc calls here to keep this async-signal-safe.
static void insertSortedCacheEntry(KSBinaryImageRangeCache *cache, const KSBinaryImageRange *entry)
{
    if (cache->count >= maxCacheEntries()) {
        return;
    }

    // Binary search for insertion point (first entry with startAddress >= entry->startAddress)
    int32_t left = 0;
    int32_t right = (int32_t)cache->count;

    while (left < right) {
        int32_t mid = left + (right - left) / 2;
        if (cache->entries[mid].startAddress < entry->startAddress) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // Shift entries to make room for the new entry.
    if (left < (int32_t)cache->count) {
        for (uint32_t i = cache->count; i > (uint32_t)left; i--) {
            cache->entries[i] = cache->entries[i - 1];
        }
    }

    cache->entries[left] = *entry;
    cache->count++;
}

// MARK: - Shared 64-bit Mach-O load-command walker

// Optional named-section lookup performed during a walk. Found addresses are returned slid
// (runtime addresses in the walked image's address space).
typedef struct {
    const char *segName;
    const char *sectName;
    uintptr_t addr;  // out: runtime (slid) address, 0 if not found
    uintptr_t size;  // out: section size, 0 if not found
    bool found;      // out
} KSMachOSectionQuery;

// Geometry recovered from one image's load commands. Runtime addresses (segments, queries,
// startAddress, endAddress) are slid; the fields explicitly documented as unslid below
// (textVMAddr, segmentBase) are raw link-time values and callers apply the slide themselves.
typedef struct {
    uintptr_t slide;
    uintptr_t textSize;      // __TEXT vmsize: the size in-process reports record for an image
    uint64_t textVMAddr;     // __TEXT vmaddr, UNSLID: what reports record as image_vmaddr, and
                             // what symbolication subtracts from the load address to get the slide
    uint64_t dylibVersion;   // LC_ID_DYLIB current_version, 0 if absent (packed major/minor/revision)
    uintptr_t segmentBase;   // UNSLID vmaddr - fileoff of __LINKEDIT (for symbol lookups), 0 if absent
    uintptr_t startAddress;  // min segment start, 0 if no segments
    uintptr_t endAddress;    // max segment end, 0 if no segments
    KSSegmentRange segments[KSBIC_MAX_SEGMENTS_PER_IMAGE];
    uint8_t segmentCount;
    const uint8_t *uuid;  // LC_UUID bytes pointing INTO the walked buffer, NULL if absent.
                          // Only valid as long as the walked buffer is; do not keep it when
                          // walking a temporary copy of a remote task's load commands.
} KSMachOImageGeometry;

// Walk a 64-bit Mach-O load-command region. `cmds` is either the live in-process region or a
// copy of a remote task's; every offset is bounds-checked against `sizeofcmds` because remote
// bytes come from a crashed (possibly memory-smashed) process and cannot be trusted.
// Async-signal-safe (no allocation, no libc beyond strncmp).
static void walkLoadCommands64(const uint8_t *cmds, uint32_t sizeofcmds, uint32_t ncmds, uintptr_t loadAddress,
                               KSMachOSectionQuery *queries, uint32_t queryCount, KSMachOImageGeometry *outGeometry)
{
    memset(outGeometry, 0, sizeof(*outGeometry));

    uintptr_t minAddr = UINTPTR_MAX;
    uintptr_t maxAddr = 0;
    bool foundText = false;
    bool foundLinkedit = false;

    // Walk via uintptr_t (not uint8_t *) so casts to load_command/segment structs don't trip
    // -Wcast-align.
    uintptr_t cmdPtr = (uintptr_t)cmds;
    const uintptr_t cmdsEnd = cmdPtr + sizeofcmds;
    for (uint32_t i = 0; i < ncmds; i++) {
        if (cmdPtr + sizeof(struct load_command) > cmdsEnd) {
            break;
        }
        const struct load_command *lc = (const struct load_command *)cmdPtr;
        if (lc->cmdsize < sizeof(struct load_command) || cmdPtr + lc->cmdsize > cmdsEnd) {
            break;
        }
        // 64-bit load commands are 8-byte aligned. Magnitude alone is not enough: an unaligned
        // cmdsize leaves every later cmdPtr misaligned, and the struct casts below then load
        // 64-bit fields from an odd address (UB, and a trap under -fsanitize=alignment).
        if (lc->cmdsize % 8 != 0) {
            break;
        }

        if (lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(struct segment_command_64)) {
            const struct segment_command_64 *seg = (const struct segment_command_64 *)cmdPtr;

            // __TEXT gives the slide, which converts vmaddrs to runtime addresses, and its
            // vmsize is the image size in-process reports record.
            if (!foundText && strncmp(seg->segname, SEG_TEXT, sizeof(seg->segname)) == 0) {
                outGeometry->slide = loadAddress - (uintptr_t)seg->vmaddr;
                outGeometry->textSize = (uintptr_t)seg->vmsize;
                outGeometry->textVMAddr = seg->vmaddr;
                foundText = true;
            }

            // __LINKEDIT gives the segment base for symbol lookups.
            if (!foundLinkedit && strncmp(seg->segname, SEG_LINKEDIT, sizeof(seg->segname)) == 0) {
                outGeometry->segmentBase = (uintptr_t)(seg->vmaddr - seg->fileoff);
                foundLinkedit = true;
            }

            // Match named sections inside this segment. nsects is untrusted (remote bytes),
            // so the bound is checked in integer space before any pointer arithmetic uses it
            // (sects + nsects on a hostile count would be out-of-bounds arithmetic, formal UB).
            if (queryCount > 0) {
                const struct section_64 *sects =
                    (const struct section_64 *)(cmdPtr + sizeof(struct segment_command_64));
                if ((uint64_t)seg->nsects * sizeof(struct section_64) <=
                    lc->cmdsize - sizeof(struct segment_command_64)) {
                    for (uint32_t q = 0; q < queryCount; q++) {
                        if (queries[q].found || strncmp(seg->segname, queries[q].segName, sizeof(seg->segname)) != 0) {
                            continue;
                        }
                        for (uint32_t s = 0; s < seg->nsects; s++) {
                            if (strncmp(sects[s].sectname, queries[q].sectName, sizeof(sects[s].sectname)) == 0) {
                                queries[q].addr = (uintptr_t)sects[s].addr;  // slid after the loop
                                queries[q].size = (uintptr_t)sects[s].size;
                                queries[q].found = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Store segments with actual file content (exclude __PAGEZERO)
            if (seg->vmsize > 0 && seg->filesize > 0) {
                uintptr_t segStart = (uintptr_t)seg->vmaddr;
                uintptr_t segEnd = segStart + (uintptr_t)seg->vmsize;
                if (segStart < minAddr) minAddr = segStart;
                if (segEnd > maxAddr) maxAddr = segEnd;
                if (outGeometry->segmentCount < KSBIC_MAX_SEGMENTS_PER_IMAGE) {
                    outGeometry->segments[outGeometry->segmentCount].start = segStart;
                    outGeometry->segments[outGeometry->segmentCount].end = segEnd;
                    outGeometry->segmentCount++;
                }
            }
        }
        if (lc->cmd == LC_UUID && lc->cmdsize >= sizeof(struct uuid_command)) {
            const struct uuid_command *ucmd = (const struct uuid_command *)cmdPtr;
            outGeometry->uuid = ucmd->uuid;
        }
        // LC_ID_DYLIB carries the version triple reports record for the image.
        if (lc->cmd == LC_ID_DYLIB && lc->cmdsize >= sizeof(struct dylib_command)) {
            const struct dylib_command *dcmd = (const struct dylib_command *)cmdPtr;
            outGeometry->dylibVersion = dcmd->dylib.current_version;
        }
        cmdPtr += lc->cmdsize;
    }

    // Apply the slide now that __TEXT has been seen.
    const uintptr_t slide = outGeometry->slide;
    for (uint8_t i = 0; i < outGeometry->segmentCount; i++) {
        outGeometry->segments[i].start += slide;
        outGeometry->segments[i].end += slide;
    }
    for (uint32_t q = 0; q < queryCount; q++) {
        if (queries[q].found) {
            queries[q].addr += slide;
        }
    }
    outGeometry->startAddress = (minAddr == UINTPTR_MAX) ? 0 : (minAddr + slide);
    outGeometry->endAddress = (maxAddr == 0) ? 0 : (maxAddr + slide);
}

// The 32-bit sibling of walkLoadCommands64, for MH_MAGIC images (arm64_32 on watchOS).
// Keep the two in lockstep: same bounds discipline, same geometry, same query handling.
// Async-signal-safe (no allocation, no libc beyond strncmp).
static void walkLoadCommands32(const uint8_t *cmds, uint32_t sizeofcmds, uint32_t ncmds, uintptr_t loadAddress,
                               KSMachOSectionQuery *queries, uint32_t queryCount, KSMachOImageGeometry *outGeometry)
{
    memset(outGeometry, 0, sizeof(*outGeometry));

    uintptr_t minAddr = UINTPTR_MAX;
    uintptr_t maxAddr = 0;
    bool foundText = false;
    bool foundLinkedit = false;

    uintptr_t cmdPtr = (uintptr_t)cmds;
    const uintptr_t cmdsEnd = cmdPtr + sizeofcmds;
    for (uint32_t i = 0; i < ncmds; i++) {
        if (cmdPtr + sizeof(struct load_command) > cmdsEnd) {
            break;
        }
        const struct load_command *lc = (const struct load_command *)cmdPtr;
        if (lc->cmdsize < sizeof(struct load_command) || cmdPtr + lc->cmdsize > cmdsEnd) {
            break;
        }
        // 32-bit load commands are 4-byte aligned; see the 64-bit walker for why magnitude
        // alone is not enough.
        if (lc->cmdsize % 4 != 0) {
            break;
        }

        if (lc->cmd == LC_SEGMENT && lc->cmdsize >= sizeof(struct segment_command)) {
            const struct segment_command *seg = (const struct segment_command *)cmdPtr;

            // __TEXT gives the slide, which converts vmaddrs to runtime addresses, and its
            // vmsize is the image size in-process reports record.
            if (!foundText && strncmp(seg->segname, SEG_TEXT, sizeof(seg->segname)) == 0) {
                outGeometry->slide = loadAddress - (uintptr_t)seg->vmaddr;
                outGeometry->textSize = (uintptr_t)seg->vmsize;
                outGeometry->textVMAddr = seg->vmaddr;
                foundText = true;
            }

            // __LINKEDIT gives the segment base for symbol lookups.
            if (!foundLinkedit && strncmp(seg->segname, SEG_LINKEDIT, sizeof(seg->segname)) == 0) {
                outGeometry->segmentBase = (uintptr_t)(seg->vmaddr - seg->fileoff);
                foundLinkedit = true;
            }

            // Match named sections inside this segment; nsects is untrusted, same as the
            // 64-bit walker.
            if (queryCount > 0) {
                const struct section *sects = (const struct section *)(cmdPtr + sizeof(struct segment_command));
                if ((uint64_t)seg->nsects * sizeof(struct section) <= lc->cmdsize - sizeof(struct segment_command)) {
                    for (uint32_t q = 0; q < queryCount; q++) {
                        if (queries[q].found || strncmp(seg->segname, queries[q].segName, sizeof(seg->segname)) != 0) {
                            continue;
                        }
                        for (uint32_t s = 0; s < seg->nsects; s++) {
                            if (strncmp(sects[s].sectname, queries[q].sectName, sizeof(sects[s].sectname)) == 0) {
                                queries[q].addr = (uintptr_t)sects[s].addr;  // slid after the loop
                                queries[q].size = (uintptr_t)sects[s].size;
                                queries[q].found = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Store segments with actual file content (exclude __PAGEZERO)
            if (seg->vmsize > 0 && seg->filesize > 0) {
                uintptr_t segStart = (uintptr_t)seg->vmaddr;
                uintptr_t segEnd = segStart + (uintptr_t)seg->vmsize;
                if (segStart < minAddr) minAddr = segStart;
                if (segEnd > maxAddr) maxAddr = segEnd;
                if (outGeometry->segmentCount < KSBIC_MAX_SEGMENTS_PER_IMAGE) {
                    outGeometry->segments[outGeometry->segmentCount].start = segStart;
                    outGeometry->segments[outGeometry->segmentCount].end = segEnd;
                    outGeometry->segmentCount++;
                }
            }
        }
        if (lc->cmd == LC_UUID && lc->cmdsize >= sizeof(struct uuid_command)) {
            const struct uuid_command *ucmd = (const struct uuid_command *)cmdPtr;
            outGeometry->uuid = ucmd->uuid;
        }
        // LC_ID_DYLIB carries the version triple reports record for the image.
        if (lc->cmd == LC_ID_DYLIB && lc->cmdsize >= sizeof(struct dylib_command)) {
            const struct dylib_command *dcmd = (const struct dylib_command *)cmdPtr;
            outGeometry->dylibVersion = dcmd->dylib.current_version;
        }
        cmdPtr += lc->cmdsize;
    }

    // Apply the slide now that __TEXT has been seen.
    const uintptr_t slide = outGeometry->slide;
    for (uint8_t i = 0; i < outGeometry->segmentCount; i++) {
        outGeometry->segments[i].start += slide;
        outGeometry->segments[i].end += slide;
    }
    for (uint32_t q = 0; q < queryCount; q++) {
        if (queries[q].found) {
            queries[q].addr += slide;
        }
    }
    outGeometry->startAddress = (minAddr == UINTPTR_MAX) ? 0 : (minAddr + slide);
    outGeometry->endAddress = (maxAddr == 0) ? 0 : (maxAddr + slide);
}

intptr_t ksbic_getImageSlide(const struct mach_header *header)
{
    if (header == NULL) {
        return 0;
    }

    uintptr_t loadAddr = (uintptr_t)header;

    if (header->magic == MH_MAGIC_64) {
        const struct mach_header_64 *header64 = (const struct mach_header_64 *)header;
        uintptr_t cmdPtr = (uintptr_t)(header64 + 1);

        for (uint32_t i = 0; i < header64->ncmds; i++) {
            const struct load_command *lc = (const struct load_command *)cmdPtr;
            if (lc->cmd == LC_SEGMENT_64) {
                const struct segment_command_64 *seg = (const struct segment_command_64 *)cmdPtr;
                // Check for __TEXT segment
                if (seg->segname[0] == '_' && seg->segname[1] == '_' && seg->segname[2] == 'T' &&
                    seg->segname[3] == 'E' && seg->segname[4] == 'X' && seg->segname[5] == 'T' &&
                    seg->segname[6] == '\0') {
                    return (intptr_t)(loadAddr - seg->vmaddr);
                }
            }
            cmdPtr += lc->cmdsize;
        }
    } else if (header->magic == MH_MAGIC) {
        uintptr_t cmdPtr = (uintptr_t)(header + 1);

        for (uint32_t i = 0; i < header->ncmds; i++) {
            const struct load_command *lc = (const struct load_command *)cmdPtr;
            if (lc->cmd == LC_SEGMENT) {
                const struct segment_command *seg = (const struct segment_command *)cmdPtr;
                // Check for __TEXT segment
                if (seg->segname[0] == '_' && seg->segname[1] == '_' && seg->segname[2] == 'T' &&
                    seg->segname[3] == 'E' && seg->segname[4] == 'X' && seg->segname[5] == 'T' &&
                    seg->segname[6] == '\0') {
                    return (intptr_t)(loadAddr - seg->vmaddr);
                }
            }
            cmdPtr += lc->cmdsize;
        }
    }

    return 0;
}

// Populate a cache entry with image info including segment ranges.
// Returns true if the image has valid segments, false otherwise.
static bool populateCacheEntry(const struct mach_header *header, const char *name, KSBinaryImageRange *entry)
{
    *entry = (KSBinaryImageRange) {
        .header = header,
        .name = name,
    };

    if (header == NULL) {
        return false;
    }

    uintptr_t loadAddr = (uintptr_t)header;
    uintptr_t slide = 0;
    uintptr_t segmentBase = 0;
    uintptr_t minAddr = UINTPTR_MAX;
    uintptr_t maxAddr = 0;
    bool foundText = false;
    bool foundLinkedit = false;
    uint8_t segCount = 0;

    if (header->magic == MH_MAGIC_64) {
        const struct mach_header_64 *header64 = (const struct mach_header_64 *)header;
        uintptr_t cmdPtr = (uintptr_t)(header64 + 1);

        for (uint32_t i = 0; i < header64->ncmds; i++) {
            const struct load_command *lc = (const struct load_command *)cmdPtr;
            if (lc->cmd == LC_SEGMENT_64) {
                const struct segment_command_64 *seg = (const struct segment_command_64 *)cmdPtr;

                // Check for __TEXT to compute slide (only once)
                if (!foundText && seg->segname[0] == '_' && seg->segname[1] == '_' && seg->segname[2] == 'T' &&
                    seg->segname[3] == 'E' && seg->segname[4] == 'X' && seg->segname[5] == 'T' &&
                    seg->segname[6] == '\0') {
                    slide = loadAddr - (uintptr_t)seg->vmaddr;
                    foundText = true;
                }

                // Check for __LINKEDIT to compute segment base (only once)
                if (!foundLinkedit && seg->segname[0] == '_' && seg->segname[1] == '_' && seg->segname[2] == 'L' &&
                    seg->segname[3] == 'I' && seg->segname[4] == 'N' && seg->segname[5] == 'K' &&
                    seg->segname[6] == 'E' && seg->segname[7] == 'D' && seg->segname[8] == 'I' &&
                    seg->segname[9] == 'T' && seg->segname[10] == '\0') {
                    segmentBase = (uintptr_t)(seg->vmaddr - seg->fileoff);
                    foundLinkedit = true;
                }

                // Store segments with actual file content (exclude __PAGEZERO)
                if (seg->vmsize > 0 && seg->filesize > 0) {
                    uintptr_t segStart = (uintptr_t)seg->vmaddr;
                    uintptr_t segEnd = segStart + (uintptr_t)seg->vmsize;
                    if (segStart < minAddr) minAddr = segStart;
                    if (segEnd > maxAddr) maxAddr = segEnd;

                    // Store segment range (will apply slide after loop)
                    if (segCount < KSBIC_MAX_SEGMENTS_PER_IMAGE) {
                        entry->segments[segCount].start = segStart;
                        entry->segments[segCount].end = segEnd;
                        segCount++;
                    } else {
                        KSLOG_DEBUG("Image %s exceeds max segments (%d), truncating", name ? name : "<unknown>",
                                    KSBIC_MAX_SEGMENTS_PER_IMAGE);
                    }
                }
            }
            if (lc->cmd == LC_UUID) {
                const struct uuid_command *ucmd = (const struct uuid_command *)cmdPtr;
                entry->uuid = ucmd->uuid;
            }
            cmdPtr += lc->cmdsize;
        }
    } else if (header->magic == MH_MAGIC) {
        uintptr_t cmdPtr = (uintptr_t)(header + 1);

        for (uint32_t i = 0; i < header->ncmds; i++) {
            const struct load_command *lc = (const struct load_command *)cmdPtr;
            if (lc->cmd == LC_SEGMENT) {
                const struct segment_command *seg = (const struct segment_command *)cmdPtr;

                // Check for __TEXT to compute slide (only once)
                if (!foundText && seg->segname[0] == '_' && seg->segname[1] == '_' && seg->segname[2] == 'T' &&
                    seg->segname[3] == 'E' && seg->segname[4] == 'X' && seg->segname[5] == 'T' &&
                    seg->segname[6] == '\0') {
                    slide = loadAddr - (uintptr_t)seg->vmaddr;
                    foundText = true;
                }

                // Check for __LINKEDIT to compute segment base (only once)
                if (!foundLinkedit && seg->segname[0] == '_' && seg->segname[1] == '_' && seg->segname[2] == 'L' &&
                    seg->segname[3] == 'I' && seg->segname[4] == 'N' && seg->segname[5] == 'K' &&
                    seg->segname[6] == 'E' && seg->segname[7] == 'D' && seg->segname[8] == 'I' &&
                    seg->segname[9] == 'T' && seg->segname[10] == '\0') {
                    segmentBase = (uintptr_t)(seg->vmaddr - seg->fileoff);
                    foundLinkedit = true;
                }

                // Store segments with actual file content (exclude __PAGEZERO)
                if (seg->vmsize > 0 && seg->filesize > 0) {
                    uintptr_t segStart = (uintptr_t)seg->vmaddr;
                    uintptr_t segEnd = segStart + (uintptr_t)seg->vmsize;
                    if (segStart < minAddr) minAddr = segStart;
                    if (segEnd > maxAddr) maxAddr = segEnd;

                    // Store segment range (will apply slide after loop)
                    if (segCount < KSBIC_MAX_SEGMENTS_PER_IMAGE) {
                        entry->segments[segCount].start = segStart;
                        entry->segments[segCount].end = segEnd;
                        segCount++;
                    } else {
                        KSLOG_DEBUG("Image %s exceeds max segments (%d), truncating", name ? name : "<unknown>",
                                    KSBIC_MAX_SEGMENTS_PER_IMAGE);
                    }
                }
            }
            if (lc->cmd == LC_UUID) {
                const struct uuid_command *ucmd = (const struct uuid_command *)cmdPtr;
                entry->uuid = ucmd->uuid;
            }
            cmdPtr += lc->cmdsize;
        }
    }

    // Apply slide to all segment ranges
    for (uint8_t i = 0; i < segCount; i++) {
        entry->segments[i].start += slide;
        entry->segments[i].end += slide;
    }

    entry->slide = slide;
    entry->segmentBase = segmentBase;
    entry->startAddress = (minAddr == UINTPTR_MAX) ? 0 : (minAddr + slide);
    entry->endAddress = (maxAddr == 0) ? 0 : (maxAddr + slide);
    entry->segmentCount = segCount;

    // Cache unwind section pointers using getsectiondata().
    // getsectiondata() correctly handles images in the dyld shared cache and is
    // async-signal-safe on Apple platforms (its only non-trivial call is strncmp,
    // which is async-signal-safe on Apple platforms).
    entry->unwindInfo.header = header;
    entry->unwindInfo.slide = slide;

    unsigned long unwindSectionSize = 0;
    entry->unwindInfo.unwindInfo =
        getsectiondata((const mach_header_t *)header, SEG_TEXT, "__unwind_info", &unwindSectionSize);
    entry->unwindInfo.unwindInfoSize = (size_t)unwindSectionSize;
    entry->unwindInfo.hasCompactUnwind = (entry->unwindInfo.unwindInfo != NULL && unwindSectionSize > 0);

    unsigned long ehFrameSize = 0;
    entry->unwindInfo.ehFrame = getsectiondata((const mach_header_t *)header, SEG_TEXT, "__eh_frame", &ehFrameSize);
    entry->unwindInfo.ehFrameSize = (size_t)ehFrameSize;
    entry->unwindInfo.hasEhFrame = (entry->unwindInfo.ehFrame != NULL && ehFrameSize > 0);

    KSLOG_TRACE("Cached image %s: unwind=%p(%zu) eh_frame=%p(%zu)", name ? name : "<unknown>",
                entry->unwindInfo.unwindInfo, entry->unwindInfo.unwindInfoSize, entry->unwindInfo.ehFrame,
                entry->unwindInfo.ehFrameSize);

    return (segCount > 0);
}

// Linear scan through dyld images to find one containing the address.
// If outEntry is provided, populates it with full image info for caching.
static const struct mach_header *linearScanForAddress(uintptr_t address, KSBinaryImageRange *outEntry)
{
    uint32_t count = 0;
    const ks_dyld_image_info *images = ksbic_getImages(&count);
    if (images == NULL) {
        return NULL;
    }

    KSBinaryImageRange tempEntry = { 0 };
    for (uint32_t i = 0; i < count; i++) {
        const struct mach_header *header = images[i].imageLoadAddress;
        if (header == NULL) {
            continue;
        }

        // Populate entry with segment info
        if (!populateCacheEntry(header, images[i].imageFilePath, &tempEntry)) {
            continue;
        }

        // Check if address is in any segment of this image
        // This is critical for dyld shared cache where segments from different images can be interleaved
        if (addressInCachedSegments(&tempEntry, address)) {
            if (outEntry) {
                *outEntry = tempEntry;
            }
            return header;
        }
    }

    return NULL;
}

/// As a general rule, access to _g_all_image_infos->infoArray_ is thread safe
/// in a way that you can iterate all you want since items will never be removed
/// and the _infoCount_ is only updated after an item is added to _infoArray_.
/// Because of this, we can iterate during a signal handler, Mach exception handler
/// or even at any point within the run of the process.
///
/// More info in this comment:
/// https://github.com/kstenerud/KSCrash/pull/655#discussion_r2211271075

// Atomic so test-only `ksbic_resetCache` can NULL the pointer without racing
// concurrent readers (getImages, getAppHeader, etc.). The pointer itself is
// the only thing that races; the dyld struct it points to is managed by dyld
// and its members have their own well-known thread-safety contract.
static _Atomic(struct dyld_all_image_infos *) g_all_image_infos = NULL;

// Original dyld notifier that we chain to after our processing
static dyld_image_notifier g_original_notifier = NULL;

// User-registered callback for image additions (atomic for thread safety)
static _Atomic(ksbic_imageCallback) g_image_added_callback = NULL;

void ksbic_registerForImageAdded(ksbic_imageCallback callback) { atomic_store(&g_image_added_callback, callback); }

// Forward declaration for use in dyld notifier
static int32_t findByHeader(const KSBinaryImageRangeCache *cache, const struct mach_header *header);

/**
 * Our custom dyld image notifier that gets called when images are added or removed.
 * We use this to invalidate/update our cache, then call the original notifier.
 *
 * IMPORTANT: This function pre-populates the cache for newly added images. This ensures
 * that unwind info is available immediately during crash handling without a cache miss.
 */
static void ksbic_dyld_image_notifier(enum dyld_image_mode mode, uint32_t infoCount,
                                      const struct dyld_image_info info[])
{
    // Only handle image additions. Image removal is effectively a no-op on Apple platforms
    // (dlclose doesn't actually unload due to Objective-C runtime, Swift, etc.)
    if (mode == dyld_image_adding) {
        KSLOG_DEBUG("dyld notifier: %u images added", infoCount);

        // Pre-populate cache so data is available immediately during crash handling.
        KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);
        if (cache != NULL) {
            for (uint32_t i = 0; i < infoCount && cache->count < maxCacheEntries(); i++) {
                const struct mach_header *header = info[i].imageLoadAddress;
                const char *name = info[i].imageFilePath;

                // Check if already in cache
                if (findByHeader(cache, header) >= 0) {
                    continue;
                }

                KSBinaryImageRange newEntry;
                if (populateCacheEntry(header, name, &newEntry)) {
                    insertSortedCacheEntry(cache, &newEntry);
                }
            }
            atomic_store(&g_cache_ptr, cache);
        }

        ksbic_imageCallback callback = atomic_load(&g_image_added_callback);
        if (callback != NULL) {
            for (uint32_t i = 0; i < infoCount; i++) {
                const struct mach_header *header = info[i].imageLoadAddress;
                intptr_t slide = ksbic_getImageSlide(header);
                callback(header, slide);
            }
        }
    }

    // Chain to the original notifier if one was installed
    dyld_image_notifier original = g_original_notifier;
    if (original != NULL) {
        original(mode, infoCount, info);
    }
}

void ksbic_init(void)
{
    KSLOG_DEBUG("Initializing binary image cache");

    struct task_dyld_info dyld_info;
    mach_msg_type_number_t count = TASK_DYLD_INFO_COUNT;
    kern_return_t err = task_info(mach_task_self(), TASK_DYLD_INFO, (task_info_t)&dyld_info, &count);
    if (err != KERN_SUCCESS) {
        KSLOG_ERROR("Failed to acquire TASK_DYLD_INFO. We won't have access to binary images.");
        return;
    }
    struct dyld_all_image_infos *allInfo = (struct dyld_all_image_infos *)dyld_info.all_image_info_addr;
    atomic_store_explicit(&g_all_image_infos, allInfo, memory_order_relaxed);

    // Initialize the address range cache
    g_cache_storage.count = 0;

    // Cache the main executable — always the first image, almost certainly in every backtrace
    if (allInfo->infoArrayCount > 0 && allInfo->infoArray != NULL) {
        const struct dyld_image_info *mainImage = &allInfo->infoArray[0];
        if (mainImage->imageLoadAddress != NULL) {
            KSBinaryImageRange mainEntry;
            if (populateCacheEntry(mainImage->imageLoadAddress, mainImage->imageFilePath, &mainEntry)) {
                insertSortedCacheEntry(&g_cache_storage, &mainEntry);
            }
        }
    }

    // Cache dyld's own image — it's not in the infoArray.
    if (allInfo->dyldImageLoadAddress != NULL) {
        KSBinaryImageRange dyldEntry;
        if (populateCacheEntry(allInfo->dyldImageLoadAddress, ksbic_getDyldPath(), &dyldEntry)) {
            insertSortedCacheEntry(&g_cache_storage, &dyldEntry);
        }
    }

    atomic_store(&g_cache_ptr, &g_cache_storage);

    // Install our dyld image notifier to track image loading/unloading.
    // Save the original notifier so we can chain to it.
    // Only install if not already installed (avoid caching our own notifier on re-init).
    if (allInfo->notification != ksbic_dyld_image_notifier) {
        g_original_notifier = allInfo->notification;
        allInfo->notification = ksbic_dyld_image_notifier;
        KSLOG_DEBUG("Installed dyld image notifier (original=%p)", (void *)g_original_notifier);
    }
}

const ks_dyld_image_info *ksbic_getImages(uint32_t *count)
{
    if (count) {
        *count = 0;
    }
    struct dyld_all_image_infos *allInfo = atomic_load_explicit(&g_all_image_infos, memory_order_relaxed);
    if (allInfo == NULL) {
        KSLOG_ERROR("Cannot access binary images");
        return NULL;
    }
    const struct dyld_image_info *images = allInfo->infoArray;
    if (images == NULL) {
        KSLOG_ERROR("Unexpected state: dyld_all_image_infos->infoArray is NULL!");
        return NULL;
    }
    if (count) {
        *count = allInfo->infoArrayCount;
    }
    return (ks_dyld_image_info *)images;
}

const struct mach_header *ksbic_getAppHeader(void)
{
    struct dyld_all_image_infos *allInfo = atomic_load_explicit(&g_all_image_infos, memory_order_relaxed);
    if (allInfo == NULL || allInfo->infoArray == NULL || allInfo->infoArrayCount == 0) {
        return NULL;
    }
    return allInfo->infoArray[0].imageLoadAddress;
}

const struct mach_header *ksbic_getDyldHeader(void)
{
    struct dyld_all_image_infos *allInfo = atomic_load_explicit(&g_all_image_infos, memory_order_relaxed);
    if (allInfo == NULL) {
        return NULL;
    }
    return allInfo->dyldImageLoadAddress;
}

const char *ksbic_getDyldPath(void)
{
    struct dyld_all_image_infos *allInfo = atomic_load_explicit(&g_all_image_infos, memory_order_relaxed);
    if (allInfo != NULL && allInfo->dyldPath != NULL) {
        return allInfo->dyldPath;
    }
    // dyldPath requires version 15 of the struct (macOS 10.12, iOS 10.0).
    // Fall back to the well-known path on older systems.
    return "/usr/lib/dyld";
}

// For testing purposes only. Used with extern in test files.
// Test seam: lower (or restore) the cache's effective capacity so the cache-full branch becomes
// reachable. Pass 0 to restore the compile-time capacity.
//
// Also trims what is already cached down to the new capacity, because that branch is reached
// only on a cache MISS against a full cache: merely capping growth would leave the pre-populated
// entries in place and every real address would still be answered from them. Trimming keeps the
// remaining entries sorted, so lookups against them behave normally.
void ksbic_testcode_setMaxCacheEntries(uint32_t maxEntries)
{
    // Clamped, not just defaulted: the backing array is fixed at KSBIC_MAX_CACHE_ENTRIES, and
    // the readers treat this value as the array's capacity. A larger one would let
    // insertSortedCacheEntry run count off the end of g_cache_storage.entries.
    uint32_t newMax = maxEntries == 0 ? KSBIC_MAX_CACHE_ENTRIES : maxEntries;
    if (newMax > KSBIC_MAX_CACHE_ENTRIES) {
        newMax = KSBIC_MAX_CACHE_ENTRIES;
    }
    atomic_store_explicit(&g_maxCacheEntries, newMax, memory_order_relaxed);

    KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);
    if (cache != NULL) {
        if (cache->count > newMax) {
            cache->count = newMax;
        }
        atomic_store(&g_cache_ptr, cache);
    }
}

void ksbic_resetCache(void)
{
    // Restore the original dyld notifier before resetting
    struct dyld_all_image_infos *allInfo = atomic_load_explicit(&g_all_image_infos, memory_order_relaxed);
    if (allInfo != NULL && allInfo->notification == ksbic_dyld_image_notifier) {
        allInfo->notification = g_original_notifier;
    }
    g_original_notifier = NULL;
    atomic_store_explicit(&g_all_image_infos, NULL, memory_order_relaxed);

    // Acquire exclusive access to the cache before resetting
    KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);
    if (cache != NULL) {
        cache->count = 0;
        atomic_store(&g_cache_ptr, cache);
    } else {
        // Cache is in use by another thread - reset storage directly
        // and restore pointer (the other thread will see stale data but
        // that's acceptable for a reset operation)
        g_cache_storage.count = 0;
        atomic_store(&g_cache_ptr, &g_cache_storage);
    }
}

const struct mach_header *ksbic_findImageForAddress(uintptr_t address, uintptr_t *outSlide, const char **outName)
{
    return ksbic_getImageDetailsForAddress(address, outSlide, NULL, outName);
}

const struct mach_header *ksbic_getImageDetailsForAddress(uintptr_t address, uintptr_t *outSlide,
                                                          uintptr_t *outSegmentBase, const char **outName)
{
    // Try to acquire exclusive access to the cache
    KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);

    if (cache != NULL) {
        // SUCCESS: We have exclusive access to the cache

        // Use binary search to find candidate entries.
        // The cache is sorted by startAddress, so we find the rightmost entry
        // with startAddress <= address, then check it and scan backwards for
        // overlapping ranges (due to dyld shared cache).
        int32_t idx = binarySearchCache(cache, address);

        // Check the found entry and scan backwards for overlapping ranges
        while (idx >= 0) {
            KSBinaryImageRange *entry = &cache->entries[idx];

            // Check if address is in range and verify with segment check
            if (address >= entry->startAddress && address < entry->endAddress) {
                if (addressInCachedSegments(entry, address)) {
                    // Cache hit - found the image
                    if (outSlide) *outSlide = entry->slide;
                    if (outSegmentBase) *outSegmentBase = entry->segmentBase;
                    if (outName) *outName = entry->name;
                    const struct mach_header *result = entry->header;

                    // Release the cache
                    atomic_store(&g_cache_ptr, cache);
                    return result;
                }
            }
            idx--;
        }

        // Cache miss - do linear scan
        KSBinaryImageRange newEntry;
        const struct mach_header *header = linearScanForAddress(address, &newEntry);

        if (header != NULL && cache->count < maxCacheEntries()) {
            // Add to cache maintaining sorted order
            insertSortedCacheEntry(cache, &newEntry);
        }

        if (header != NULL) {
            if (outSlide) *outSlide = newEntry.slide;
            if (outSegmentBase) *outSegmentBase = newEntry.segmentBase;
            if (outName) *outName = newEntry.name;
        } else {
            if (outSlide) *outSlide = 0;
            if (outSegmentBase) *outSegmentBase = 0;
            if (outName) *outName = NULL;
        }

        // Release the cache
        atomic_store(&g_cache_ptr, cache);
        return header;
    } else {
        // FAILED: Cache is in use by another caller
        // Fall back to linear scan without caching
        KSBinaryImageRange entry;
        const struct mach_header *header = linearScanForAddress(address, &entry);

        if (header != NULL) {
            if (outSlide) *outSlide = entry.slide;
            if (outSegmentBase) *outSegmentBase = entry.segmentBase;
            if (outName) *outName = entry.name;
        } else {
            if (outSlide) *outSlide = 0;
            if (outSegmentBase) *outSegmentBase = 0;
            if (outName) *outName = NULL;
        }
        return header;
    }
}

// Find cache entry by header pointer. Returns index or -1 if not found.
static int32_t findByHeader(const KSBinaryImageRangeCache *cache, const struct mach_header *header)
{
    for (uint32_t i = 0; i < cache->count; i++) {
        if (cache->entries[i].header == header) {
            return (int32_t)i;
        }
    }
    return -1;
}

bool ksbic_getUnwindInfoForHeader(const struct mach_header *header, KSBinaryImageUnwindInfo *outInfo)
{
    if (header == NULL) {
        return false;
    }

    KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);

    if (cache != NULL) {
        int32_t idx = findByHeader(cache, header);
        if (idx >= 0) {
            if (outInfo) {
                *outInfo = cache->entries[idx].unwindInfo;
            }
            atomic_store(&g_cache_ptr, cache);
            return true;
        }

        // Not in cache - populate and add maintaining sorted order
        KSBinaryImageRange newEntry;
        if (populateCacheEntry(header, NULL, &newEntry)) {
            if (cache->count < maxCacheEntries()) {
                insertSortedCacheEntry(cache, &newEntry);
            }
            if (outInfo) {
                *outInfo = newEntry.unwindInfo;
            }
            atomic_store(&g_cache_ptr, cache);
            return true;
        }

        atomic_store(&g_cache_ptr, cache);
        return false;
    } else {
        // Cache busy — fall back to uncached population
        KSBinaryImageRange entry;
        if (populateCacheEntry(header, NULL, &entry)) {
            if (outInfo) {
                *outInfo = entry.unwindInfo;
            }
            return true;
        }
        return false;
    }
}

bool ksbic_getUnwindInfoForAddress(uintptr_t address, KSBinaryImageUnwindInfo *outInfo)
{
    KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);

    if (cache != NULL) {
        // Binary search for the image containing this address
        int32_t idx = binarySearchCache(cache, address);

        while (idx >= 0) {
            KSBinaryImageRange *entry = &cache->entries[idx];
            if (address >= entry->startAddress && address < entry->endAddress) {
                if (addressInCachedSegments(entry, address)) {
                    if (outInfo) {
                        *outInfo = entry->unwindInfo;
                    }
                    atomic_store(&g_cache_ptr, cache);
                    return true;
                }
            }
            idx--;
        }

        // Cache miss — populate and add
        // Answering the lookup and caching the answer are separate concerns, as in
        // ksbic_getImageDetailsForAddress above: a full cache must still fill *outInfo, because
        // callers key off the return value alone. tryCompactUnwindForPC reads an uninitialized
        // KSBinaryImageUnwindInfo the moment this returns true without writing it, then unwinds
        // through a garbage section pointer inside the crash handler.
        KSBinaryImageRange newEntry;
        const struct mach_header *header = linearScanForAddress(address, &newEntry);
        if (header != NULL) {
            if (cache->count < maxCacheEntries()) {
                insertSortedCacheEntry(cache, &newEntry);
            }
            if (outInfo) {
                *outInfo = newEntry.unwindInfo;
            }
        }

        atomic_store(&g_cache_ptr, cache);
        return header != NULL;
    } else {
        // Cache busy — fall back to uncached population
        KSBinaryImageRange entry;
        if (linearScanForAddress(address, &entry) != NULL) {
            if (outInfo) {
                *outInfo = entry.unwindInfo;
            }
            return true;
        }
        return false;
    }
}

const uint8_t *ksbic_getUUIDForHeader(const struct mach_header *header)
{
    if (header == NULL) {
        return NULL;
    }

    KSBinaryImageRangeCache *cache = atomic_exchange(&g_cache_ptr, NULL);

    if (cache != NULL) {
        int32_t idx = findByHeader(cache, header);
        if (idx >= 0) {
            const uint8_t *uuid = cache->entries[idx].uuid;
            atomic_store(&g_cache_ptr, cache);
            return uuid;
        }

        // Not in cache — populate and add maintaining sorted order
        KSBinaryImageRange newEntry;
        if (populateCacheEntry(header, NULL, &newEntry)) {
            const uint8_t *uuid = newEntry.uuid;
            if (cache->count < maxCacheEntries()) {
                insertSortedCacheEntry(cache, &newEntry);
            }
            atomic_store(&g_cache_ptr, cache);
            return uuid;
        }

        atomic_store(&g_cache_ptr, cache);
        return NULL;
    } else {
        // Cache busy — fall back to uncached population
        KSBinaryImageRange entry;
        if (populateCacheEntry(header, NULL, &entry)) {
            return entry.uuid;
        }
        return NULL;
    }
}

// MARK: - Owned image metadata

// Copy a Mach-O header of either width and its load commands out of `task`. Returns the malloc'd
// load-command buffer (caller frees) or NULL when the header cannot be read, its magic is unknown,
// or its sizes are implausible. Every read goes through vm_read_overwrite, so an image unmapped
// underneath us (a concurrent final dlclose) yields NULL instead of a fault.
static uint8_t *copyLoadCommandsFromTask(task_t task, uintptr_t loadAddress, struct mach_header_64 *outHeader)
{
    // Read only the common prefix first: a 32-bit header is shorter than mach_header_64.
    memset(outHeader, 0, sizeof(*outHeader));
    if (!ksmem_copySafelyFromTask(task, (const void *)loadAddress, outHeader, sizeof(struct mach_header))) {
        return NULL;
    }
    uintptr_t headerSize;
    if (outHeader->magic == MH_MAGIC_64) {
        headerSize = sizeof(struct mach_header_64);
    } else if (outHeader->magic == MH_MAGIC) {
        headerSize = sizeof(struct mach_header);
    } else {
        return NULL;
    }

    // Sanity-bound the load-command region before allocating, in case the header is garbage.
    const uint32_t maxCmdsSize = 1u << 20;  // 1 MiB; real load-command regions are a few KiB.
    if (outHeader->sizeofcmds == 0 || outHeader->sizeofcmds > maxCmdsSize) {
        return NULL;
    }
    uint8_t *cmds = malloc(outHeader->sizeofcmds);
    if (cmds == NULL) {
        return NULL;
    }
    if (!ksmem_copySafelyFromTask(task, (const void *)(loadAddress + headerSize), cmds, (int)outHeader->sizeofcmds)) {
        free(cmds);
        return NULL;
    }
    return cmds;
}

bool ksbic_copyTaskImage(task_t task, uintptr_t loadAddress, KSBinaryImage *outImage, uint8_t *outUUID)
{
    if (outImage == NULL || outUUID == NULL) {
        return false;
    }
    struct mach_header_64 header;
    uint8_t *cmds = copyLoadCommandsFromTask(task, loadAddress, &header);
    if (cmds == NULL) {
        return false;
    }
    KSMachOImageGeometry geometry;
    if (header.magic == MH_MAGIC_64) {
        walkLoadCommands64(cmds, header.sizeofcmds, header.ncmds, loadAddress, NULL, 0, &geometry);
    } else {
        walkLoadCommands32(cmds, header.sizeofcmds, header.ncmds, loadAddress, NULL, 0, &geometry);
    }
    // geometry.uuid points into cmds; copy it out before the buffer goes away.
    const bool hasUUID = geometry.uuid != NULL;
    if (hasUUID) {
        memcpy(outUUID, geometry.uuid, 16);
    }
    free(cmds);
    if (geometry.textSize == 0) {
        return false;
    }

    // Same values ksdl_binaryImageForHeader records for a live local image.
    outImage->address = loadAddress;
    outImage->vmAddress = geometry.textVMAddr;
    outImage->vmAddressSlide = (uint64_t)geometry.slide;
    outImage->size = geometry.textSize;
    outImage->uuid = hasUUID ? outUUID : NULL;
    outImage->cpuType = header.cputype;
    outImage->cpuSubType = header.cpusubtype;
    outImage->majorVersion = geometry.dylibVersion >> 16;
    outImage->minorVersion = (geometry.dylibVersion >> 8) & 0xff;
    outImage->revisionVersion = geometry.dylibVersion & 0xff;
    return true;
}
