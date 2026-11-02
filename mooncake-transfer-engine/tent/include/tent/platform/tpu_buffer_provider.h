// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// C ABI between Mooncake's built-in TPU adapter (libmooncake_tpu_pjrt.so, see
// tent/plugins/tpu) and a TPU BUFFER PROVIDER: the library that owns the
// framework-side device buffers and performs the actual HBM<->host DMA, e.g.
// google/tpu-sync for JAX arrays.
//
// Why a provider: reaching a framework array's device buffer requires
// framework/runtime internals (for JAX, jaxlib and XLA headers at the exact
// installed version), which Mooncake does not build against. The provider is
// built with the framework and exposes only this C table; Mooncake's adapter
// keeps the address bookkeeping TENT needs (fake addresses, interior pointers).
//
// Discovery mirrors tent/device_plugin.h: the provider library exports one
// symbol, MC_TPU_BUFFER_PROVIDER_ENTRY, which fills a versioned function table.
//
// Handles: the provider hands out opaque non-zero mc_tpu_buffer_t values (for
// JAX, one per addressable shard of an array) through its own framework API.
// Once a handle is passed to the adapter, the adapter owns it and calls
// `release` exactly once when it is no longer needed.
//
// Threading: every function may be called concurrently from multiple threads,
// including for the same handle (TENT stages several chunks at once).
//
// Versions: version 2 appends dma_map and dma_unmap to the version 1 table.
// The adapter still binds version 1 providers, without DMA mapping.

#ifndef TENT_PLATFORM_TPU_BUFFER_PROVIDER_H_
#define TENT_PLATFORM_TPU_BUFFER_PROVIDER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MC_TPU_BUFFER_PROVIDER_ABI_VERSION 2u
#define MC_TPU_BUFFER_PROVIDER_ENTRY "mc_tpu_buffer_provider_get"

// Opaque provider-owned handle to one contiguous device buffer. 0 is invalid.
typedef uint64_t mc_tpu_buffer_t;

typedef struct mc_tpu_buffer_provider {
    // ABI version the provider implemented when filling this table.
    uint32_t abi_version;
    // Human-readable provider name for logs, e.g. "tpu-sync".
    const char *name;

    // Size in bytes of the buffer's on-device representation, or 0 if the
    // handle is invalid. Offsets passed to the copy functions index into this
    // representation, which may be tiled; both ends of a transfer must
    // therefore use buffers with identical shape, dtype and layout.
    size_t (*size)(mc_tpu_buffer_t buffer);

    // Local device ordinal of the buffer (the N in TENT's "tpu:N"), or -1.
    int (*device_index)(mc_tpu_buffer_t buffer);

    // NUMA node closest to the buffer's device, or -1 if unknown.
    int (*numa_node)(mc_tpu_buffer_t buffer);

    // Blocking copies of bytes [offset, offset + len) of the buffer to/from
    // host memory. The adapter guarantees the range lies inside the buffer
    // and passes `offset` and `len` through unchanged: it never rounds or
    // widens a range. A provider may accept only some offsets (tpu-sync
    // requires multiples of 32 bytes, with any length and host pointer) and
    // must fail the others without copying; TENT then fails the transfer.
    // Return 0 on success, non-zero on failure.
    int (*copy_to_host)(mc_tpu_buffer_t buffer, size_t offset, void *host_dst,
                        size_t len);
    int (*copy_from_host)(mc_tpu_buffer_t buffer, size_t offset,
                          const void *host_src, size_t len);

    // Drops the adapter's reference to `buffer`.
    void (*release)(mc_tpu_buffer_t buffer);

    // --- Version 2 ---------------------------------------------------------

    // Registers host memory [host, host + len) with the TPU DMA engine (for
    // PJRT, PjRtClient::DmaMap), so that copies into and out of it run as
    // direct DMA. TENT maps each host staging pool it allocates: `host` and
    // `len` are multiples of the page size, and the range overlaps no live
    // mapping. Return 0 on success, non-zero on failure; TENT then uses the
    // memory unmapped, only with slower copies.
    int (*dma_map)(void *host, size_t len);

    // Undoes the dma_map that returned success for the range starting at
    // `host`. TENT calls it when it releases the pool, after every copy into
    // or out of it has returned. Return 0 on success, non-zero on failure.
    int (*dma_unmap)(void *host);
} mc_tpu_buffer_provider_t;

// Signature of MC_TPU_BUFFER_PROVIDER_ENTRY. Fills `*out` with the table of
// ABI version `requested_version`, or of an older version the provider
// supports, sets `out->abi_version` to the version filled, and returns 0;
// returns non-zero if it can serve neither. Never fill a version newer than
// `requested_version`: `*out` may be only as large as that version's table.
// Every function pointer of the filled version must be non-null. The adapter
// asks for MC_TPU_BUFFER_PROVIDER_ABI_VERSION and, if refused, for version 1.
typedef int (*mc_tpu_buffer_provider_get_fn)(uint32_t requested_version,
                                             mc_tpu_buffer_provider_t *out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENT_PLATFORM_TPU_BUFFER_PROVIDER_H_
