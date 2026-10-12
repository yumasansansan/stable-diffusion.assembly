// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The weights the kernels have packed, kept from one computation to the next.
// A model's weights are the same at every step of the sampler, and packing them
// -- widening FP16 to single precision and laying them out in the panels the
// micro-kernel reads, or transforming them for Winograd's convolution -- is a
// pass over all of them every step, an eighth of the UNet's time on the i7-1360P
// here; kept, it is done once.
//
// A packed form is kept for the tensor it was packed from, by its address, its
// name, its type and its shape, and for a layout of the kernels; it is filled
// by the threads of the operation that first needs it, as they would have
// filled their work buffer, and is ready for the operations after it. It is
// forgotten whenever the memory it was packed from may have changed: when ggml
// sets the data of a tensor there or clears or frees the buffer it lies in
// (patches/ggml/0004-...), and when a node of a graph writes there. Nothing is
// kept past the budget, SDASM_STORE_MB in the environment, and an operation
// whose weights are not kept packs them into its work buffer, as before.
//
// The budget is none unless the environment gives one: measured on the i7-1360P
// with SD 1.5, the kept weights, in single precision, are twice the bytes of the
// FP16 weights they come from, more than any cache holds, and reading them from
// memory every step costs what packing the FP16 ones did, with four gigabytes
// more of memory taken. The quantized weights (quant.h) are kept unless the
// environment says otherwise, up to kQuantizedBudget: they are fewer bytes than
// the FP16 ones, and quantizing them again for each computation costs a
// quarter and more of the time of the products of the layers of few rows.
// An operation's own packing reads FP16 and keeps its
// panels in the cache while its threads use them.

#ifndef SDASM_STORE_H
#define SDASM_STORE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

struct ggml_tensor;

namespace sdasm {

// The layouts of the kernels: the panels of the micro-kernel for convolutions,
// Winograd's transformed weights, the panels of the linear layers; on AMX
// (x86_64/amx.S), the tiles of B of the linear layers and the tiles of A of
// Winograd's transformed weights and of the convolutions', of AMX-BF16's form,
// and those of the linear layers and of the convolutions of AMX-FP16's; and
// the quantized panels of the linear layers, of Winograd's transformed weights
// and of the direct convolutions (quant.h).
enum class Layout {
    conv_rows,
    winograd,
    linear_columns,
    amx_linear,
    amx_winograd,
    amx_conv,
    amx_linear_fp16,
    amx_conv_fp16,
    quantized_linear,
    quantized_winograd,
    quantized_conv,
};

struct Packed {
    float * data = nullptr;
    std::size_t bytes = 0;
    std::atomic<int> state{0};

    // Ready to read: filled by the operation that first needed it.
    bool ready() const;
};

// The packed form of `tensor` in `layout`, of `bytes`, for every thread of an
// operation alike: one that is ready to read, one to fill, or none, when the
// budget has no room for it. The first thread to ask for one that is not kept
// makes it, and the others get the same; the operation marks it ready with
// publish() once all its threads have filled it. `variant` tells apart the
// forms of a layout that the tuning chooses among, the bits and the groups of
// the quantized ones.
std::shared_ptr<Packed> packed(const ggml_tensor * tensor, Layout layout, std::size_t bytes, std::uint64_t variant = 0);
void publish(const std::shared_ptr<Packed> & p);

// Forgets what was packed from memory in [begin, begin + size).
void forget(const void * begin, std::size_t size);

}  // namespace sdasm

#endif
