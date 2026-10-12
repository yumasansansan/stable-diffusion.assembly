// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The operations of ggml that this project computes, each with what it takes
// of them (supported), what of the work buffer it needs (work_size) and the
// computation itself, which every thread of the pool calls (compute). hooks.cc
// hands them the nodes.

#ifndef SDASM_OPS_H
#define SDASM_OPS_H

#include <cstddef>

struct ggml_compute_params;
struct ggml_tensor;

namespace sdasm {

struct Epilogue;

// GGML_OP_CONV_2D, the direct convolution: by the matrix product, with the
// columns of the image that im2col would make packed straight from the image.
bool conv2d_supported(const ggml_tensor * const node);
std::size_t conv2d_work_size(const ggml_tensor * const node, int n_threads);
void conv2d_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain);

// GGML_OP_CONV_2D of a kernel of three by three at a stride of one, by
// Winograd's F(2x2, 3x3), which takes it before the direct convolution.
bool winograd_supported(const ggml_tensor * const node);
std::size_t winograd_work_size(const ggml_tensor * const node, int n_threads);
void winograd_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain);

// GGML_OP_MUL_MAT of a matrix of weights, the same for every matrix of the
// other operand, as the linear layers have it.
bool mul_mat_supported(const ggml_tensor * const node);
std::size_t mul_mat_work_size(const ggml_tensor * const node, int n_threads);
void mul_mat_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain);
// The same with the gated GELU after it (fusion.h) into out, when the memory of
// out allows it.
bool mul_mat_gated_supported(const ggml_tensor * const node, const Epilogue & chain, const ggml_tensor * const out);
void mul_mat_compute_gated(const ggml_compute_params * const params, const ggml_tensor * const node,
                           const Epilogue & chain, const ggml_tensor * const out);

// GGML_OP_FLASH_ATTN_EXT without a mask, as stable-diffusion.cpp's attention
// without flash attention computes it: the scores, their softmax and the sum of
// the values by it, with the scores of a block of queries kept in the cache.
bool attention_supported(const ggml_tensor * const node);
std::size_t attention_work_size(const ggml_tensor * const node, int n_threads);
void attention_compute(const ggml_compute_params * const params, const ggml_tensor * const node);

// GGML_OP_GROUP_NORM and GGML_OP_NORM of single-precision tensors, with the
// chain after them (norm.cc).
bool group_norm_supported(const ggml_tensor * const node);
void group_norm_compute(const ggml_compute_params * const params, const ggml_tensor * const node,
                        const Epilogue & chain);
bool norm_supported(const ggml_tensor * const node);
void norm_compute(const ggml_compute_params * const params, const ggml_tensor * const node, const Epilogue & chain);

}  // namespace sdasm

#endif
