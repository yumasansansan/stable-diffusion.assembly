// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The nodes after an operation that it computes along with it. A chain is the
// operations, in the order of the graph, that follow a node and change its
// output in place, element by element -- the bias a convolution or a linear
// layer adds after its product, the scale and the shift a normalisation
// multiplies and adds after it, the SiLU after those -- with nothing between
// them but views. Such a chain is applied to the output of the operation while
// it is still in the cache, a stretch at a time, in the order of the graph, as
// the nodes it stands in for would apply it -- in single precision, with SiLU
// and GELU computed as scalar_math.h computes them, a little closer to the
// exact functions than ggml's. It is a chain in place: the memory its result is
// in is the operation's own output, which the graph's allocator gave it while
// every input of the operation was alive, so nothing it writes can be anything
// that another part of the operation still reads -- but for the tensor of a
// residual connection that ends it (Epilogue::redirect), which is checked.

#ifndef SDASM_FUSION_H
#define SDASM_FUSION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "memory.h"

struct ggml_tensor;

namespace sdasm {

// The nodes of a graph, as the graph holds them (sdasm.h).
using Graph_nodes = std::span<ggml_tensor * const>;

// The nodes of a graph, from the array and the number that ggml hands over:
// the one place that takes the number on trust (sdasm_compute_nodes() and
// sdasm_profile_graph() check it is above 0 and the array is there).
Graph_nodes graph_nodes(ggml_tensor * const * const nodes, int n_nodes);

// Node i of a graph, 0 <= i < nodes.size().
inline ggml_tensor * node_at(Graph_nodes nodes, int i) {
    return nodes[static_cast<std::size_t>(i)];
}

struct Epilogue_step {
    enum class Kind { add, mul, silu };
    Kind kind = Kind::add;
    // The other operand of an addition or a multiplication, its elements: its
    // element for the output element (i0, i1, i2, i3) at the sum of ik * nb[k],
    // nb in elements, where a dimension it is broadcast along has nb[k] = 0.
    Elements<const float> operand;
    std::array<std::int64_t, 4> nb{};
};

struct Epilogue {
    static constexpr int kMaxSteps = 6;
    int count = 0;
    std::array<Epilogue_step, kMaxSteps> steps{};
    std::array<const ggml_tensor *, kMaxSteps> operands{};  // the other operand of each step, if any
    int last_node = -1;    // the index of the last node of the chain
    std::int64_t ne0 = 0;  // the length of a row of the output, for SiLU
    // The tensor the operation writes in place of its own, when the chain ends
    // in an addition that is not in place -- a residual connection, which adds
    // the operation's output and the input of the block it ends into a tensor
    // of its own: the operation writes its output there and adds to it, and
    // its own tensor and those of the chain's nodes are not written at all,
    // which match_chain() makes sure nothing after reads.
    ggml_tensor * redirect = nullptr;
    // That tensor holds the other operand of the addition: the allocator
    // gave the sum the memory of the operand, which it no longer needs. The
    // operation then begins its sum from it, rather than adding it at the
    // end, and the addition is not a step of the chain; the chain's steps
    // before it are all additions, and in any order give the same sum.
    bool from_redirect = false;

    bool empty() const { return count == 0 && !from_redirect; }

    // Applies the chain to the elements of the output `out` holds, which lie
    // one after another along dimension 0, the first at output coordinates
    // (i0, i1, i2, i3).
    void apply(Elements<float> out, std::int64_t i0, std::int64_t i1, std::int64_t i2, std::int64_t i3) const;
    // The same for positions of channel i2 of image i3, which run on across
    // the rows of the image, the first at `position` (y * ne0 + x).
    void apply_positions(Elements<float> out, std::int64_t position, std::int64_t i2, std::int64_t i3) const;
};

// The chain after node `index` of the graph that an operation can apply:
// additions and multiplications by a tensor broadcast to the output's shape, in
// place, and SiLU if `silu` allows it. With `positions`, for an operation that
// applies it to the positions of an image as they run on across its rows, an
// operand broadcast along dimension 0 is broadcast along dimension 1 too, and
// one that is not lies in rows as the output does.
//
// With `redirect`, the chain may end in an addition that is not in place
// (Epilogue::redirect), when the memory allows it (redirect_apart()).
Epilogue match_chain(Graph_nodes nodes, int index, bool silu, bool positions, bool redirect = false);

// Whether the operation of node may write its chain's redirect: the memory of
// the tensor written is none of what the operation or the chain reads -- the
// allocator may have given it, for one, the memory of the operation's input,
// free once the operation has read it, as it would be without the fusion.
bool redirect_apart(const ggml_tensor * const node, const Epilogue & chain);

// The gated GELU that stable-diffusion.cpp's GEGLU takes after a linear layer:
// the layer's output, with the chain after it (its bias), split into halves
// along dimension 0 by two views, x and the gate; the gate made contiguous,
// its GELU taken in place, and x multiplied by that into a tensor of its own,
// `out`. The layer computes the product, the chain and the gated GELU at once,
// into `out`, and the nodes between are not computed: their tensors are never
// written, which match_gated_gelu() makes sure nothing after reads.
struct Gated_gelu {
    ggml_tensor * out = nullptr;
    int last_node = -1;  // the index of out's node
};

Gated_gelu match_gated_gelu(Graph_nodes nodes, int index, const Epilogue & chain);

// out = x GELU(gate), element by element, of out's elements: GELU in single
// precision, by its definition with tanh, as 0.5 x (1 + tanh(y)) =
// x / (1 + exp(-2 y)), y = sqrt(2 / pi) (x + 0.044715 x^3) -- what ggml
// computes, without the FP16 table it reads it from, which rounds the argument
// and the result to FP16.
void gated_gelu_run(Elements<float> out, Elements<const float> x, Elements<const float> gate);

}  // namespace sdasm

#endif
