/*
 * SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of stable-diffusion.assembly. The SPDX lines name its
 * copyright holder and its license, in the machine-readable form of the REUSE
 * specification: the GNU General Public License, version 3 or any later version
 * (LICENSES/GPL-3.0-or-later.txt).
 *
 * What ggml's CPU backend calls, with this project's patch
 * (patches/ggml/0001-...): before it computes an operation itself, it offers
 * the operation here, and when it plans a graph, it asks how much of its work
 * buffer the operations taken here need. The patch declares these two functions
 * itself, so that ggml needs no include path of this project's; this header is
 * for the project's own sources, and says the same.
 */

#ifndef SDASM_SDASM_H
#define SDASM_SDASM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;
struct ggml_compute_params;
struct ggml_tensor;

/* Computes <node> on the thread params->ith of params->nth and returns true, or
 * returns false, on every thread alike, when the node is not one this project
 * takes; ggml then computes it as it would have. Every thread of the pool calls
 * it for every node, as ggml's own kernels are called. */
bool sdasm_compute_forward(const struct ggml_compute_params * params, struct ggml_tensor * node);

/* Computes node <index> of <graph> on the thread params->ith of params->nth,
 * with the nodes after it that change its output in place (fusion.h), and
 * returns how many nodes of the graph that was, or returns 0, on every thread
 * alike, when the node is not one this project takes. The graph's own loop over
 * its nodes calls it, in place of sdasm_compute_forward(). */
int sdasm_compute_nodes(const struct ggml_compute_params * params, const struct ggml_cgraph * graph, int index);

/* Sets *size to the bytes of the work buffer that <node> needs when it is
 * computed here by <n_threads> threads and returns true, or returns false when
 * the node is not one this project takes. */
bool sdasm_work_size(const struct ggml_tensor * node, int n_threads, size_t * size);

/* Forgets what the kernels keep packed from memory in [begin, begin + size),
 * which is about to change (store.h); ggml calls it when it sets the data of a
 * tensor, or clears or frees a buffer. */
void sdasm_forget(const void * begin, size_t size);

/* The profile of the graphs, with SDASM_PROFILE in the environment
 * (profile.cc): the first thread of the pool times each node and each graph
 * with sdasm_profile_now(), when sdasm_profile_enabled() says so, and hands the
 * times in nanoseconds here. A node computed together with the <fused> nodes
 * after it is one time. */
bool sdasm_profile_enabled(void);
uint64_t sdasm_profile_now(void);
void sdasm_profile_node(const struct ggml_cgraph * graph, const struct ggml_tensor * node, int fused,
                        uint64_t nanoseconds);
void sdasm_profile_graph(const struct ggml_cgraph * graph, uint64_t nanoseconds);

#ifdef __cplusplus
}
#endif

#endif
