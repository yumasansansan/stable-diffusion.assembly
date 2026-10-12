// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The phases of the kernels, timed for the profile (profile.cc) when
// SDASM_PROFILE is set: each thread counts the nanoseconds it spends in each
// phase of an operation -- packing the weights, packing or transforming the
// other operand, the products of the micro-kernel, the softmax of attention,
// the transform of Winograd's outputs, waiting for the other threads -- and
// hands them to the profile when the operation ends, which writes, for each
// kind of operation, how its time divides between them. A scope costs two
// reads of the steady clock, and is put only around stretches of many
// microseconds; without SDASM_PROFILE it costs a test of a flag.

#ifndef SDASM_PHASES_H
#define SDASM_PHASES_H

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace sdasm {

enum class Phase { pack_weights, pack_inputs, product, softmax, transform_out, epilogue, wait, count };

const char * phase_name(Phase p);

bool phases_enabled();

// The nanoseconds of this thread in each phase since the last take, which it
// zeroes.
using Phase_times = std::array<std::uint64_t, static_cast<std::size_t>(Phase::count)>;
void phases_take(Phase_times & nanoseconds);
void phase_add(Phase p, std::uint64_t nanoseconds);

inline std::uint64_t phase_clock() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

class Phase_scope {
public:
    explicit Phase_scope(Phase p) : phase_(p), start_(phases_enabled() ? phase_clock() : 0) {}
    Phase_scope(const Phase_scope &) = delete;
    Phase_scope & operator=(const Phase_scope &) = delete;
    ~Phase_scope() {
        if (start_ != 0) {
            phase_add(phase_, phase_clock() - start_);
        }
    }

private:
    Phase phase_;
    std::uint64_t start_;
};

}  // namespace sdasm

#endif
