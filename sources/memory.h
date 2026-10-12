// SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of stable-diffusion.assembly. The SPDX lines name its
// copyright holder and its license, in the machine-readable form of the REUSE
// specification: the GNU General Public License, version 3 or any later version
// (LICENSES/GPL-3.0-or-later.txt).
//
// The memory the operations read and write, typed. ggml hands an operation the
// data of its tensors behind void *, and a work buffer of bytes; here they are
// spans of the elements they hold (Elements), and every address an operation
// computes is an element of one of those spans. No pointer is cast to another
// type than its own, none is made from an integer, and none is made into one.
//
//  - A tensor's data holds elements of the type its ggml type names: float of
//    GGML_TYPE_F32, the bits of an IEEE half of GGML_TYPE_F16. elements_of<T>()
//    takes them as T, the one conversion from void * there is, after checking
//    that the tensor's type says T, that its data is aligned for T and that
//    each of its strides is a whole number of T's -- what an operation's
//    supported() checks first (holds<T>()), so that it never takes a tensor it
//    could not.
//  - The work buffer is bytes, which each operation lays out into arrays of the
//    types it needs (Work_layout): those all its threads share, and each
//    thread's own. A Work_buffer begins their lifetimes as those types
//    (std::start_lifetime_as_array), on every thread, before the threads first
//    meet (Work_queue, gemm.h); each thread reaches its own arrays and the
//    shared ones after that, and nothing else of the buffer.
//  - The store's buffers (store.h) are allocated as the type their layout
//    holds.
//
// Where elements are reached through a span, every index is checked against its
// bounds in a build with checks (SDASM_CHECKS: Debug, and the sanitizers' and
// the emulated preset's builds, which define _GLIBCXX_ASSERTIONS); a Release
// build checks what it is handed once, where it takes it -- a tensor's type,
// alignment and strides, the size of the work buffer -- and leaves the indices
// unchecked, as the C++ library leaves those of its spans.
//
// The kernels written by hand take pointers, as the System V calling
// convention passes them: data() of a span, the bounds of which their callers
// have checked. elements_at() goes the other way, from a pointer and a count
// to a span, for the few places that are handed a pointer -- ggml's arrays, and
// the C++ kernels, which take the same pointers the kernels written by hand
// take -- and each use of it is marked as such (-Wunsafe-buffer-usage).

#ifndef SDASM_MEMORY_H
#define SDASM_MEMORY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

struct ggml_compute_params;
struct ggml_tensor;

#if !defined(NDEBUG) || defined(_GLIBCXX_ASSERTIONS)
#define SDASM_CHECKS 1
#else
#define SDASM_CHECKS 0
#endif

namespace sdasm {

// Stops the program, with what was found and where: for what no input of the
// program can make happen, a mistake of this project's.
[[noreturn]] void fail(const char * const what, const char * const file, int line);

#define SDASM_REQUIRE(condition)                                  \
    do {                                                          \
        if (!(condition)) {                                       \
            ::sdasm::fail(#condition, __FILE__, __LINE__);        \
        }                                                         \
    } while (false)

// Checked in a build with checks, and not in a Release build.
#if SDASM_CHECKS
#define SDASM_CHECK(condition) SDASM_REQUIRE(condition)
#else
#define SDASM_CHECK(condition) static_cast<void>(0)
#endif

// A run of elements in memory, indexed by the signed numbers the operations
// compute in -- ggml's ne and nb are std::int64_t -- every index checked in a
// build with checks: a std::span of them, which owns nothing, as Clang's
// analysis of lifetimes knows (gsl::Pointer), and tells of one that outlives
// what it points into.
template <class T>
class [[gsl::Pointer(T)]] Elements {
public:
    using element_type = T;

    constexpr Elements() = default;
    constexpr Elements(std::span<T> s [[clang::lifetimebound]]) : span_(s) {}
    template <class U>
        requires std::is_convertible_v<U (*)[], T (*)[]>
    constexpr Elements(Elements<U> other) : span_(other.span()) {}
    template <class U, std::size_t N>
        requires std::is_convertible_v<U (*)[], T (*)[]>
    constexpr Elements(std::array<U, N> & a [[clang::lifetimebound]]) : span_(a) {}
    template <class U, std::size_t N>
        requires std::is_convertible_v<const U (*)[], T (*)[]>
    constexpr Elements(const std::array<U, N> & a [[clang::lifetimebound]]) : span_(a) {}

    constexpr std::int64_t size() const { return static_cast<std::int64_t>(span_.size()); }
    constexpr bool empty() const { return span_.empty(); }

    constexpr T & operator[](std::int64_t i) const {
        SDASM_CHECK(i >= 0 && i < size());
        return span_[static_cast<std::size_t>(i)];
    }

    // The elements from `offset` to the end, and `count` of them from `offset`.
    constexpr Elements from(std::int64_t offset) const {
        SDASM_CHECK(offset >= 0 && offset <= size());
        return Elements(span_.subspan(static_cast<std::size_t>(offset)));
    }
    constexpr Elements part(std::int64_t offset, std::int64_t count) const {
        SDASM_CHECK(offset >= 0 && count >= 0 && offset <= size() && count <= size() - offset);
        return Elements(span_.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(count)));
    }
    constexpr Elements first(std::int64_t count) const { return part(0, count); }

    // The address of the first element, for the kernels written by hand, which
    // read and write as far as their callers have checked.
    constexpr T * data() const { return span_.data(); }
    constexpr std::span<T> span() const { return span_; }

    constexpr auto begin() const { return span_.begin(); }
    constexpr auto end() const { return span_.end(); }

private:
    std::span<T> span_;
};

// The elements of `count` T's at p: memory that the caller is handed as a
// pointer and vouches for. Each use is marked.
template <class T>
[[clang::unsafe_buffer_usage]] Elements<T> elements_at(T * p, std::int64_t count) {
    SDASM_REQUIRE(count >= 0 && (p != nullptr || count == 0));
#pragma clang unsafe_buffer_usage begin
    return Elements<T>(std::span<T>(p, static_cast<std::size_t>(count)));
#pragma clang unsafe_buffer_usage end
}

// Copies the elements of src to dst, and sets every element of dst to x.
template <class T>
void copy(Elements<T> dst, Elements<const T> src) {
    SDASM_CHECK(dst.size() >= src.size());
    for (std::int64_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i];
    }
}
template <class T>
void fill_with(Elements<T> dst, T x) {
    for (T & e : dst) {
        e = x;
    }
}

// The arguments of a program, from main()'s argc and argv: the one place a
// program takes them as the C library hands them over.
std::vector<std::string_view> program_arguments(int argc, char ** const argv);

// ---------------------------------------------------------------- tensors

// The element type of a ggml type: float of GGML_TYPE_F32, std::uint16_t, the
// bits of an IEEE half, of GGML_TYPE_F16.
template <class T>
bool type_is(const ggml_tensor * const t);
template <>
bool type_is<float>(const ggml_tensor * const t);
template <>
bool type_is<std::uint16_t>(const ggml_tensor * const t);

// Whether p is aligned to `alignment` bytes, a power of two.
bool aligned(void * const p, std::size_t alignment);

// Whether the bytes [a, a + a_size) and [b, b + b_size) share any: each the
// memory of a tensor, or of a buffer, from its first byte to one past its last,
// which ggml hands over as an address and a size. The addresses are compared,
// as std::less orders them, and never read.
bool overlaps(const void * const a, std::size_t a_size, const void * const b, std::size_t b_size);

// The data of tensor t, its bytes from its first element to its last, as ggml
// counts them (ggml_nbytes()); and its strides, in bytes.
std::size_t data_bytes(const ggml_tensor * const t);
std::int64_t stride_bytes(const ggml_tensor * const t, int dimension);
// The number of elements of tensor t along a dimension, 0 to 3.
std::int64_t extent(const ggml_tensor * const t, int dimension);
void * data_address(const ggml_tensor * const t);

// Whether tensor t holds elements of type T that elements_of<T>() can take: of
// the type, aligned for it, each stride a whole number of them.
template <class T>
bool holds(const ggml_tensor * t) {
    if (!type_is<std::remove_const_t<T>>(t) || !aligned(data_address(t), alignof(T))) {
        return false;
    }
    for (int k = 0; k < 4; ++k) {
        if (stride_bytes(t, k) % static_cast<std::int64_t>(sizeof(T)) != 0) {
            return false;
        }
    }
    return true;
}

// The elements of tensor t, which holds<T>(t), from its first to its last:
// the element (i0, i1, i2, i3) at i0 nb[0] + i1 nb[1] + i2 nb[2] + i3 nb[3] of
// them, nb the strides in elements (stride_of()).
template <class T>
Elements<T> elements_of(const ggml_tensor * t) {
    SDASM_REQUIRE(holds<T>(t));
    // The tensor's type says what its data holds: the void * is ggml's, and
    // is taken here as what it points to.
    T * const first = static_cast<T *>(data_address(t));
#pragma clang unsafe_buffer_usage begin
    return elements_at(first, static_cast<std::int64_t>(data_bytes(t) / sizeof(T)));
#pragma clang unsafe_buffer_usage end
}

// The stride of dimension k of tensor t, which holds<T>(t), in elements.
template <class T>
std::int64_t stride_of(const ggml_tensor * t, int k) {
    return stride_bytes(t, k) / static_cast<std::int64_t>(sizeof(T));
}

// ---------------------------------------------------------------- the work buffer

// The bytes the arrays of the work buffer are aligned to: a line of the cache,
// and of a vector of AVX-512.
inline constexpr std::size_t kWorkAlign = 64;

// An array of the work buffer: shared, for all threads together, or one for
// each thread; `index` is its place in the layout.
template <class T>
struct Shared {
    int index = -1;
    std::int64_t count = 0;
};
template <class T>
struct Own {
    int index = -1;
    std::int64_t count = 0;
};

// The arrays of an operation's work buffer, in the order they are declared:
// the shared ones first, each kWorkAlign-aligned, then those of each thread,
// thread after thread.
class Work_layout {
public:
    static constexpr int kMaxArrays = 16;

    template <class T>
    Shared<T> shared(std::int64_t count) {
        return Shared<T>{add(true, count, sizeof(T), &create<T>), count};
    }
    template <class T>
    Own<T> own(std::int64_t count) {
        return Own<T>{add(false, count, sizeof(T), &create<T>), count};
    }

    // The bytes of the buffer for n_threads threads, with room to align it.
    std::size_t bytes(int n_threads) const;

private:
    friend class Work_buffer;

    using Create = void * (*)(void *, std::size_t);
    template <class T>
    static void * create(void * p, std::size_t count) {
        static_assert(std::is_implicit_lifetime_v<T>);
        return std::start_lifetime_as_array<T>(p, count);
    }

    struct Array {
        bool shared = false;
        std::size_t offset = 0;  // from the shared arrays' start, or from a thread's
        std::size_t count = 0;
        std::size_t bytes = 0;
        Create create = nullptr;
    };

    int add(bool shared, std::int64_t count, std::size_t size, Create create);

    std::array<Array, kMaxArrays> arrays_{};
    int count_ = 0;
    std::size_t shared_bytes_ = 0;
    std::size_t thread_bytes_ = 0;
};

// The work buffer ggml hands an operation, laid out: the constructor, which
// every thread of the operation calls before the threads first meet, begins the
// lifetimes of the shared arrays and of its own thread's; after they meet
// (Work_queue::Work_queue(params, buffer)), operator[] hands them out.
class Work_buffer {
public:
    Work_buffer(const ggml_compute_params * const params, const Work_layout & layout);

    template <class T>
    Elements<T> operator[](Shared<T> a) const {
        return typed<T>(a.index, a.count);
    }
    template <class T>
    Elements<T> operator[](Own<T> a) const {
        return typed<T>(a.index, a.count);
    }

    // The threads have met since the arrays began their lifetimes.
    void met() { met_ = true; }

private:
    template <class T>
    Elements<T> typed(int index, std::int64_t count) const {
        SDASM_CHECK(met_ && index >= 0 && index < count_);
        const auto i = static_cast<std::size_t>(index);
        // An array began its lifetime as T, by create<T>(), and its address
        // was kept as void *: it is taken back as what it was.
        SDASM_REQUIRE(creates_[i] == &Work_layout::create<T>);
        T * const first = static_cast<T *>(arrays_[i]);
#pragma clang unsafe_buffer_usage begin
        return elements_at(first, count);
#pragma clang unsafe_buffer_usage end
    }

    std::array<void *, Work_layout::kMaxArrays> arrays_{};
    std::array<Work_layout::Create, Work_layout::kMaxArrays> creates_{};
    int count_ = 0;
    bool met_ = false;
};

}  // namespace sdasm

#endif
