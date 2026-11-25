#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace rwkv {
inline void check_cuda(cudaError_t status, const char *expr) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expr) + ": " +
                             cudaGetErrorString(status));
}
#define CUDA(call) ::rwkv::check_cuda((call), #call)

// Pinned host batches keep copies asynchronous across the NCCL ranks.
template <class T> struct PinnedAllocator {
  using value_type = T;
  PinnedAllocator() = default;
  template <class U> PinnedAllocator(const PinnedAllocator<U> &) {}
  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    T *p;
    CUDA(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault));
    return p;
  }
  void deallocate(T *p, std::size_t) { (void)cudaFreeHost(p); }
  template <class U> bool operator==(const PinnedAllocator<U> &) const {
    return true;
  }
  template <class U> bool operator!=(const PinnedAllocator<U> &) const {
    return false;
  }
};

// Owning buffers are created during initialization. Training temporaries use
// non-owning typed views into the model's compute-stream arena.
inline std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b && a > SIZE_MAX / b)
    throw std::length_error("device buffer size overflow");
  return a * b;
}
inline std::size_t checked_sum(std::size_t a, std::size_t b) {
  if (a > SIZE_MAX - b)
    throw std::length_error("device buffer size overflow");
  return a + b;
}

template <typename T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(std::size_t n) {
    if (n)
      CUDA(cudaMalloc(&ptr_, checked_product(n, sizeof(T))));
  }
  ~Buffer() {
    if (ptr_)
      (void)cudaFree(ptr_);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
  Buffer &operator=(Buffer &&other) noexcept {
    std::swap(ptr_, other.ptr_);
    return *this;
  }
  T *get() const { return ptr_; }
private:
  T *ptr_ = nullptr;
};

template <class T> struct View {
  T *ptr = nullptr;
  T *get() const { return ptr; }
};

class Arena {
public:
  explicit Arena(std::size_t bytes) : storage_(bytes), capacity_(bytes) {}
  template <class T> View<T> take(std::size_t n) {
    const auto start = checked_sum(offset_, 255) & ~std::size_t(255);
    const auto end = checked_sum(start, checked_product(n, sizeof(T)));
    if (end > capacity_)
      throw std::length_error("compute arena capacity exceeded");
    offset_ = end;
    return {reinterpret_cast<T *>(storage_.get() + start)};
  }
  std::size_t mark() const { return offset_; }
  void rewind(std::size_t mark) { offset_ = mark; }

private:
  Buffer<unsigned char> storage_;
  std::size_t capacity_, offset_ = 0;
};
} // namespace rwkv
