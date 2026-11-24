#pragma once
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
    CUDA(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable));
    return p;
  }
  void deallocate(T *p, std::size_t) { cudaFreeHost(p); }
  template <class U> bool operator==(const PinnedAllocator<U> &) const {
    return true;
  }
  template <class U> bool operator!=(const PinnedAllocator<U> &) const {
    return false;
  }
};

template <typename T> class Buffer {
public:
  Buffer() = default;
  Buffer(cudaStream_t stream, std::size_t n) : stream_(stream) {
    if (n)
      CUDA(cudaMallocAsync(&ptr_, n * sizeof(T), stream_));
  }
  ~Buffer() {
    if (ptr_)
      cudaFreeAsync(ptr_, stream_);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), stream_(other.stream_) {}
  Buffer &operator=(Buffer &&other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(stream_, other.stream_);
    return *this;
  }
  T *get() const { return ptr_; }

private:
  T *ptr_ = nullptr;
  cudaStream_t stream_ = nullptr;
};
} // namespace rwkv
