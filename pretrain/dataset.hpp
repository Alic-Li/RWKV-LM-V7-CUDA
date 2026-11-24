#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace rwkv {
class Dataset {
public:
  struct Source;
  Dataset(const std::string &path, const std::string &sidecar,
          const std::string &tokenizer, int context, int vocab);
  ~Dataset();
  std::uint64_t size() const;
  // Writes exactly context positions. Masked labels have value -1.
  void sample(std::uint64_t window, int *input, std::int64_t *labels) const;

private:
  std::unique_ptr<Source> source_;
  int context_, vocab_;
};
} // namespace rwkv
