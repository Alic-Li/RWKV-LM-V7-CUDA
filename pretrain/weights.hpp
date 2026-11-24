#pragma once
#include "rwkv/io/pth_tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rwkv {
struct Shape {
  int layers, channels, ffn, vocab;
  int decay, alpha, value, gate;
};

struct Weight {
  std::string name;
  std::vector<int> dims;
  std::size_t size() const;
};
using Weights = std::vector<Weight>;
Weights weight_layout(const Shape &shape);
void check_shape(const Shape &shape);

class Checkpoint {
public:
  explicit Checkpoint(const std::string &path);
  Shape shape() const;
  std::vector<float> read(const Weight &) const;

private:
  llm_infer::PthArchive archive_;
  std::vector<llm_infer::TensorRecord> records_;
};
void save_weights(
    const std::string &path, const Weights &layout,
    const std::function<std::vector<std::uint16_t>(const Weight &)> &get);
} // namespace rwkv
