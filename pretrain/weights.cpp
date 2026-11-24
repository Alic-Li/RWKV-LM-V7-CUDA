#include "weights.hpp"

#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unistd.h>

namespace rwkv {
namespace {
template <class T> T checked(llm_infer::Result<T> result) {
  if (!result.ok())
    throw std::runtime_error(result.status().message());
  return std::move(result.value());
}
using Bytes = std::vector<std::uint8_t>;

void word(Bytes &p, std::uint32_t x) {
  for (int i = 0; i < 4; ++i)
    p.push_back(x >> (i * 8));
}
void string(Bytes &p, const std::string &s) {
  p.push_back('X');
  word(p, s.size());
  p.insert(p.end(), s.begin(), s.end());
}
void integer(Bytes &p, std::uint64_t value) {
  if (value <= INT32_MAX) {
    p.push_back('J');
    word(p, value);
    return;
  }
  p.insert(p.end(), {0x8a, 8}); // LONG1 with a non-negative signed payload.
  word(p, value);
  word(p, value >> 32);
}
void append(Bytes &p, const std::string &s) {
  p.insert(p.end(), s.begin(), s.end());
}

void verify_archive(archive *a, int status) {
  if (status != ARCHIVE_OK) {
    const char *message = archive_error_string(a);
    throw std::runtime_error(message ? message : "libarchive write failed");
  }
}
void archive_file(archive *a, const std::string &name, const void *data,
                  std::size_t size) {
  std::unique_ptr<archive_entry, decltype(&archive_entry_free)> entry(
      archive_entry_new(), archive_entry_free);
  if (!entry)
    throw std::bad_alloc();
  archive_entry_set_pathname(entry.get(), name.c_str());
  archive_entry_set_filetype(entry.get(), AE_IFREG);
  archive_entry_set_perm(entry.get(), 0644);
  archive_entry_set_size(entry.get(), size);
  verify_archive(a, archive_write_header(a, entry.get()));
  if (archive_write_data(a, data, size) != static_cast<la_ssize_t>(size))
    throw std::runtime_error("failed to write PTH entry " + name);
  verify_archive(a, archive_write_finish_entry(a));
}
} // namespace

std::size_t Weight::size() const {
  std::size_t n = 1;
  for (int dim : dims) {
    if (dim <= 0 ||
        n > std::numeric_limits<std::size_t>::max() / dim / sizeof(float))
      throw std::runtime_error("invalid weight size for " + name);
    n *= dim;
  }
  return n;
}

void check_shape(const Shape &s) {
  if (s.layers < 1 || s.channels < 64 || s.channels % 64 || s.ffn < 32 ||
      s.ffn % 32 || s.vocab < 2 || s.decay < 1 || s.alpha < 1 || s.value < 1 ||
      s.gate < 1)
    throw std::invalid_argument(
        "expected x070 dimensions: channels multiple of 64, FFN multiple of "
        "32, positive low-rank dimensions");
}

Weights weight_layout(const Shape &s) {
  check_shape(s);
  int c = s.channels;
  Weights layout{{"emb.weight", {s.vocab, c}}};
  const auto norm = [&](const std::string &name) {
    layout.push_back({name + ".weight", {c}});
    layout.push_back({name + ".bias", {c}});
  };
  for (int i = 0; i < s.layers; ++i) {
    const auto b = "blocks." + std::to_string(i) + ".";
    if (!i)
      norm(b + "ln0");
    norm(b + "ln1");
    norm(b + "ln2");
    const auto a = b + "att.";
    for (auto name : {"x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0",
                      "v0", "k_k", "k_a"})
      layout.push_back({a + name, {1, 1, c}});
    for (const auto &[prefix, rank] : std::vector<std::pair<char, int>>{
             {'w', s.decay}, {'a', s.alpha}, {'v', s.value}, {'g', s.gate}}) {
      layout.push_back({a + prefix + "1", {c, rank}});
      layout.push_back({a + prefix + "2", {rank, c}});
    }
    layout.push_back({a + "r_k", {c / 64, 64}});
    for (auto name : {"receptance", "key", "value", "output"})
      layout.push_back({a + name + ".weight", {c, c}});
    norm(a + "ln_x");
    layout.push_back({b + "ffn.x_k", {1, 1, c}});
    layout.push_back({b + "ffn.key.weight", {s.ffn, c}});
    layout.push_back({b + "ffn.value.weight", {c, s.ffn}});
  }
  norm("ln_out");
  layout.push_back({"head.weight", {s.vocab, c}});
  return layout;
}

Checkpoint::Checkpoint(const std::string &path)
    : archive_(checked(llm_infer::PthArchive::open(path, true))),
      records_(checked(llm_infer::parse_pth_tensor_records(archive_))) {}

Shape Checkpoint::shape() const {
  const auto dim = [&](const std::string &name, unsigned axis) {
    const auto it = std::find_if(records_.begin(), records_.end(),
                                 [&](const auto &r) { return r.name == name; });
    if (it == records_.end() || axis >= it->shape.size() ||
        it->shape[axis] <= 0 || it->shape[axis] > INT32_MAX)
      throw std::runtime_error("missing or invalid tensor dimension: " + name);
    return int(it->shape[axis]);
  };
  int layers = 0;
  for (const auto &record : records_) {
    if (record.name.compare(0, 7, "blocks.") == 0 &&
        record.name.find(".ln1.weight") != std::string::npos)
      ++layers;
  }
  Shape s{layers,
          dim("emb.weight", 1),
          dim("blocks.0.ffn.key.weight", 0),
          dim("emb.weight", 0),
          dim("blocks.0.att.w1", 1),
          dim("blocks.0.att.a1", 1),
          dim("blocks.0.att.v1", 1),
          dim("blocks.0.att.g1", 1)};
  auto layout = weight_layout(s);
  if (records_.size() != layout.size())
    throw std::runtime_error("checkpoint tensor count does not match x070");
  return s;
}

std::vector<float> Checkpoint::read(const Weight &weight) const {
  const auto it =
      std::find_if(records_.begin(), records_.end(),
                   [&](const auto &r) { return r.name == weight.name; });
  if (it == records_.end() || it->shape.size() != weight.dims.size() ||
      !std::equal(it->shape.begin(), it->shape.end(), weight.dims.begin()))
    throw std::runtime_error("missing or incompatible checkpoint tensor: " +
                             weight.name);
  return checked(llm_infer::load_tensor_select(archive_, *it, false, false, true)).values;
}

void save_weights(
    const std::string &path, const Weights &layout,
    const std::function<std::vector<std::uint16_t>(const Weight &)> &get) {
  const auto temp = path + ".tmp." + std::to_string(getpid());
  try {
    std::unique_ptr<archive, decltype(&archive_write_free)> a(
        archive_write_new(), archive_write_free);
    if (!a)
      throw std::bad_alloc();
    verify_archive(a.get(), archive_write_set_format_zip(a.get()));
    verify_archive(a.get(), archive_write_set_format_option(
                                a.get(), "zip", "compression", "store"));
    verify_archive(a.get(), archive_write_set_bytes_in_last_block(a.get(), 1));
    verify_archive(a.get(), archive_write_open_filename(a.get(), temp.c_str()));
    Bytes p{0x80, 2, '}', '('};
    for (std::size_t i = 0; i < layout.size(); ++i) {
      const auto &t = layout[i];
      const auto key = std::to_string(i);
      const auto n = t.size();
      string(p, t.name);
      append(p, "ctorch._utils\n_rebuild_tensor_v2\n((");
      string(p, "storage");
      append(p, "ctorch\nBFloat16Storage\n");
      string(p, key);
      string(p, "cpu");
      integer(p, n);
      append(p, "tQ");
      integer(p, 0);
      p.push_back('(');
      for (int dim : t.dims)
        integer(p, dim);
      append(p, "t(");
      auto stride = n;
      for (int dim : t.dims) {
        stride /= dim;
        integer(p, stride);
      }
      append(p, "t");
      p.insert(p.end(), {0x89, '}', 't', 'R'});
      const auto data = get(t);
      if (data.size() != n)
        throw std::runtime_error("weight size mismatch: " + t.name);
      archive_file(a.get(), "archive/data/" + key, data.data(),
                   data.size() * sizeof(data[0]));
    }
    append(p, "u.");
    archive_file(a.get(), "archive/data.pkl", p.data(), p.size());
    archive_file(a.get(), "archive/byteorder", "little", 6);
    archive_file(a.get(), "archive/version", "3\n", 2);
    verify_archive(a.get(), archive_write_close(a.get()));
    std::filesystem::rename(temp, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temp, ignored);
    throw;
  }
}
} // namespace rwkv
