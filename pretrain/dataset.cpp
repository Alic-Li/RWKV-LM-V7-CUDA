#include "dataset.hpp"
#include "compat/jsonl.hpp"
#include "compat/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rwkv {
namespace {
class Mapping {
public:
  explicit Mapping(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      throw std::runtime_error("cannot open " + path);
    struct stat info{};
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_size <= 0) {
      close(fd);
      throw std::runtime_error("expected a nonempty regular file: " + path);
    }
    size = info.st_size;
    const auto addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED)
      throw std::runtime_error("cannot mmap " + path);
    data = static_cast<const char *>(addr);
  }
  ~Mapping() { munmap(const_cast<char *>(data), size); }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;

  template <class T> T read(std::uint64_t off) const {
    if (off > size || sizeof(T) > size - off)
      throw std::runtime_error("truncated dataset or index");
    T result;
    std::memcpy(&result, data + off, sizeof(T));
    return result;
  }

  const char *data;
  std::uint64_t size;
};

struct Span {
  std::uint64_t offset, bytes, tokens;
};
struct Row {
  std::vector<int> tokens;
  std::vector<std::int64_t> labels;
};

Row tokenize(OptimizedTrieTokenizer &tokenizer, const std::string &line) {
  Row row;
  std::vector<std::size_t> eos;
  for (const auto &segment : rwkv7_state_tuning::parse_jsonl_row(line)) {
    const auto tokens = tokenizer.encode(segment.text);
    row.tokens.insert(row.tokens.end(), tokens.begin(), tokens.end());
    if (segment.train)
      row.labels.insert(row.labels.end(), tokens.begin(), tokens.end());
    else
      row.labels.insert(row.labels.end(), tokens.size(), -1);
    if (segment.end_target == 0)
      eos.push_back(row.tokens.size());
  }
  for (const auto pos : eos) {
    if (pos >= row.labels.size() || row.labels[pos] != -1)
      throw std::runtime_error(
          "end_target requires a subsequent context-only token");
    row.labels[pos] = 0;
  }
  auto end = row.labels.size();
  while (end > 1 && row.labels[end - 1] == -1)
    --end;
  row.labels.resize(end);
  row.tokens.resize(end);
  return row;
}

std::uint64_t modified(const std::string &path) {
  return std::filesystem::last_write_time(path).time_since_epoch().count();
}

struct JsonIndexHeader {
  std::uint64_t magic, source_size, source_mtime, vocab_size, vocab_mtime;
};

std::unique_ptr<Mapping> json_index(const Mapping &source,
                                    const std::string &path,
                                    const std::string &vocab,
                                    OptimizedTrieTokenizer &tokenizer) {
  const auto index = path + ".rwkvidx";
  const JsonIndexHeader expected{
      0x31305844494b5752ULL, source.size, modified(path),
      std::filesystem::file_size(vocab), modified(vocab)};
  if (std::filesystem::exists(index)) {
    auto old = std::make_unique<Mapping>(index);
    if (old->size >= sizeof(expected) &&
        (old->size - sizeof(expected)) % sizeof(Span) == 0 &&
        std::memcmp(old->data, &expected, sizeof(expected)) == 0)
      return old;
  }

  const auto temp = index + ".tmp." + std::to_string(getpid());
  try {
    std::ofstream out(temp, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write(reinterpret_cast<const char *>(&expected), sizeof(expected));
    std::uint64_t line_number = 0;
    for (std::uint64_t begin = 0; begin < source.size;) {
      ++line_number;
      const auto *newline = static_cast<const char *>(
          std::memchr(source.data + begin, '\n', source.size - begin));
      const auto end =
          newline ? std::uint64_t(newline - source.data) : source.size;
      const std::string line(source.data + begin, end - begin);
      if (line.find_first_not_of(" \t\r") != std::string::npos) {
        try {
          const auto row = tokenize(tokenizer, line);
          if (row.tokens.size() >= 2) {
            const Span span{begin, end - begin, row.tokens.size()};
            out.write(reinterpret_cast<const char *>(&span), sizeof(span));
          }
        } catch (const std::exception &e) {
          throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                   ": " + e.what());
        }
      }
      begin = end + (newline != nullptr);
    }
    if (std::filesystem::file_size(path) != expected.source_size ||
        modified(path) != expected.source_mtime)
      throw std::runtime_error("JSONL changed while building its index");
    out.close();
    std::filesystem::rename(temp, index);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temp, ignored);
    throw;
  }
  return std::make_unique<Mapping>(index);
}
} // namespace

struct Dataset::Source {
  virtual ~Source() = default;
  virtual void read(std::size_t item, std::uint64_t offset, int count,
                    int *input, std::int64_t *labels) const = 0;

  void add(Span span, int context) {
    if (span.tokens < 2)
      return;
    const auto windows = 1 + (span.tokens - 2) / context;
    if (windows > UINT64_MAX - total)
      throw std::runtime_error("too many dataset windows");
    spans.push_back(span);
    total += windows;
    ends.push_back(total);
  }

  std::vector<Span> spans;
  std::vector<std::uint64_t> ends;
  std::uint64_t total = 0;
};

namespace {
class JsonSource final : public Dataset::Source {
public:
  JsonSource(const std::string &path, const std::string &vocab, int context)
      : file_(path), tokenizer_(vocab) {
    if (!tokenizer_.inited())
      throw std::runtime_error("cannot load tokenizer: " + vocab);
    const auto index = json_index(file_, path, vocab, tokenizer_);
    for (auto i = sizeof(JsonIndexHeader); i < index->size; i += sizeof(Span)) {
      const auto span = index->read<Span>(i);
      if (span.offset > file_.size || span.bytes > file_.size - span.offset ||
          span.tokens > span.bytes)
        throw std::runtime_error("invalid JSONL index entry");
      add(span, context);
    }
  }
  void read(std::size_t item, std::uint64_t off, int n, int *input,
            std::int64_t *labels) const override {
    const auto span = spans[item];
    const auto row =
        tokenize(tokenizer_, std::string(file_.data + span.offset, span.bytes));
    if (row.tokens.size() != span.tokens)
      throw std::runtime_error("JSONL contents do not match the index");
    std::copy_n(row.tokens.data() + off, n, input);
    std::copy_n(row.labels.data() + off + 1, n, labels);
  }

private:
  Mapping file_;
  mutable OptimizedTrieTokenizer tokenizer_;
};

class BinSource final : public Dataset::Source {
public:
  BinSource(const std::string &prefix, const std::string &mask_path,
            int context)
      : bin_(prefix + ".bin"),
        mask_(mask_path.empty() ? prefix + ".mask" : mask_path) {
    const Mapping idx(prefix + ".idx");
    if (idx.size < 34 || std::memcmp(idx.data, "MMIDIDX\0\0", 9) ||
        idx.read<std::uint64_t>(9) != 1)
      throw std::runtime_error("unsupported binidx index");
    dtype_ = idx.read<std::uint8_t>(17);
    if (dtype_ < 1 || dtype_ > 8 || dtype_ == 6 || dtype_ == 7)
      throw std::runtime_error("binidx needs an integer token dtype");
    width_ = std::array<int, 9>{0, 1, 1, 2, 4, 8, 0, 0, 2}[dtype_];
    if (bin_.size % width_ || mask_.size != bin_.size / width_)
      throw std::runtime_error("binidx mask must have one byte per token");
    const auto items = idx.read<std::uint64_t>(18);
    const auto docs = idx.read<std::uint64_t>(26);
    if (items > (idx.size - 34) / 12 ||
        docs != (idx.size - 34 - items * 12) / 8 ||
        34 + items * 12 + docs * 8 != idx.size)
      throw std::runtime_error("invalid binidx metadata size");
    std::uint64_t next = 0;
    for (std::uint64_t i = 0; i < items; ++i) {
      const auto count = idx.read<std::int32_t>(34 + i * 4);
      const auto offset = idx.read<std::uint64_t>(34 + items * 4 + i * 8);
      if (count < 0 || offset != next || offset > bin_.size ||
          std::uint64_t(count) > (bin_.size - offset) / width_)
        throw std::runtime_error("invalid binidx item bounds");
      next += std::uint64_t(count) * width_;
    }
    if (next != bin_.size)
      throw std::runtime_error("binidx index does not cover the token file");

    const auto doc_base = 34 + items * 12;
    if (docs < 2 || idx.read<std::uint64_t>(doc_base) != 0 ||
        idx.read<std::uint64_t>(doc_base + (docs - 1) * 8) != items)
      throw std::runtime_error(
          "binidx document boundaries must cover every item");
    std::uint64_t start_item = 0, begin = 0;
    for (std::uint64_t i = 1; i < docs; ++i) {
      const auto doc = idx.read<std::uint64_t>(doc_base + i * 8);
      if (doc < start_item || doc > items)
        throw std::runtime_error("invalid binidx document boundary");
      const auto end = doc == items
                           ? bin_.size
                           : idx.read<std::uint64_t>(34 + items * 4 + doc * 8);
      add({begin, end - begin, (end - begin) / width_}, context);
      start_item = doc;
      begin = end;
    }
  }
  void read(std::size_t item, std::uint64_t offset, int n, int *input,
            std::int64_t *labels) const override {
    const auto start = spans[item].offset / width_ + offset;
    input[0] = token(start);
    for (int i = 0; i < n; ++i) {
      const auto next = start + i + 1;
      const int value = token(next);
      if (i + 1 < n)
        input[i + 1] = value;
      switch (mask_.read<std::uint8_t>(next)) {
      case 0:
        labels[i] = -1;
        break;
      case 1:
        labels[i] = value;
        break;
      case 2:
        labels[i] = 0;
        break;
      default:
        throw std::runtime_error("expected binidx mask byte 0, 1, or 2");
      }
    }
  }

private:
  int token(std::uint64_t i) const {
    const auto offset = i * width_;
    std::int64_t value;
    switch (dtype_) {
    case 1:
      value = bin_.read<std::uint8_t>(offset);
      break;
    case 2:
      value = bin_.read<std::int8_t>(offset);
      break;
    case 3:
      value = bin_.read<std::int16_t>(offset);
      break;
    case 4:
      value = bin_.read<std::int32_t>(offset);
      break;
    case 5:
      value = bin_.read<std::int64_t>(offset);
      break;
    default:
      value = bin_.read<std::uint16_t>(offset);
      break;
    }
    if (value < 0 || value > INT32_MAX)
      throw std::runtime_error("invalid binidx token ID");
    return value;
  }
  Mapping bin_, mask_;
  int dtype_, width_;
};
} // namespace

Dataset::Dataset(const std::string &path, const std::string &sidecar,
                 const std::string &vocab_path, int context, int vocab)
    : context_(context), vocab_(vocab) {
  const auto ext = std::filesystem::path(path).extension();
  if (ext == ".jsonl") {
    if (!sidecar.empty())
      throw std::invalid_argument("--mask is only used with binidx");
    source_ = std::make_unique<JsonSource>(path, vocab_path, context_);
  } else {
    const auto prefix =
        ext == ".idx" || ext == ".bin" ? path.substr(0, path.size() - 4) : path;
    source_ = std::make_unique<BinSource>(prefix, sidecar, context_);
  }
  if (!size())
    throw std::runtime_error("dataset contains no next-token samples");
}
Dataset::~Dataset() = default;
std::uint64_t Dataset::size() const { return source_->total; }

void Dataset::sample(std::uint64_t window, int *input,
                     std::int64_t *labels) const {
  if (window >= size())
    throw std::out_of_range("dataset window index");
  const auto item =
      std::upper_bound(source_->ends.begin(), source_->ends.end(), window) -
      source_->ends.begin();
  const auto off = (window - (item ? source_->ends[item - 1] : 0)) * context_;
  const int n =
      std::min<std::uint64_t>(source_->spans[item].tokens - 1 - off, context_);
  source_->read(item, off, n, input, labels);
  for (int i = 0; i < n; ++i) {
    if (input[i] < 0 || input[i] >= vocab_ || labels[i] >= vocab_)
      throw std::runtime_error("dataset token exceeds checkpoint vocabulary");
  }
  std::fill(input + n, input + context_, 0);
  std::fill(labels + n, labels + context_, -1);
}
} // namespace rwkv
