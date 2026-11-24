#pragma once
#include <charconv>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace rwkv {
template <class T> T integer(const std::string &s) {
  T value{};
  const auto r = std::from_chars(s.data(), s.data() + s.size(), value);
  if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
    throw std::invalid_argument("invalid integer: " + s);
  return value;
}

class Options {
public:
  Options(int argc, char **argv, int begin) {
    for (int i = begin; i < argc; i += 2) {
      std::string name = argv[i];
      if (name.compare(0, 2, "--") || i + 1 >= argc ||
          !args_.emplace(name, argv[i + 1]).second)
        throw std::invalid_argument("expected a unique option and value: " +
                                    name);
    }
  }
  std::string take(const std::string &name, std::string fallback = {}) {
    auto node = args_.extract(name);
    return node.empty() ? std::move(fallback) : std::move(node.mapped());
  }
  template <class T> T number(const std::string &name, T fallback) {
    return integer<T>(take(name, std::to_string(fallback)));
  }
  double real(const std::string &name, double fallback) {
    auto s = take(name);
    if (s.empty())
      return fallback;
    std::size_t end;
    auto result = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(result))
      throw std::invalid_argument("invalid number for " + name);
    return result;
  }
  void finish() const {
    if (!args_.empty())
      throw std::invalid_argument("unknown option " + args_.begin()->first);
  }

private:
  std::map<std::string, std::string> args_;
};
} // namespace rwkv
