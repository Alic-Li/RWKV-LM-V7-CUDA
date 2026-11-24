#pragma once

#include <string>
#include <vector>

namespace rwkv7_state_tuning {

// One tokenization unit of a sample. Tokens produced from a segment with
// train == false are context only: they are fed forward but never used as
// loss targets.
struct TextSegment {
  std::string text;
  bool train = true;
  // Optional training label after this segment; never inserted into inputs.
  int end_target = -1;
};

std::vector<TextSegment> parse_jsonl_row(const std::string &line);

} // namespace rwkv7_state_tuning
