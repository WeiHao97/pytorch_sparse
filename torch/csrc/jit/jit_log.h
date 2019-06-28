#pragma once

#include <string>

namespace torch {
namespace jit {

enum class JitLoggingLevels {
  OFF,
  GRAPH_DUMP,
  INFO,
  DEBUG,
};

JitLoggingLevels jit_log_level();

std::ostream& operator<<(std::ostream& out, JitLoggingLevels level);

std::string add_jit_log_prefix(
    JitLoggingLevels level,
    const std::string& in_str);

bool apply_file_regex(const char* filename);

#define JIT_LOG(level, ...)                                                   \
  if (jit_log_level() != JitLoggingLevels::OFF && jit_log_level() >= level && \
      apply_file_regex(__FILE__)) {                                           \
    std::cerr << add_jit_log_prefix(level, ::c10::str(__VA_ARGS__));          \
  }

// use JIT_GRAPH_DUMP for dumping graphs after optimization passes
#define JIT_GRAPH_DUMP(MSG, G) \
  JIT_LOG(JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());
// use JIT_INFO for reporting graph transformations (i.e. node deletion,
// constant folding, CSE)
#define JIT_INFO(...) JIT_LOG(JitLoggingLevels::INFO, __VA_ARGS__);
// use JIT_DEBUG to provide information useful for debugging a particular opt
// pass
#define JIT_DEBUG(...) JIT_LOG(JitLoggingLevels::DEBUG, __VA_ARGS__);

} // namespace jit
} // namespace torch
