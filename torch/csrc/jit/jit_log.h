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

#define JIT_LOG(level, ...)                                                   \
  if (jit_log_level() != JitLoggingLevels::OFF && jit_log_level() >= level) { \
    std::cerr << add_jit_log_prefix(level, ::c10::str(__VA_ARGS__));          \
  }

// print graph after every pass
#define JIT_GRAPH_DUMP(MSG, G) \
  JIT_LOG(JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());
// report evey graph transformation
#define JIT_INFO(...) JIT_LOG(JitLoggingLevels::INFO, __VA_ARGS__);
// details needed to debug a particular pass
#define JIT_DEBUG(...) JIT_LOG(JitLoggingLevels::DEBUG, __VA_ARGS__);

} // namespace jit
} // namespace torch
