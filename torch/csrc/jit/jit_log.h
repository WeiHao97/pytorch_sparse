#pragma once
#include <string>

// To enable logging please set(export) PYTORCH_JIT_LOG_LEVEL to
// one of the following logging levels: JIT_GRAPH_DUMP,
// JIT_INFO, JIT_DEBUG.
// The descriptions of the logging levels are below

namespace torch {
namespace jit {

enum class JitLoggingLevels {
  OFF,
  GRAPH_DUMP,
  INFO,
  DEBUG,
};

JitLoggingLevels jit_log_level();

std::string jit_log_prefix(JitLoggingLevels level, const std::string& in_str);

std::ostream& operator<<(std::ostream& out, JitLoggingLevels level);

#define JIT_LOG(level, ...)                                                   \
  if (jit_log_level() != JitLoggingLevels::OFF && jit_log_level() >= level) { \
    std::cerr << jit_log_prefix(level, ::c10::str(__VA_ARGS__));              \
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
