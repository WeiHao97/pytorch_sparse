#include <torch/csrc/jit/jit_log.h>
#include <c10/util/Exception.h>
#include <caffe2/utils/string_utils.h>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>

namespace torch {
namespace jit {

JitLoggingLevels jit_log_level() {
  static const char* c_log_level = std::getenv("PYTORCH_JIT_LOG_LEVEL");
  static const JitLoggingLevels log_level = c_log_level
      ? static_cast<JitLoggingLevels>(std::atoi(c_log_level))
      : JitLoggingLevels::OFF;
  return log_level;
}

static std::string level_and_tid(JitLoggingLevels level) {
  std::stringstream ss;
  ss << "[" << level << " T";
  ss << std::this_thread::get_id();
  ss << "]: ";
  return ss.str();
}

std::once_flag file_filter_initialized;
std::vector<std::regex> file_regexes;

bool apply_file_regex(const char* filename) {
  static const char* c_file_filter = getenv("PYTORCH_JIT_LOG_FILE_FILTER");
  if (!c_file_filter) {
    return true;
  }

  std::call_once(file_filter_initialized, [&]() {
    const auto str_regexes = caffe2::split(':', c_file_filter);

    for (const auto& str_regex : str_regexes) {
      file_regexes.emplace_back(std::regex(str_regex));
    }
  });

  for (const auto& regex : file_regexes) {
    if (std::regex_search(filename, regex)) {
      return true;
    }
  }

  return false;
}

std::string add_jit_log_prefix(
    JitLoggingLevels level,
    const std::string& in_str) {
  static const char* enable_suffix =
      std::getenv("PYTORCH_JIT_LOG_ENABLE_PREFIX");
  if (!enable_suffix) {
    return in_str;
  }

  std::stringstream in_ss(in_str);
  std::stringstream out_ss(in_str);
  std::string line;
  auto prefix = level_and_tid(level);

  while (std::getline(in_ss, line, '\n')) {
    out_ss << prefix << line << std::endl;
  }

  return out_ss.str();
}

std::ostream& operator<<(std::ostream& out, JitLoggingLevels level) {
  switch (level) {
    case JitLoggingLevels::OFF:
      TORCH_INTERNAL_ASSERT("UNREACHABLE");
    case JitLoggingLevels::GRAPH_DUMP:
      out << "JIT_GRAPH_DUMP";
      break;
    case JitLoggingLevels::INFO:
      out << "JIT_INFO";
      break;
    case JitLoggingLevels::DEBUG:
      out << "JIT_DEBUG";
      break;
    default:
      TORCH_INTERNAL_ASSERT("Invalid level");
  }

  return out;
}

} // namespace jit
} // namespace torch
