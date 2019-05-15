#pragma once
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <memory>
#include <iostream>
namespace torch {
namespace jit {

// a range of a shared string 'file_' with functions to help debug by highlight
// that
// range.
struct CAFFE2_API SourceRange {
  SourceRange(
      std::shared_ptr<std::string> file,
      size_t start,
      size_t end,
      c10::optional<std::shared_ptr<SourceRange>> context = c10::nullopt)
      : file_(std::move(file)), start_(start), end_(end), context_(context) {}
  explicit SourceRange(std::string string_range)
      : file_(std::make_shared<std::string>(std::move(string_range))),
        start_(0),
        end_(file_->size()) {}

  const std::string text() const {
    return file().substr(start(), end() - start());
  }
  size_t size() const {
    return end() - start();
  }
  static const size_t CONTEXT = 10;
  void highlight(std::ostream& out) const;
  const std::string& file() const {
    return *file_;
  }
  const std::shared_ptr<std::string>& file_ptr() const {
    return file_;
  }
  size_t start() const {
    return start_;
  }
  size_t end() const {
    return end_;
  }
  std::string str() const {
    std::stringstream ss;
    highlight(ss);
    return ss.str();
  }
  const c10::optional<std::shared_ptr<SourceRange>> context() const {
    return context_;
  }
  std::string wrapException(
      const std::exception& e,
      const std::string& additional = "") {
    std::stringstream msg;
    msg << "\n" << e.what() << ":\n";
    if (!additional.empty()) {
      msg << additional << ":\n";
    }
    highlight(msg);
    return msg.str();
  }
  void wrapAndRethrowException(
      const std::exception& e,
      const std::string& additional = "") {
    throw std::runtime_error(wrapException(e, additional));
  }

 private:
  std::shared_ptr<std::string> file_;
  size_t start_;
  size_t end_;
  c10::optional<std::shared_ptr<SourceRange>> context_;
};

inline std::ostream& operator<<(std::ostream& out, const SourceRange& range) {
  range.highlight(out);
  return out;
}

} // namespace jit
} // namespace torch
