#include <c10/util/BFloat16.h>
#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout<BFloat16>::value,
    "c10::BFloat16 must be standard layout.");

std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  out << (float)value;
  return out;
}
} // namespace c10
