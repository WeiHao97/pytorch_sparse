#include <gtest/gtest.h>
#include <c10/util/constexpr_string_functions.h>

namespace strlen_test {
  static_assert(0 == c10::util::strlen(""), "");
  static_assert(1 == c10::util::strlen("a"), "");
  static_assert(10 == c10::util::strlen("0123456789"), "");
}

namespace starts_with_test {
  static_assert(c10::util::starts_with("house", ""), "");
  static_assert(c10::util::starts_with("house", "h"), "");
  static_assert(c10::util::starts_with("house", "ho"), "");
  static_assert(c10::util::starts_with("house", "hou"), "");
  static_assert(c10::util::starts_with("house", "hous"), "");
  static_assert(c10::util::starts_with("house", "house"), "");

  static_assert(!c10::util::starts_with("house", "b"), "");
  static_assert(!c10::util::starts_with("house", "bouse"), "");
  static_assert(!c10::util::starts_with("house", "houseb"), "");
}

namespace strequal_test {
  static_assert(c10::util::strequal("", ""), "");
  static_assert(c10::util::strequal("a", "a"), "");
  static_assert(c10::util::strequal("ab", "ab"), "");
  static_assert(c10::util::strequal("0123456789", "0123456789"), "");

  static_assert(!c10::util::strequal("", "0"), "");
  static_assert(!c10::util::strequal("0", ""), "");
  static_assert(!c10::util::strequal("0123", "012"), "");
  static_assert(!c10::util::strequal("012", "0123"), "");
  static_assert(!c10::util::strequal("0123456789", "0123556789"), "");
  static_assert(!c10::util::strequal("0123456789", "0123456788"), "");
}

namespace skip_until_first_of_test {
  static_assert(c10::util::strequal("tring", c10::util::skip_until_first_of("string", 's')), "");
  static_assert(c10::util::strequal("ring", c10::util::skip_until_first_of("string", 't')), "");
  static_assert(c10::util::strequal("ing", c10::util::skip_until_first_of("string", 'r')), "");
  static_assert(c10::util::strequal("ng", c10::util::skip_until_first_of("string", 'i')), "");
  static_assert(c10::util::strequal("g", c10::util::skip_until_first_of("string", 'n')), "");
  static_assert(c10::util::strequal("", c10::util::skip_until_first_of("string", 'g')), "");

  static_assert(c10::util::strequal("", c10::util::skip_until_first_of("string", 'a')), "");
  static_assert(c10::util::strequal("", c10::util::skip_until_first_of("", 'a')), "");
}

namespace csv_contains_test {
  static_assert(c10::util::csv_contains2("", ""), "");
  static_assert(!c10::util::csv_contains2("", "a"), "");
  static_assert(!c10::util::csv_contains2("a", ""), "");
  static_assert(!c10::util::csv_contains2("a,bc", ""), "");

  static_assert(c10::util::csv_contains2("a,bc,d", "a"), "");
  static_assert(c10::util::csv_contains2("a,bc,d", "bc"), "");
  static_assert(c10::util::csv_contains2("a,bc,d", "d"), "");
  static_assert(!c10::util::csv_contains2("a,bc,d", "e"), "");
  static_assert(!c10::util::csv_contains2("a,bc,d", ""), "");
}
