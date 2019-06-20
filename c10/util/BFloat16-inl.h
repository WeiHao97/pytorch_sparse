#pragma once

#include <c10/macros/Macros.h>
#include <limits>
#include <inttypes.h>
#include <iostream>

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE BFloat16::BFloat16(float value) {
  val_ = value;
  //val_ = 5;
  //std::cout << "bits_from_f32: in: " << value << std::endl;
  //std::cout << "bits_from_f32: out: " << std::hex << val_ << std::endl;
  //uint32_t temp;
  //std::memcpy(&temp, &value, sizeof(value));
  //std::cout << "bits_from_f32: out2: " << std::hex << temp << std::endl;
}

/// Implicit conversions

inline C10_HOST_DEVICE BFloat16::operator float() const {
  //std::cout << "float(): val_" << std::hex << val_ << std::endl;
  //float res = detail::f32_from_bits(val_);
  //std::cout << "float(): res: " << res;
  return val_;
}

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::BFloat16> {
  public:
    static constexpr bool is_signed = true;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr c10::BFloat16 lowest() {
      return at::BFloat16(0xFBFF, at::BFloat16::from_bits());
    }
    static constexpr c10::BFloat16 max() {
      return at::BFloat16(0x7BFF, at::BFloat16::from_bits());
    }
};

} // namespace std
