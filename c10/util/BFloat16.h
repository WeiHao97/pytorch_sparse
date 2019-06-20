#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

namespace c10 {

namespace detail {
  inline C10_HOST_DEVICE float f32_from_bits(float src) {
    float res = src;
    //float tmp = src;
    //tmp <<= 16;
    //std::memcpy(&res, &tmp, sizeof(tmp));
    return res;
  }

  inline C10_HOST_DEVICE float bits_from_f32(float src) {
    float res = src;
    //std::memcpy(&res, &src, sizeof(res));
    return res;
  }
} // namespace detail

struct alignas(4) BFloat16 {
  float val_;

  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  constexpr C10_HOST_DEVICE BFloat16(float bits, from_bits_t) : val_(bits){};
  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;
};

} // namespace c10


#include <c10/util/BFloat16-inl.h>
