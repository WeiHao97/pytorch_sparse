#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at { namespace native {

// Methods

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, Scalar value) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cuda::_th_masked_fill_(self, mask, value);
  } else {
    return legacy::cuda::_th_masked_fill_bool_(self, mask, value);
  }
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cuda::_th_masked_fill_(self, mask, value);
  } else {
    return legacy::cuda::_th_masked_fill_bool_(self, mask, value);
  }
}

Tensor & masked_scatter__cuda(Tensor& self, const Tensor & mask, const Tensor & source) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cuda::_th_masked_scatter_(self, mask, source);
  } else {
    return legacy::cuda::_th_masked_scatter_bool_(self, mask, source);
  }
}

Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cuda::_th_masked_select(self, mask);
  } else {
    return legacy::cuda::_th_masked_select_bool(self, mask);
  }
}

Tensor & gather_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cuda::_th_gather_out(result, self, dim, index);
}

Tensor gather_cuda(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cuda::_th_gather(self, dim, index);
}

Tensor & lt_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.lt received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_lt_byte_out(result, self, other);
  } else {
    return legacy::cuda::_th_lt_out(result, self, other);
  }
}

Tensor & lt_scalar_out_cuda(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.lt received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_lt_byte_out(result, self, value);
  } else {
    return legacy::cuda::_th_lt_out(result, self, value);
  }
}

Tensor & le_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.le received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_le_byte_out(result, self, other);
  } else {
    return legacy::cuda::_th_le_out(result, self, other);
  }
}

Tensor & le_scalar_out_cuda(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.le received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_le_byte_out(result, self, value);
  } else {
    return legacy::cuda::_th_le_out(result, self, value);
  }
}

Tensor & gt_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.gt received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_gt_byte_out(result, self, other);
  } else {
    return legacy::cuda::_th_gt_out(result, self, other);
  }
}

Tensor & gt_scalar_out_cuda(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.gt received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_gt_byte_out(result, self, value);
  } else {
    return legacy::cuda::_th_gt_out(result, self, value);
  }
}

Tensor & ge_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ge received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_ge_byte_out(result, self, other);
  } else {
    return legacy::cuda::_th_ge_out(result, self, other);
  }
}

Tensor & ge_scalar_out_cuda(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ge received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_ge_byte_out(result, self, value);
  } else {
    return legacy::cuda::_th_ge_out(result, self, value);
  }
}

Tensor & eq_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.eq received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_eq_byte_out(result, self, other);
  } else {
    return legacy::cuda::_th_eq_out(result, self, other);
  }
}

Tensor & eq_scalar_out_cuda(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.eq received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_eq_byte_out(result, self, value);
  } else {
    return legacy::cuda::_th_eq_out(result, self, value);
  }
}

Tensor & ne_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ne received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_ne_byte_out(result, self, other);
  } else {
    return legacy::cuda::_th_ne_out(result, self, other);
  }
}

Tensor & ne_scalar_out_cuda(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ne received a result with dtype torch.uint8, this behavior is now deprecated," \
            "please use a result with dtype torch.bool instead.");
    return legacy::cuda::_th_ne_byte_out(result, self, value);
  } else {
    return legacy::cuda::_th_ne_out(result, self, value);
  }
}
}} // namespace at::native
