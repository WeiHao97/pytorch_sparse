#include <ATen/native/ReduceOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#ifdef BUILD_NAMEDTENSOR
#include <ATen/NamedTensorUtils.h>
#endif

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <map>

namespace at {
namespace native {

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(std_var_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(norm_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(and_stub);
DEFINE_DISPATCH(or_stub);
DEFINE_DISPATCH(min_values_stub);
DEFINE_DISPATCH(max_values_stub);

static inline Tensor integer_upcast(const Tensor& self, optional<ScalarType> dtype) {
  ScalarType scalarType = self.scalar_type();
  ScalarType upcast_scalarType = dtype.value_or(at::isIntegralType(scalarType) ? ScalarType::Long : scalarType);
  return self.toType(upcast_scalarType);
}

using DimMask = TensorIterator::DimMask;

static DimMask make_dim_mask(IntArrayRef dims, int64_t ndim) {
  auto mask = DimMask();
  if (dims.empty()) {
    mask.flip();
  } else {
    for (int64_t dim : dims) {
      mask.set(maybe_wrap_dim(dim, ndim));
    }
  }
  return mask;
}

static void allocate_reduction_result(
    Tensor& result, const Tensor& self, DimMask mask, bool keepdim,
    ScalarType dtype)
{
  auto shape = DimVector(self.sizes());
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  if (result.defined()) {
    result.resize_(shape);
  } else {
    result = at::empty(shape, self.options().dtype(dtype));
  }
}

static Tensor review_reduce_result(const Tensor& result, int ndim, DimMask mask, bool keepdim) {
  if (keepdim) {
    return result;
  }
  auto shape = DimVector(result.sizes());
  auto stride = DimVector(result.strides());
  for (int dim = 0; dim < ndim; dim++) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }
  return result.as_strided(shape, stride);
}

static std::unique_ptr<TensorIterator> make_reduction(
    const char* name, Tensor& result, const Tensor& self, IntArrayRef dim,
    bool keepdim, ScalarType dtype)
{
  // check that result type and dtype match if provided
  TORCH_CHECK(
      !result.defined() || result.scalar_type() == dtype,
      name, ": provided dtype must match dtype of result. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype),
      ".");
  int64_t ndim = self.dim();
  auto mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result, self, mask, keepdim, dtype);
  auto viewed_result = review_reduce_result(result, ndim, mask, keepdim);

  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // not generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  if (self.scalar_type() == dtype ||
      (self.is_cuda() && self.scalar_type() == kHalf && dtype == kFloat)) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(dtype));
}

static std::unique_ptr<TensorIterator> make_reduction(
    const char* name, Tensor& result1, Tensor& result2, const Tensor& self, IntArrayRef dim,
    bool keepdim, ScalarType dtype)
{
  // check that result type and dtype match if provided
  for (const Tensor *t: {&result1, &result2}) {
    const Tensor& result = *t;
    TORCH_CHECK(
        !result.defined() || result.type().scalarType() == dtype,
        name, ": provided dtype must match dtype of result. Got ",
        toString(result.type().scalarType()),
        " and ",
        toString(dtype),
        ".");
  }

  int64_t ndim = self.dim();
  DimMask mask = make_dim_mask(dim, ndim);
  allocate_reduction_result(result1, self, mask, keepdim, dtype);
  auto viewed_result1 = review_reduce_result(result1, ndim, mask, keepdim);

  allocate_reduction_result(result2, self, mask, keepdim, dtype);
  auto viewed_result2 = review_reduce_result(result2, ndim, mask, keepdim);

  // special case for type promotion in mixed precision, improves computational
  // efficiency.
  // We don't generalize this to common mismatched input/output types to avoid cross
  // product of templated kernel launches.
  if (self.type().scalarType() == dtype ||
      (self.is_cuda() && self.type().scalarType() == kHalf && dtype == kFloat)) {
    return TensorIterator::reduce_op(viewed_result1, viewed_result2, self);
  }
  return TensorIterator::reduce_op(viewed_result1, viewed_result2, self.to(dtype));
}

Tensor cumsum(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  return at::_cumsum(integer_upcast(self, dtype), dim);
}

Tensor& cumsum_out(Tensor& result, const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumsum. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  return at::_cumsum_out(result, self.toType(result.scalar_type()), dim);
}

Tensor cumprod(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  return at::_cumprod(integer_upcast(self, dtype), dim);
}

Tensor& cumprod_out(Tensor& result, const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype) {
  // result type is favored over dtype; check that they match if provided (NumPy doesn't check)
  TORCH_CHECK(
      !dtype.has_value() || (result.scalar_type() == dtype.value()),
      "provided dtype must match dtype of result in cumprod. Got ",
      toString(result.scalar_type()),
      " and ",
      toString(dtype.value()),
      ".");
  return at::_cumprod_out(result, self.toType(result.scalar_type()), dim);
}


// ALL REDUCE #################################################################

static ScalarType get_dtype(Tensor& result, const Tensor& self, optional<ScalarType> dtype,
                            bool promote_integers=false) {
  if (dtype.has_value()) {
    return dtype.value();
  } else if (result.defined()) {
    return result.scalar_type();
  }
  ScalarType src_type = self.scalar_type();
  if (promote_integers && (at::isIntegralType(src_type) || src_type == ScalarType::Bool)) {
    return kLong;
  }
  return src_type;
}

Tensor& sum_out(Tensor& result, const Tensor& self, IntArrayRef dim,
                       bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("sum", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter->device_type(), *iter);
  }
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names_for_reduction(result, self, dim, keepdim);
#endif
  return result;
}

Tensor sum(const Tensor &self, c10::optional<ScalarType> dtype) {
  return at::native::sum(self, std::vector<int64_t>{}, false, dtype);
}
Tensor sum(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::sum_out(result, self, dim, keepdim, dtype);
}
#ifdef BUILD_NAMEDTENSOR
Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& sum_out(Tensor& result, const Tensor& self, DimnameList dim,
                bool keepdim, optional<ScalarType> opt_dtype) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}
#endif

static Tensor& prod_out_impl(Tensor& result, const Tensor& self, IntArrayRef dim,
                        bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("prod", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.fill_(1);
  } else {
    prod_stub(iter->device_type(), *iter);
  }
  return result;
}

Tensor prod(const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  Tensor result;
  native::prod_out_impl(result, self, dim, keepdim, dtype);
  return result;
}

Tensor prod(const Tensor &self, c10::optional<ScalarType> dtype) {
  Tensor result;
  return at::native::prod_out_impl(result, self, {}, false, dtype);
}

Tensor& prod_out(Tensor& result, const Tensor& self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype) {
  return at::native::prod_out_impl(result, self, dim, keepdim, dtype);
}

Tensor &mean_out(Tensor &result, const Tensor &self, IntArrayRef dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");
  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  // TODO: the TensorIterator reduction implementation of mean
  // (mean_kernel_impl()) is unvectorized and leads to very poor performance
  // for production workloads. Once that's fixed, the following code can be used
  // in lieu of the sum + divide implementation below.
  if (self.device().is_cpu()) {
    int64_t dim_prod = 1;
    if (dim.size() == 0 || self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    at::sum_out(result, self, dim, keepdim, dtype).div_(dim_prod);
    return result;
  }

  auto iter = make_reduction("mean", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    mean_stub(iter->device_type(), *iter);
  }
  return result;
}

Tensor mean(const Tensor &self, optional<ScalarType> dtype) {
  return at::native::mean(self, {}, false, dtype);
}

Tensor mean(const Tensor& self, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  Tensor result;
  return at::native::mean_out(result, self, dim, keepdim, dtype);
}

static Tensor squeeze_multiple(const Tensor& self, IntArrayRef dims) {
  int ndims = self.sizes().size();
  auto dims_to_squeeze = at::dim_list_to_bitset(dims, ndims);
  Tensor result = self;
  for (int i = ndims - 1; i >= 0; --i) {
    if (dims_to_squeeze[i]) {
      result = result.squeeze(i);
    }
  }
  return result;
}

Tensor& logsumexp_out(Tensor& result, const Tensor &self, IntArrayRef dims, bool keepdim) {
  // can't take max of empty tensor
  if (self.numel() != 0) {
    auto maxes = at::max_values(self, dims, true);
    auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
    maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
    at::sum_out(result, at::exp(self - maxes), dims, keepdim);
    result.log_().add_(maxes_squeezed);
  } else {
    at::sum_out(result, at::exp(self), dims, keepdim);
    result.log_();
  }
  return result;
}

Tensor logsumexp(const Tensor &self, IntArrayRef dims, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::logsumexp_out(result, self, dims, keepdim);
}

static Tensor& norm_out(Tensor &result, const Tensor &self, optional<Scalar> opt_p,
                               IntArrayRef dim, bool keepdim, optional<ScalarType> opt_dtype) {
  auto p = opt_p.value_or(2.0);
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "norm only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));

  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");

  ScalarType dtype = get_dtype(result, self, opt_dtype, true);
  auto iter = make_reduction("norm", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.zero_();
  } else {
    norm_stub(iter->device_type(), *iter, p);
  }
  return result;
}

static inline Tensor _norm(const Tensor &self, Scalar p) {
  if (self.is_sparse()) {
    return at::native_norm(self, p);
  } else {
    TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
             "norm only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
    TORCH_CHECK(at::isFloatingType(self.scalar_type()), "norm only supports floating-point dtypes");

    Tensor result;
    return at::native::norm_out(result, self, p, {}, false, c10::nullopt);
  }
}

Tensor &norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm_out(result, self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor &norm_out(Tensor& result, const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  return at::native::norm_out(result, self, p, dim, keepdim, c10::nullopt);
}

static Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim,
            optional<ScalarType> opt_dtype) {
  Tensor result;
  return at::native::norm_out(result, self, p, dim, keepdim, opt_dtype);
}

Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype) {
  return at::native::norm(self, p, dim, keepdim, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, optional<Scalar> p, ScalarType dtype) {
  return at::native::norm(self, p, {}, false, optional<ScalarType>(dtype));
}

Tensor norm(const Tensor& self, optional<Scalar> p, IntArrayRef dim, bool keepdim) {
  return at::native::norm(self, p, dim, keepdim, c10::nullopt);
}

// leave it so we support sparse tensors
Tensor norm(const Tensor& self, Scalar p) {
  return at::native::_norm(self, p);
}

inline Tensor & _all(Tensor & result, std::unique_ptr<TensorIterator> & iter) {
  if (iter->numel() == 0) {
    result.fill_(1);
  } else {
    and_stub(iter->device_type(), *iter);
  }

  return result;
}

Tensor all(const Tensor& self) {
  TORCH_CHECK(self.type().backend() == Backend::CPU ||
    self.type().backend() == Backend::CUDA, "all only supports CPU AND CUDA "
    "backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter = make_reduction(
    "all", result, self, {}, false, self.scalar_type());
  return _all(result, iter);
}

Tensor all(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::all_out(result, self, dim, keepdim);
}

Tensor &all_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.type().backend() == Backend::CPU ||
    self.type().backend() == Backend::CUDA, "all only supports CPU AND CUDA "
    "backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 1, dim, keepdim)) {
    return result;
  } else {
    auto iter = make_reduction(
      "all", result, self, dim, keepdim, self.scalar_type());
    return _all(result, iter);
  }
}

inline Tensor & _any(Tensor & result, std::unique_ptr<TensorIterator> & iter) {
  if (iter->numel() == 0) {
    result.fill_(0);
  } else {
    or_stub(iter->device_type(), *iter);
  }

  return result;
}

Tensor any(const Tensor& self) {
  TORCH_CHECK(self.type().backend() == Backend::CPU ||
    self.type().backend() == Backend::CUDA, "any only supports CPU AND CUDA "
    "backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");

  Tensor result = at::empty({0}, self.options());
  auto iter = make_reduction(
    "any", result, self, {}, false, self.scalar_type());
  return _any(result, iter);
}

Tensor any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::any_out(result, self, dim, keepdim);
}

Tensor &any_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.type().backend() == Backend::CPU ||
    self.type().backend() == Backend::CUDA, "any only supports CPU AND CUDA "
    "backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(self.scalar_type() == at::ScalarType::Byte || self.scalar_type() == at::ScalarType::Bool,
    "all only supports torch.uint8 and torch.bool dtypes");
  dim = maybe_wrap_dim(dim, self.dim());
  if (_dimreduce_return_trivial(result, self, 0, dim, keepdim)) {
    return result;
  } else {
    auto iter = make_reduction(
      "any", result, self, dim, keepdim, self.scalar_type());
    return _any(result, iter);
  }
}

Tensor min_values(const Tensor& self, IntArrayRef dims, bool keepdim) {
  if (dims.size() == 1) {
    return std::get<0>(self.min(dims[0], keepdim));
  } else {
    Tensor result = at::empty({0}, self.options());
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("min_values", result, self, dims, keepdim, dtype);
    TORCH_CHECK(iter->numel() > 0, "min_values on a tensor with no elements is not defined.");
    min_values_stub(iter->device_type(), *iter);
    return result;
  }
}

Tensor max_values(const Tensor& self, IntArrayRef dims, bool keepdim) {
  if (dims.size() == 1) {
    return std::get<0>(self.max(dims[0], keepdim));
  } else {
    Tensor result = at::empty({0}, self.options());
    ScalarType dtype = get_dtype(result, self, {}, true);
    auto iter = make_reduction("max_values", result, self, dims, keepdim, dtype);
    TORCH_CHECK(iter->numel() > 0, "max_values on a tensor with no elements is not defined.");
    max_values_stub(iter->device_type(), *iter);
    return result;
  }
}

static Tensor &std_var_out(Tensor &result, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt) {
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "std and var only support CPU AND CUDA backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "std and var only support floating-point dtypes");
  ScalarType dtype = get_dtype(result, self, {}, true);
  auto iter = make_reduction("std or var", result, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result.fill_(NAN);
  } else {
    std_var_stub(iter->device_type(), *iter, unbiased, take_sqrt);
  }
  return result;
}

static std::tuple<Tensor&,Tensor&> std_var_mean_out(const char* fname, Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim, bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           fname, " only support CPU and CUDA backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(at::isFloatingType(self.type().scalarType()), fname, " only support floating-point dtypes");
  TORCH_CHECK(result1.type().scalarType() == result2.type().scalarType(),
           "provided by result1 dtype must match dtype of result2. Got ",
           toString(result1.type().scalarType()),
           " and ",
           toString(result2.type().scalarType()),
           ".");
  ScalarType dtype = get_dtype(result1, self, {}, true);
  auto iter = make_reduction(fname, result1, result2, self, dim, keepdim, dtype);
  if (iter->numel() == 0) {
    result1.fill_(NAN);
    result2.fill_(NAN);
  } else {
    std_var_stub(iter->device_type(), *iter, unbiased, take_sqrt);
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

std::tuple<Tensor&,Tensor&> var_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_mean_out("var_mean", result1, result2, self, dim, unbiased, keepdim, false);
}

std::tuple<Tensor&,Tensor&> std_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_mean_out("std_mean", result1, result2, self, dim, unbiased, keepdim, true);
}

std::tuple<Tensor&,Tensor&> var_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, bool unbiased) {
  return std_var_mean_out("var_mean", result1, result2, self, {}, unbiased, false, false);
}

std::tuple<Tensor&,Tensor&> std_mean_out(Tensor &result1, Tensor &result2, const Tensor &self, bool unbiased) {
  return std_var_mean_out("std_mean", result1, result2, self, {}, unbiased, false, true);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::var_mean_out(result1, result2, self, dim, unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::std_mean_out(result1, result2, self, dim, unbiased, keepdim);
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, bool unbiased) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::std_mean_out(result1, result2, self, unbiased);
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, bool unbiased) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  return at::native::var_mean_out(result1, result2, self, unbiased);
}

Tensor var(const Tensor& self, bool unbiased) {
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "var only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "var only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  return trivial_return.has_value() ? trivial_return.value() : at::_var(self, unbiased);
}

Tensor var(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::var_out(result, self, dim, unbiased, keepdim);
}

Tensor &var_out(Tensor &result, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_out(result, self, dim, unbiased, keepdim, false);
}

Tensor std(const Tensor& self, bool unbiased) {
  TORCH_CHECK(self.type().backend() == Backend::CPU || self.type().backend() == Backend::CUDA,
           "std only supports CPU AND CUDA backend, got: ", toString(self.type().backend()));
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "std only supports floating-point dtypes");
  auto trivial_return = _allreduce_return_trivial(self, std::numeric_limits<double>::quiet_NaN());
  return trivial_return.has_value() ? trivial_return.value() : at::_std(self, unbiased);
}

Tensor std(const Tensor& self, IntArrayRef dim, bool unbiased, bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  return at::native::std_out(result, self, dim, unbiased, keepdim);
}

Tensor &std_out(Tensor &result, const Tensor &self, IntArrayRef dim, bool unbiased, bool keepdim) {
  return std_var_out(result, self, dim, unbiased, keepdim, true);
}

}} // namespace at::native
