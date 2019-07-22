r""" Functional interface (quantized)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch._ops import ops as _ops
from torch._jit_internal import List as _List

def _extend_to_list(val, length=2):
    if not isinstance(val, (tuple, list)):
        val = [val] * length
    return val

def relu(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""relu(input, inplace=False) -> Tensor

    .. note::
      :attr:`inplace` is not supported for the quantized relu.

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.quantized.ReLU` for more details.
    """
    if inplace:
        raise NotImplementedError("`inplace` is not implemented in quantized relu")
    return _ops.quantized.relu(input)

def linear(input, weight, bias=None, scale=None, zero_point=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    r"""
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.
    See :class:`~torch.nn.quantized.Linear`

    .. note::

      Current implementation uses packed weights. This has penalty on performance.
      If you want to avoid the overhead, use :class:`~torch.nn.quantized.Linear`.

    Args:
      input (Tensor): Quantized input of type `torch.quint8`
      weight (Tensor): Quantized weight of type `torch.qint8`
      bias (Tensor): None or Quantized bias of type `torch.qint32`
      scale (double): output scale. If None, derived from the input scale
      zero_point (long): output zero point. If None, derived from the input zero_point

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if scale is None:
        scale = input.q_scale()
    if zero_point is None:
        zero_point = input.q_zero_point()
    _packed_weight = torch.ops.quantized.fbgemm_linear_prepack(weight)
    return torch.ops.quantized.fbgemm_linear(input, _packed_weight, bias, scale,
                                             zero_point)

def conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8,
           prepacked=True):
    r"""
    conv2d(input, weight, bias,
           stride=1, padding=0, dilation=1, groups=1,
           padding_mode='zeros',
           scale=1.0, zero_point=0,
           dtype=torch.quint8,
           prepacked=True) -> Tensor

    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.nn.quantized.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.int32`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``
        prepacked: assume that the weights are prepacked. Default: True

    Examples::

        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(4, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype = torch.quint8
        >>>
        >>> q_filters = torch.quantize_linear(filters, scale, zero_point, dtype)
        >>> q_inputs = torch.quantize_linear(inputs, scale, zero_point, dtype)
        >>> q_bias = torch.quantize_linear(bias, scale, zero_point, torch.quint8)
        >>> qF.conv2d(q_inputs, q_filters, q_bias, scale, zero_point, padding=1)
    """  # noqa: E501
    if padding_mode != 'zeros':
        raise NotImplementedError("Only zero-padding is supported!")
    spatial_dim_len = len(input.shape) - 2  # no batches and channels
    stride = _extend_to_list(stride, spatial_dim_len)
    padding = _extend_to_list(padding, spatial_dim_len)
    dilation = _extend_to_list(dilation, spatial_dim_len)

    if not prepacked:
        weight = _ops.quantized.fbgemm_conv_prepack(weight, groups)
    return _ops.quantized.fbgemm_conv2d(input, weight, bias,
                                        stride, padding, dilation,
                                        groups, scale, zero_point)

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    r"""Applies a 2D max pooling over a quantized input signal composed of
    several quantized input planes.

    See :class:`~torch.nn.quantized.MaxPool2d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(_List[int], [])
    return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding,
                                          dilation, ceil_mode, return_indices)
