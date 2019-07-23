import unittest
from common_utils import TestCase, run_tests
from common_cuda import TEST_CUDA
from collections import namedtuple
import itertools
import torch
import sys


def namedtensor_enabled():
    return '-DBUILD_NAMEDTENSOR' in torch.__config__.show()

skipIfNamedTensorDisabled = \
    unittest.skipIf(not namedtensor_enabled(),
                    'PyTorch not compiled with namedtensor support')

def pass_name_to_python_arg_parser(name):
    x = torch.empty(2, names=(name,))


def flatten(lst):
    return [item for sublist in lst for item in sublist]


Function = namedtuple('TestCase', ['name', 'lambd'])


class TestNamedTensor(TestCase):
    def test_trivial(self):
        pass

    def _test_factory(self, factory, device):
        x = factory([], device=device)
        self.assertEqual(x.names, ())

        x = factory(1, 2, 3, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=None, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=('N', 'T', 'D'), device=device)
        self.assertEqual(x.names, ('N', 'T', 'D'))

        x = factory(1, 2, 3, names=('N', None, 'D'), device=device)
        self.assertEqual(x.names, ('N', None, 'D'))

        with self.assertRaisesRegex(RuntimeError,
                                    'must contain alphabetical characters and/or underscore'):
            x = factory(2, names=('?',), device=device)

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            x = factory(2, 1, names=('N',), device=device)

        with self.assertRaisesRegex(TypeError, 'invalid combination of arguments'):
            x = factory(2, 1, names='N', device=device)

        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('N', 'C', 'N'), device=device)

        # Tests for tagged names
        x = factory(2, 3, 1, names=('C.in', 'H', 'C.out'), device=device)
        self.assertEqual(x.names, ('C.in', 'H', 'C.out'))

        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('C.in', 'H', 'C.in'), device=device)

        with self.assertRaisesRegex(
                RuntimeError,
                'with duplicate names unless they are tagged and have different tags'):
            x = factory(2, 1, 1, names=('C.in', 'H', 'C'), device=device)


    def test_empty(self):
        self._test_factory(torch.empty, 'cpu')

    def test_copy_transpose(self):
        # This type of copy is special-cased and therefore needs its own test
        def _test(self_names, other_names, expected_names):
            x = torch.empty(2, 5, names=self_names)
            y = torch.empty(5, 2).t().set_names_(other_names)
            x.copy_(y)
            self.assertEqual(x.names, expected_names)

        _test(('N', 'C'), ('N', 'C'), ('N', 'C'))
        _test(('N', None), ('N', 'C'), ('N', 'C'))
        _test(None, ('N', 'C'), ('N', 'C'))

    def test_set_names_(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))
        self.assertEqual(tensor.set_names_(None).names, (None, None))
        self.assertEqual(tensor.set_names_(['H', 'W']).names, ('H', 'W'))
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.set_names_(['N', 'C', 'W'])
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.set_names_(['N', 'N'])

    def test_set_names_property(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))

        tensor.names = None
        self.assertEqual(tensor.names, (None, None))

        tensor.names = ('N', 'W')
        self.assertEqual(tensor.names, ('N', 'W'))

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.names = ['N', 'C', 'W']
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.names = ['N', 'N']

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_empty_cuda(self):
        self._test_factory(torch.empty, 'cuda')

    def test_size(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.size('N'), 2)
        self.assertEqual(t.size('C'), 5)
        with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name*'):
            t.size(None)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.size('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).size('N')

    def test_stride(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.stride('N'), 3 * 5)
        self.assertEqual(t.stride('C'), 1)
        with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
            t.stride(None)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.stride('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).stride('N')

    def test_info_smoke(self):
        # Smoke test for info functions / methods / attributes on named tensors.
        tensor = torch.empty(1, 1, names=('N', 'D'))

        tensor.device
        tensor.dtype
        tensor.get_device()
        tensor.is_complex()
        tensor.is_floating_point()
        tensor.is_nonzero()
        torch.is_same_size(tensor, tensor)
        torch.is_signed(tensor)
        tensor.layout
        tensor.numel()
        tensor.dim()
        tensor.element_size()
        tensor.is_contiguous()
        tensor.is_cuda
        tensor.is_leaf
        tensor.is_pinned()
        tensor.is_shared()
        tensor.is_sparse
        tensor.ndimension()
        tensor.nelement()
        tensor.shape
        tensor.size()
        tensor.size(1)
        tensor.storage()
        tensor.storage_offset()
        tensor.storage_type()
        tensor.stride()
        tensor.stride(1)
        tensor.data
        tensor.data_ptr()
        tensor.ndim
        tensor.item()

    def test_split_fns_propagates_names(self):
        fns = [
            lambda x: x.split(1, 0),
            lambda x: x.split([1, 1], 1),
            lambda x: x.chunk(2, 0),
        ]

        for device in torch.testing.get_all_device_types():
            orig_tensor = torch.empty(2, 2, names=('N', 'D'), device=device)
            for fn in fns:
                splits = fn(orig_tensor)
                for split in splits:
                    self.assertEqual(split.names, orig_tensor.names)

    def test_binary_ops(self):
        def test_basic(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            b = torch.empty(2, 3, names=('C', 'N'))
            c = torch.empty(3, names=('C',))
            d = torch.empty(3, names=('W',))

            self.assertEqual(op(a, a).names, ('N', 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))

            with self.assertRaisesRegex(RuntimeError, "do not match"):
                op(a, d)
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                op(a, b)

        def test_wildcard(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            c = torch.empty(2, 3, names=(None, 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))

            b = torch.empty(2, 3)
            self.assertEqual(op(a, b).names, ('N', 'C'))

            d = torch.empty(2, 3, names=('C', None))
            with self.assertRaisesRegex(RuntimeError, "misaligned"):
                op(d, c)

        def method(name, *args, **kwargs):
            return [Function(name, lambda a, b: getattr(a, name)(b, *args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(a, b):
                result = a.new_empty([0])
                out_fn(a, b, *args, out=result, **kwargs)
                return result

            return [Function(name, fn)]

        def fn_method_and_inplace(name, *args, **kwargs):
            return (
                method(name, *args, **kwargs) +
                method(name + '_', *args, **kwargs) +
                out_function(name, *args, **kwargs)
            )

        tests = [
            fn_method_and_inplace('mul'),
            method('copy_'),
        ]
        tests = flatten(tests)

        for _, op in tests:
            test_basic(op)
            test_wildcard(op)

    def test_unary_propagate_names_fns(self):
        def _test(testcase, names=('N', 'D'), device='cpu'):
            sizes = [2] * len(names)
            tensor = torch.empty(sizes, names=names, device=device)
            out = testcase.lambd(tensor)
            self.assertEqual(out.names, tensor.names,
                             message=testcase.name)

        def method(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(t, name)(*args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(tensor):
                result = tensor.new_empty([0])
                out_fn(tensor, *args, out=result, **kwargs)
                return result

            return [Function(name + '_out', fn)]

        def fn_method_and_inplace(name, *args, **kwargs):
            return (
                method(name, *args, **kwargs) +
                method(name + '_', *args, **kwargs) +
                out_function(name, *args, **kwargs)
            )

        # All of these operate on 2x2 tensors.
        tests = [
            # unary pointwise
            fn_method_and_inplace('abs'),
            fn_method_and_inplace('acos'),
            fn_method_and_inplace('asin'),
            fn_method_and_inplace('atan'),
            fn_method_and_inplace('ceil'),
            fn_method_and_inplace('clamp', -1, 1),
            fn_method_and_inplace('clamp_min', -2),
            fn_method_and_inplace('clamp_max', 2),
            method('cauchy_'),
            fn_method_and_inplace('cos'),
            fn_method_and_inplace('cosh'),
            fn_method_and_inplace('digamma'),
            fn_method_and_inplace('erf'),
            fn_method_and_inplace('erfc'),
            fn_method_and_inplace('erfinv'),
            fn_method_and_inplace('exp'),
            fn_method_and_inplace('expm1'),
            method('exponential_'),
            fn_method_and_inplace('floor'),
            fn_method_and_inplace('frac'),
            method('geometric_', p=0.5),
            fn_method_and_inplace('lgamma'),
            fn_method_and_inplace('log'),
            fn_method_and_inplace('log10'),
            fn_method_and_inplace('log1p'),
            fn_method_and_inplace('log2'),
            method('log_normal_'),
            fn_method_and_inplace('neg'),
            method('normal_'),
            [Function('polygamma', lambda t: torch.polygamma(1, t))],
            method('polygamma_', 1),
            fn_method_and_inplace('reciprocal'),
            method('random_', 0, 1),
            method('random_', 1),
            method('random_'),
            fn_method_and_inplace('round'),
            fn_method_and_inplace('rsqrt'),
            fn_method_and_inplace('sigmoid'),
            fn_method_and_inplace('sign'),
            fn_method_and_inplace('sin'),
            fn_method_and_inplace('sinh'),
            fn_method_and_inplace('sqrt'),
            fn_method_and_inplace('tan'),
            fn_method_and_inplace('tanh'),
            fn_method_and_inplace('trunc'),
            method('uniform_'),
            method('zero_'),
            method('fill_', 1),
            method('fill_', torch.tensor(3.14)),

            # views
            method('narrow', 0, 0, 1),
        ]
        tests = flatten(tests)

        for testcase, device in itertools.product(tests, torch.testing.get_all_device_types()):
            _test(testcase, device=device)

    def test_reduction_fns(self):
        def test_simple_reduce(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)
            self.assertEqual(op(t, 1).names, ['N', 'L'])
            self.assertEqual(op(t, 'C').names, ['N', 'L'])
            with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                op(t, None)
            with self.assertRaisesRegex(RuntimeError, 'Name \'H\' not found'):
                op(t, 'H')

        def test_complete_reduce(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)
            self.assertEqual(op(t).names, [])

        def test_multidim_reduce(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)

            self.assertEqual(op(t, [1, 2]).names, ['N'])
            self.assertEqual(op(t, ['C', 'L']).names, ['N'])
            with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                op(t, [None, 'C'])

        def test_out_variant(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            out = t.new_empty([0])
            getattr(torch, op_name)(t, 'C', out=out)
            self.assertEqual(out.names, ['N', 'L'])

        def test_keepdim(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)
            self.assertEqual(op(t, 'C', keepdim=True).names, ['N', 'C', 'L'])

        Case = namedtuple('Case', [
            'op_name',
            'supports_complete_reduce',
            'supports_multidim_reduce',
        ])

        tests = [
            Case(op_name='sum', supports_complete_reduce=True, supports_multidim_reduce=True),
            Case(op_name='prod', supports_complete_reduce=True, supports_multidim_reduce=False),
        ]

        for testcase, device in itertools.product(tests, torch.testing.get_all_device_types()):
            op_name = testcase.op_name
            test_simple_reduce(op_name, device)
            test_keepdim(op_name, device)
            test_out_variant(op_name, device)

            if testcase.supports_complete_reduce:
                test_complete_reduce(op_name, device)
            if testcase.supports_multidim_reduce:
                test_multidim_reduce(op_name, device)

    def test_using_seen_interned_string_doesnt_bump_refcount(self):
        def see_name():
            seen_name = 'N'
            pass_name_to_python_arg_parser(seen_name)

        see_name()
        seen_name = 'N'
        old_refcnt = sys.getrefcount(seen_name)

        pass_name_to_python_arg_parser(seen_name)

        new_refcnt = sys.getrefcount(seen_name)
        self.assertEqual(new_refcnt, old_refcnt)

    def test_using_unseen_interned_string_bumps_refcount_permanently(self):
        # Please don't use this as a name in a different test.
        unseen_name = 'abcdefghi'
        old_refcnt = sys.getrefcount(unseen_name)

        pass_name_to_python_arg_parser(unseen_name)

        new_refcnt = sys.getrefcount(unseen_name)
        self.assertEqual(new_refcnt, old_refcnt + 1)

    def test_using_unseen_uninterned_string_refcounts(self):
        # Please don't use this as a name in a different test.
        # non-compile-time constants are not interned
        unseen_name = ''.join(['abc', 'def', 'ghi', 'jkl'])
        interned_unseen_name = 'abcdefghijkl'
        self.assertFalse(unseen_name is interned_unseen_name)

        old_uninterned_refcnt = sys.getrefcount(unseen_name)
        old_interned_refcnt = sys.getrefcount(interned_unseen_name)

        pass_name_to_python_arg_parser(unseen_name)

        new_uninterned_refcnt = sys.getrefcount(unseen_name)
        new_interned_refcnt = sys.getrefcount(interned_unseen_name)

        # Internally, PyTorch should not hold a reference to the uninterned string
        self.assertEqual(new_uninterned_refcnt, old_uninterned_refcnt)

        # Instead, we should hold a new reference to the interned version.
        self.assertEqual(new_interned_refcnt, old_interned_refcnt + 1)

    def _test_select(self, device):
        x = torch.empty(2, 3, 4, 5, names=('N', 'C', 'H', 'W'), device=device)
        y = x.select(1, 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        y = x.select('C', 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        with self.assertRaisesRegex(
                RuntimeError, 'Please look up dimensions by name'):
            y = x.select(None, 1)

        with self.assertRaisesRegex(
                RuntimeError, 'Name \'C.in\' not found in'):
            y = x.select('C.in', 1)

        x = torch.empty(2, 3, 4, 5, names=('N', 'C.in', 'H', 'W'), device=device)
        y = x.select('C', 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        x = torch.empty(2, 3, 4, 5, names=('C.out', 'C.in', 'H', 'W'), device=device)
        y = x.select('C.in', 1)
        self.assertEqual(y.names, ('C.out', 'H', 'W'))

        with self.assertRaisesRegex(
                RuntimeError, 'Name \'C\' could refer to multiple dimensions'):
            y = x.select('C', 1)


    def test_select(self):
        self._test_select('cpu')

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_select_cuda(self):
        self._test_select('cuda')

    def _test_as_strided(self, device):
        x = torch.empty(2, 3, 4, 5, names=('N', 'C', 'H', 'W'), device=device)
        y = x.as_strided([2 * 3 * 4 * 5], [1])
        self.assertEqual(y.names, (None,))

    def test_as_strided(self):
        self._test_as_strided('cpu')

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_as_strided_cuda(self):
        self._test_as_strided('cuda')

# Disable all tests if named tensor is not available.
for attr in dir(TestNamedTensor):
    if attr.startswith('test_'):
        new_test = skipIfNamedTensorDisabled(getattr(TestNamedTensor, attr))
        setattr(TestNamedTensor, attr, new_test)

if __name__ == '__main__':
    run_tests()
