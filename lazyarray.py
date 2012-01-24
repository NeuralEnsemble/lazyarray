"""
lazyarray is a Python package that provides a lazily-evaluated numerical array
class, ``larray``, based on and compatible with NumPy arrays.

Copyright Andrew P. Davison, 2012
"""
from __future__ import division
import numpy
import operator
from copy import deepcopy
import collections
from functools import wraps

__version__ = "0.1.0dev"


def check_shape(meth):
    """
    Decorator for larray magic methods, to ensure that the operand has
    the same shape as the array.
    """
    @wraps(meth)
    def wrapped_meth(self, val):
        if isinstance(val, (larray, numpy.ndarray)):
            if val.shape != self.shape:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        return meth(self, val)
    return wrapped_meth


def requires_shape(meth):
    @wraps(meth)
    def wrapped_meth(self, *args, **kwargs):
        if self.shape is None:
            raise ValueError("Shape of larray not specified")
        return meth(self, *args, **kwargs)
    return wrapped_meth


def reverse(func):
    """Given a function f(a, b), returns f(b, a)"""
    def reversed_func(a, b):
        return func(b, a)
    reversed_func.__doc__ = "Reversed argument form of %s" % func.__doc__
    return reversed_func


def lazy_operation(name, reversed=False):
    def op(self, val):
        new_map = deepcopy(self)
        f = getattr(operator, name)
        if reversed:
            f = reverse(f)
        new_map.operations.append((f, val))
        return new_map
    return check_shape(op)


def lazy_inplace_operation(name):
    def op(self, val):
        self.operations.append((getattr(operator, name), val))
        return self
    return check_shape(op)


def lazy_unary_operation(name):
    def op(self):
        new_map = deepcopy(self)
        new_map.operations.append((getattr(operator, name), None))
        return new_map
    return op


class larray(object):
    """
    Optimises storage of and operations on arrays in various ways:
      - stores only a single value if all the values in the array are the same;
      - if the array is created from a function `f(i)` or `f(i,j)`, then
        elements are only evaluated when they are accessed. Any operations
        performed on the array are also queued up to be executed on access.

    Two use cases for the latter are:
      - to save memory for very large arrays by accessing them one row or
        column at a time: the entire array need never be in memory.
      - in parallelized code, different rows or columns may be evaluated
        on different nodes or in different threads.
    """

    def __init__(self, value, shape=None):
        """
        Create a new lazy array.

        `value` : may be an int, long, float, bool, NumPy array, iterator,
                  generator or a function, `f(i)` or `f(i,j)`, depending on the
                  dimensions of the array.

        `f(i,j)` should return a single number when `i` and `j` are integers,
        and a 1D array when either `i` or `j` or both is a NumPy array (in the
        latter case the two arrays musy have equal lengths).
        """
        if isinstance(value, collections.Sized):  # False for numbers, generators, functions, iterators
            #assert numpy.isreal(value).all()
            if not isinstance(value, numpy.ndarray):
                value = numpy.array(value)
            if shape:
                assert value.shape == shape,  "Array has shape %s, value has shape %s" % (shape, value.shape)
            self.shape = value.shape
        else:
            assert numpy.isreal(value)  # also True for callables, generators, iterators
            self.shape = shape
        self.base_value = value
        self.operations = []

    def __deepcopy__(self, memo):
        obj = larray.__new__(larray)
        try:
            obj.base_value = deepcopy(self.base_value)
        except TypeError:  # base_value cannot be copied, e.g. is a generator (but see generator_tools from PyPI)
            obj.base_value = self.base_value  # so here we create a reference rather than deepcopying - could cause problems
        obj.shape = self.shape
        obj.operations = deepcopy(self.operations)
        return obj

    @property
    @requires_shape
    def nrows(self):
        """Size of the first dimension of the array."""
        return self.shape[0]

    @property
    @requires_shape
    def ncols(self):
        """Size of the second dimension (if it exists) of the array."""
        return self.shape[1]

    @property
    def is_homogeneous(self):
        """True if all the elements of the array are the same."""
        return isinstance(self.base_value, (int, long, float, bool))

    def _homogeneous_array(self, addr):
        def size(x, max):
            if isinstance(x, (int, long)):
                return 1
            elif isinstance(x, slice):
                return ((x.stop or max) - (x.start or 0)) // (x.step or 1)
            elif isinstance(x, collections.Sized):
                return len(x)
            else:
                raise Exception("something went wrong")
        shape = (size(x, max) for (x, max) in zip(addr, self.shape))
        if len(shape) == 1 and shape[0] == 1:
            return 1
        else:
            return numpy.ones(shape, type(self.base_value))

    def _array_indices(self, addr):
        def axis_indices(x, max):
            if isinstance(x, (int, long)):
                return x
            elif isinstance(x, slice):
                return numpy.arange((x.start or 0),
                                    (x.stop or max),
                                    (x.step or 1),
                                    dtype=int)
            elif isinstance(x, collections.Sized):
                return x
            else:
                raise Exception("something went wrong")
        if isinstance(addr, (int, long)):
            addr = (addr,)
        if len(addr) < len(self.shape):
            full_addr = [slice(None)] * len(self.shape)
            for i, val in enumerate(addr):
                full_addr[i] = val
            addr = full_addr
        indices = [axis_indices(x, max) for (x, max) in zip(addr, self.shape)]
        if len(indices) == 1:
            if isinstance(indices[0], collections.Sized):
                return indices[0]
            else:
                return indices
        elif len(indices) == 2:
            if isinstance(indices[0], collections.Sized):
                if isinstance(indices[1], collections.Sized):
                    return numpy.meshgrid(*indices)
            return indices
        else:
            raise NotImplementedError("Only 1D and 2D arrays supported")

    @requires_shape
    def __getitem__(self, addr):
        if self.is_homogeneous:
            base_val = self._homogeneous_array(addr) * self.base_value
        elif isinstance(self.base_value, numpy.ndarray):
            base_val = self.base_value[addr]
        elif callable(self.base_value):
            indices = self._array_indices(addr)
            base_val = self.base_value(*indices)
        elif isinstance(self.base_value, collections.Iterator):
            raise NotImplementedError
        else:
            raise Exception("something went wrong")
        return self._apply_operations(base_val)

    @requires_shape
    def check_bounds(self, addr):
        """
        Check whether the given address is within the array bounds.
        """
        if isinstance(addr, (int, long, float)):
            addr = (addr,)
        for i, size in zip(addr, self.shape):
            if (i < -size) or (i >= size):
                raise IndexError("index out of bounds")

    def apply(self, f):
        """
        Add the function `f(x)` to the list of the operations to be performed,
        where `x` will be a scalar or a numpy array.

        >>> m = larray(4, shape=(2,2))
        >>> m.apply(numpy.sqrt)
        >>> m.evaluate()
        array([[ 2.,  2.],
               [ 2.,  2.]])
        """
        self.operations.append((f, None))

    def _apply_operations(self, x):
        for f, arg in self.operations:
            if arg is None:
                x = f(x)
            elif isinstance(arg, larray):
                x = f(x, arg.evaluate())  # need to be cleverer, for partial evaluation
            else:
                x = f(x, arg)
        return x

    @requires_shape
    def by_column(self, mask=None):
        """
        Iterate over the columns of the array. Columns will be yielded either
        as a 1D array or as a single value (for a flat array).

        `mask`: either None or a boolean array indicating which columns should
                be included.
        """
        column_indices = numpy.arange(self.ncols)
        if mask is not None:
            assert len(mask) == self.ncols
            column_indices = column_indices[mask]
        if self.is_homogeneous:
            for j in column_indices:
                yield self._apply_operations(self.base_value)
        elif isinstance(self.base_value, numpy.ndarray):
            for j in column_indices:
                yield self._apply_operations(self.base_value[:, j])
#        elif isinstance(self.base_value, random.RandomDistribution):
#            if mask is None:
#                for j in column_indices:
#                    yield self._apply_operations(self.base_value.next(self.nrows, mask_local=False))
#            else:
#                column_indices = numpy.arange(self.ncols)
#                for j,local in zip(column_indices, mask):
#                    col = self.base_value.next(self.nrows, mask_local=False)
#                    if local:
#                        yield self._apply_operations(col)
        #elif isinstance(self.base_value, larray):
        #    for column in self.base_value.by_column(mask=mask):
        #        yield self._apply_operations(column)
        elif callable(self.base_value):  # a function of (i,j)
            row_indices = numpy.arange(self.nrows, dtype=int)
            for j in column_indices:
                yield self._apply_operations(self.base_value(row_indices, j))
        else:
            raise Exception("invalid mapping")

    @requires_shape
    def evaluate(self, simplify=False):
        """
        Return the lazy array as a real NumPy array.

        If the array is homogeneous and ``simplify`` is ``True``, return a
        single numerical value.
        """
        # need to catch the situation where a generator-based larray is evaluated a second time
        if self.is_homogeneous:
            if simplify:
                x = self.base_value
            else:
                x = self.base_value * numpy.ones(self.shape)
        elif isinstance(self.base_value, numpy.ndarray):
            x = self.base_value
#        elif isinstance(self.base_value, random.RandomDistribution):
#            n = self.nrows*self.ncols
#            x = self.base_value.next(n).reshape(self.shape)
        elif callable(self.base_value):
            row_indices = numpy.arange(self.nrows, dtype=int)
            x = numpy.array([self.base_value(row_indices, j) for j in range(self.ncols)]).T  # is this not equivalent to numpy.fromfunction?
        elif isinstance(self.base_value, collections.Iterator):
            x = numpy.fromiter(self.base_value, dtype=float)
            if x.shape != self.shape:
                x = x.reshape(self.shape)
        else:
            raise Exception("invalid mapping")
        return self._apply_operations(x)

    __iadd__ = lazy_inplace_operation('add')
    __isub__ = lazy_inplace_operation('sub')
    __imul__ = lazy_inplace_operation('mul')
    __idiv__ = lazy_inplace_operation('div')
    __ipow__ = lazy_inplace_operation('pow')

    __add__  = lazy_operation('add')
    __radd__ = __add__
    __sub__  = lazy_operation('sub')
    __rsub__ = lazy_operation('sub', reversed=True)
    __mul__  = lazy_operation('mul')
    __rmul__ = __mul__
    __div__  = lazy_operation('div')
    __rdiv__ = lazy_operation('div', reversed=True)
    __truediv__ = lazy_operation('truediv')
    __truediv__ = lazy_operation('truediv', reversed=True)
    __pow__  = lazy_operation('pow')

    __lt__   = lazy_operation('lt')
    __gt__   = lazy_operation('gt')
    __le__   = lazy_operation('le')
    __ge__   = lazy_operation('ge')

    __neg__  = lazy_unary_operation('neg')
    __pos__  = lazy_unary_operation('pos')
    __abs__  = lazy_unary_operation('abs')


def _build_ufunc(func):
    """Return a ufunc that works with lazy arrays"""
    def larray_compatible_ufunc(x):
        if isinstance(x, larray):
            y = deepcopy(x)
            y.apply(func)
            return y
        else:
            return func(x)
    return larray_compatible_ufunc


# build lazy-array comptible versions of NumPy ufuncs
namespace = globals()
for name in dir(numpy):
    obj = getattr(numpy, name)
    if isinstance(obj, numpy.ufunc):
        namespace[name] = _build_ufunc(obj)
