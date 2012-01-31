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

__version__ = "0.1.0"

# stuff for Python 3 compatibility
try:
    long
except NameError:
    long = int

try:
    reduce
except NameError:
    from functools import reduce


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
        if len(self.shape) > 1:
            return self.shape[1]
        else:
            return 1

    @property
    @requires_shape
    def size(self):
        return reduce(operator.mul, self.shape)

    @property
    def is_homogeneous(self):
        """True if all the elements of the array are the same."""
        hom_base = isinstance(self.base_value, (int, long, float, bool))
        hom_ops = all(isinstance(obj.base_value, (int, long, float, bool))
                      for obj in self.operations if  isinstance(obj, larray))
        return hom_base and hom_ops

    def _homogeneous_array(self, addr):
        self.check_bounds(addr)
        def size(x, max):
            if isinstance(x, (int, long)):
                return 1
            elif isinstance(x, slice):
                return ((x.stop or max) - (x.start or 0)) // (x.step or 1)
            elif isinstance(x, collections.Sized):
                return len(x)
        addr = self._full_address(addr)
        shape = [size(x, max) for (x, max) in zip(addr, self.shape)]
        if shape == [1] or shape == [1, 1]:
            return 1
        else:
            shape = [x for x in shape if x > 1] # remove empty dimensions
            return numpy.ones(shape, type(self.base_value))

    def _full_address(self, addr):
        if not isinstance(addr, tuple):
            addr = (addr,)
        if len(addr) < len(self.shape):
            full_addr = [slice(None)] * len(self.shape)
            for i, val in enumerate(addr):
                full_addr[i] = val
            addr = full_addr
        return addr

    def _array_indices(self, addr):
        self.check_bounds(addr)
        def axis_indices(x, max):
            if isinstance(x, (int, long)):
                return x
            elif isinstance(x, slice): # need to handle negative values in slice
                return numpy.arange((x.start or 0),
                                    (x.stop or max),
                                    (x.step or 1),
                                    dtype=int)
            elif isinstance(x, collections.Sized):
                return x
        addr = self._full_address(addr)
        indices = [axis_indices(x, max) for (x, max) in zip(addr, self.shape)]
        if len(indices) == 1:
            return indices
        elif len(indices) == 2:
            if isinstance(indices[0], collections.Sized):
                if isinstance(indices[1], collections.Sized):
                    mesh_xy = numpy.meshgrid(*indices)
                    return (mesh_xy[0].T, mesh_xy[1].T) # meshgrid works on (x,y), not (i,j)
            return indices
        else:
            raise NotImplementedError("Only 1D and 2D arrays supported")

    @requires_shape
    def __getitem__(self, addr):
        return self._partially_evaluate(addr, simplify=False)

    def _partially_evaluate(self, addr, simplify=False):
        """
        Return part of the lazy array.
        """
        if self.is_homogeneous:
            base_val = self._homogeneous_array(addr) * self.base_value
        elif isinstance(self.base_value, numpy.ndarray):
            base_val = self.base_value[addr]
        elif callable(self.base_value):
            indices = self._array_indices(addr)
            base_val = self.base_value(*indices)
        elif isinstance(self.base_value, collections.Iterator):
            raise NotImplementedError("coming soon...")
        else:
            raise ValueError("invalid base value for array") 
        return self._apply_operations(base_val, addr, simplify=simplify)

    @requires_shape
    def check_bounds(self, addr):
        """
        Check whether the given address is within the array bounds.
        """
        def check_axis(x, size):
            if isinstance(x, (int, long)):
                lower = upper = x
            elif isinstance(x, slice):
                lower = x.start or 0
                upper = x.stop or size-1
            elif isinstance(x, collections.Sized):
                lower = min(x)
                upper = max(x)
            else:
                raise TypeError("check_bounds() requires a valid array address")
            if (lower < -size) or (upper >= size):
                raise IndexError("index out of bounds")
        addr = self._full_address(addr)
        for i, size in zip(addr, self.shape):
            check_axis(i, size)    

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

    def _apply_operations(self, x, addr=None, simplify=False):
        for f, arg in self.operations:
            if arg is None:
                x = f(x)
            elif isinstance(arg, larray):
                if addr is None:
                    x = f(x, arg.evaluate(simplify=simplify))  
                else:
                    x = f(x, arg._partially_evaluate(addr, simplify=simplify))
            else:
                x = f(x, arg)
        return x

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
        elif callable(self.base_value):
            x = numpy.fromfunction(self.base_value, shape=self.shape)
        elif isinstance(self.base_value, collections.Iterator):
            x = numpy.fromiter(self.base_value, dtype=float, count=self.size)
            if x.shape != self.shape:
                x = x.reshape(self.shape)
        else:
            raise ValueError("invalid base value for array")
        return self._apply_operations(x, simplify=simplify)

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
