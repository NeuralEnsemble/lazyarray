# encoding: utf-8
"""
lazyarray is a Python package that provides a lazily-evaluated numerical array
class, ``larray``, based on and compatible with NumPy arrays.

Copyright Andrew P. Davison, Joël Chavas and Elodie Legouée (CNRS), 2012-2020
"""

import numbers
import operator
from copy import deepcopy
import collections
from functools import wraps, reduce
import logging

import numpy as np
try:
    from scipy import sparse
    from scipy.sparse import bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dia_matrix, dok_matrix, lil_matrix
    have_scipy = True
except ImportError:
    have_scipy = False


__version__ = "0.5.0"

logger = logging.getLogger("lazyarray")


def check_shape(meth):
    """
    Decorator for larray magic methods, to ensure that the operand has
    the same shape as the array.
    """
    @wraps(meth)
    def wrapped_meth(self, val):
        if isinstance(val, (larray, np.ndarray)) and val.shape:
            if val.shape != self._shape:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        return meth(self, val)
    return wrapped_meth


def requires_shape(meth):
    @wraps(meth)
    def wrapped_meth(self, *args, **kwargs):
        if self._shape is None:
            raise ValueError("Shape of larray not specified")
        return meth(self, *args, **kwargs)
    return wrapped_meth


def full_address(addr, full_shape):
    if not (isinstance(addr, np.ndarray) and addr.dtype == bool and addr.ndim == len(full_shape)):
        if not isinstance(addr, tuple):
            addr = (addr,)
        if len(addr) < len(full_shape):
            full_addr = [slice(None)] * len(full_shape)
            for i, val in enumerate(addr):
                full_addr[i] = val
            addr = full_addr
    return addr


def partial_shape(addr, full_shape):
    """
    Calculate the size of the sub-array represented by `addr`
    """
    def size(x, max):
        if isinstance(x, (int, np.integer)):
            return None
        elif isinstance(x, slice):
            y = min(max, x.stop or max)  # slice limits can go past the bounds
            return 1 + (y - (x.start or 0) - 1) // (x.step or 1)
        elif isinstance(x, collections.Sized):
            if hasattr(x, 'dtype') and x.dtype == bool:
                return x.sum()
            else:
                return len(x)
        else:
            raise TypeError("Unsupported index type %s" % type(x))

    addr = full_address(addr, full_shape)
    if isinstance(addr, np.ndarray) and addr.dtype == bool:
        return (addr.sum(),)
    elif all(isinstance(x, collections.Sized) for x in addr):
        return (len(addr[0]),)
    else:
        shape = [size(x, max) for (x, max) in zip(addr, full_shape)]
        return tuple([x for x in shape if x is not None])  # remove empty dimensions


def reverse(func):
    """Given a function f(a, b), returns f(b, a)"""
    @wraps(func)
    def reversed_func(a, b):
        return func(b, a)
    reversed_func.__doc__ = "Reversed argument form of %s" % func.__doc__
    reversed_func.__name__ = "reversed %s" % func.__name__
    return reversed_func
# "The hash of a function object is hash(func_code) ^ id(func_globals)" ?
# see http://mail.python.org/pipermail/python-dev/2000-April/003397.html


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


def is_array_like(value):
    # False for numbers, generators, functions, iterators
    if not isinstance(value, collections.Sized):
        return False
    if sparse.issparse(value):
        return True
    if isinstance(value, collections.Mapping):
        # because we may wish to have lazy arrays in which each
        # item is a dict, for example
        return False
    if getattr(value, "is_lazyarray_scalar", False):
        # for user-defined classes that are "Sized" but that should
        # be treated as individual elements in a lazy array
        # the attribute "is_lazyarray_scalar" can be defined with value
        # True.
        return False
    return True



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


    def __init__(self, value, shape=None, dtype=None):
        """
        Create a new lazy array.

        `value` : may be an int, float, bool, NumPy array, iterator,
                  generator or a function, `f(i)` or `f(i,j)`, depending on the
                  dimensions of the array.

        `f(i,j)` should return a single number when `i` and `j` are integers,
        and a 1D array when either `i` or `j` or both is a NumPy array (in the
        latter case the two arrays must have equal lengths).
        """

        self.dtype = dtype
        self.operations = []
        if isinstance(value, str):
            raise TypeError("An larray cannot be created from a string")
        elif isinstance(value, larray):
            if shape is not None and value.shape is not None:
                assert shape == value.shape
            self._shape = shape or value.shape
            self.base_value = value.base_value
            self.dtype = dtype or value.dtype
            self.operations = value.operations  # should deepcopy?

        elif is_array_like(value):  # False for numbers, generators, functions, iterators
            if have_scipy and sparse.issparse(value):  # For sparse matrices
                self.dtype = dtype or value.dtype
            elif not isinstance(value, np.ndarray):
                value = np.array(value, dtype=dtype)
            elif dtype is not None:
               assert np.can_cast(value.dtype, dtype, casting='safe')  # or could convert value to the provided dtype
            if shape and value.shape and value.shape != shape:
                raise ValueError("Array has shape %s, value has shape %s" % (shape, value.shape))
            if value.shape:
                self._shape = value.shape
            else:
                self._shape = shape
            self.base_value = value

        else:
            assert np.isreal(value)  # also True for callables, generators, iterators
            self._shape = shape
            if dtype is None or isinstance(value, dtype):
                self.base_value = value
            else:
                try:
                    self.base_value = dtype(value)
                except TypeError:
                    self.base_value = value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.base_value == other.base_value and self.operations == other.operations and self._shape == other.shape
        elif isinstance(other, numbers.Number):
            if len(self.operations) == 0:
                if isinstance(self.base_value, numbers.Number):
                    return self.base_value == other
                elif isinstance(self.base_value, np.ndarray):
                    return (self.base_value == other).all()
            # todo: we could perform the evaluation ourselves, but that could have a performance hit
            raise Exception("You will need to evaluate this lazyarray before checking for equality")
        else:
            # todo: add support for NumPy arrays
            raise TypeError("Cannot at present compare equality of lazyarray and {}".format(type(other)))

    def __deepcopy__(self, memo):
        obj = type(self).__new__(type(self))
        if isinstance(self.base_value, VectorizedIterable):  # special case, but perhaps need to rethink
            obj.base_value = self.base_value                 # whether deepcopy is appropriate everywhere
        else:
            try:
                obj.base_value = deepcopy(self.base_value)
            except TypeError:  # base_value cannot be copied, e.g. is a generator (but see generator_tools from PyPI)
                obj.base_value = self.base_value  # so here we create a reference rather than deepcopying - could cause problems
        obj._shape = self._shape
        obj.dtype = self.dtype
        obj.operations = []
        for f, arg in self.operations:
            if isinstance(f, np.ufunc):
                obj.operations.append((f, deepcopy(arg)))
            else:
                obj.operations.append((deepcopy(f), deepcopy(arg)))
        return obj

    def __repr__(self):
        return "<larray: base_value=%r shape=%r dtype=%r, operations=%r>" % (self.base_value,
                                                                             self.shape,
                                                                             self.dtype,
                                                                             self.operations)

    def _set_shape(self, value):
        if (hasattr(self.base_value, "shape") and
                self.base_value.shape and   # values of type np.float have an empty shape
                    self.base_value.shape != value):
            raise ValueError("Lazy array has fixed shape %s, cannot be changed to %s" % (self.base_value.shape, value))
        self._shape = value
        for op in self.operations:
            if isinstance(op[1], larray):
                op[1].shape = value
    shape = property(fget=lambda self: self._shape,
                     fset=_set_shape, doc="Shape of the array")

    @property
    @requires_shape
    def nrows(self):
        """Size of the first dimension of the array."""
        return self._shape[0]

    @property
    @requires_shape
    def ncols(self):
        """Size of the second dimension (if it exists) of the array."""
        if len(self.shape) > 1:
            return self._shape[1]
        else:
            return 1

    @property
    @requires_shape
    def size(self):
        """Total number of elements in the array."""
        return reduce(operator.mul, self._shape)

    @property
    def is_homogeneous(self):
        """True if all the elements of the array are the same."""
        hom_base = isinstance(self.base_value, (int, np.integer, float, bool)) \
                   or type(self.base_value) == self.dtype \
                   or (isinstance(self.dtype, type) and isinstance(self.base_value, self.dtype))
        hom_ops = all(obj.is_homogeneous for f, obj in self.operations if isinstance(obj, larray))
        return hom_base and hom_ops

    def _partial_shape(self, addr):
        """
        Calculate the size of the sub-array represented by `addr`
        """
        return partial_shape(addr, self._shape)

    def _homogeneous_array(self, addr):
        self.check_bounds(addr)
        shape = self._partial_shape(addr)
        return np.ones(shape, type(self.base_value))

    def _full_address(self, addr):
        return full_address(addr, self._shape)

    def _array_indices(self, addr):
        self.check_bounds(addr)

        def axis_indices(x, max):
            if isinstance(x, (int, np.integer)):
                return x
            elif isinstance(x, slice):  # need to handle negative values in slice
                return np.arange((x.start or 0),
                                    (x.stop or max),
                                    (x.step or 1),
                                    dtype=int)
            elif isinstance(x, collections.Sized):
                if hasattr(x, 'dtype') and x.dtype == bool:
                    return np.arange(max)[x]
                else:
                    return np.array(x)
            else:
                raise TypeError("Unsupported index type %s" % type(x))
        addr = self._full_address(addr)
        if isinstance(addr, np.ndarray) and addr.dtype == bool:
            if addr.ndim == 1:
                return (np.arange(self._shape[0])[addr],)
            else:
                raise NotImplementedError()
        elif all(isinstance(x, collections.Sized) for x in addr):
            indices = [np.array(x) for x in addr]
            return indices
        else:
            indices = [axis_indices(x, max) for (x, max) in zip(addr, self._shape)]
            if len(indices) == 1:
                return indices
            elif len(indices) == 2:
                if isinstance(indices[0], collections.Sized):
                    if isinstance(indices[1], collections.Sized):
                        mesh_xy = np.meshgrid(*indices)
                        return (mesh_xy[0].T, mesh_xy[1].T)  # meshgrid works on (x,y), not (i,j)
                return indices
            else:
                raise NotImplementedError("Only 1D and 2D arrays supported")

    @requires_shape
    def __getitem__(self, addr):
        """
        Return one or more items from the array, as for NumPy arrays.

        `addr` may be a single integer, a slice, a NumPy boolean array or a
        NumPy integer array.
        """
        return self._partially_evaluate(addr, simplify=False)

    def _partially_evaluate(self, addr, simplify=False):
        """
        Return part of the lazy array.
        """
        if self.is_homogeneous:
            if simplify:
                base_val = self.base_value
            else:
                base_val = self._homogeneous_array(addr) * self.base_value
        elif isinstance(self.base_value, (int, np.integer, float, bool)):
            base_val = self._homogeneous_array(addr) * self.base_value
        elif isinstance(self.base_value, np.ndarray):
            base_val = self.base_value[addr]
        elif have_scipy and sparse.issparse(self.base_value):  # For sparse matrices larr[2, :]
            base_val = self.base_value[addr]
        elif callable(self.base_value):
            indices = self._array_indices(addr)
            base_val = self.base_value(*indices)
            if isinstance(base_val, np.ndarray) and base_val.shape == (1,):
                base_val = base_val[0]
        elif hasattr(self.base_value, "lazily_evaluate"):
            base_val = self.base_value.lazily_evaluate(addr, shape=self._shape)
        elif isinstance(self.base_value, VectorizedIterable):
            partial_shape = self._partial_shape(addr)
            if partial_shape:
                n = reduce(operator.mul, partial_shape)
            else:
                n = 1
            base_val = self.base_value.next(n)  # note that the array contents will depend on the order of access to elements
            if n == 1:
                base_val = base_val[0]
            elif partial_shape and base_val.shape != partial_shape:
                base_val = base_val.reshape(partial_shape)
        elif isinstance(self.base_value, collections.Iterator):
            raise NotImplementedError("coming soon...")
        else:
            raise ValueError("invalid base value for array (%s)" % self.base_value)
        return self._apply_operations(base_val, addr, simplify=simplify)

    @requires_shape
    def check_bounds(self, addr):
        """
        Check whether the given address is within the array bounds.
        """
        def is_boolean_array(arr):
            return hasattr(arr, 'dtype') and arr.dtype == bool

        def check_axis(x, size):
            if isinstance(x, (int, np.integer)):
                lower = upper = x
            elif isinstance(x, slice):
                lower = x.start or 0
                upper = min(x.stop or size - 1, size - 1)  # slices are allowed to go past the bounds
            elif isinstance(x, collections.Sized):
                if is_boolean_array(x):
                    lower = 0
                    upper = x.size - 1
                else:
                    if len(x) == 0:
                        raise ValueError("Empty address component (address was %s)" % str(addr))
                    if hasattr(x, "min"):
                        lower = x.min()
                    else:
                        lower = min(x)
                    if hasattr(x, "max"):
                        upper = x.max()
                    else:
                        upper = max(x)
            else:
                raise TypeError("Invalid array address: %s (element of type %s)" % (str(addr), type(x)))
            if (lower < -size) or (upper >= size):
                raise IndexError("Index out of bounds")
        full_addr = self._full_address(addr)
        if isinstance(addr, np.ndarray) and addr.dtype == bool:
            if len(addr.shape) > len(self._shape):
                raise IndexError("Too many indices for array")
            for xmax, size in zip(addr.shape, self._shape):
                upper = xmax - 1
                if upper >= size:
                    raise IndexError("Index out of bounds")
        else:
            for i, size in zip(full_addr, self._shape):
                check_axis(i, size)

    def apply(self, f):
        """
        Add the function `f(x)` to the list of the operations to be performed,
        where `x` will be a scalar or a numpy array.

        >>> m = larray(4, shape=(2,2))
        >>> m.apply(np.sqrt)
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
    def evaluate(self, simplify=False, empty_val=0):
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
                x = self.base_value * np.ones(self._shape, dtype=self.dtype)
        elif isinstance(self.base_value, (int, np.integer, float, bool, np.bool_)):
            x = self.base_value * np.ones(self._shape, dtype=self.dtype)
        elif isinstance(self.base_value, np.ndarray):
            x = self.base_value
        elif callable(self.base_value):
            x = np.array(np.fromfunction(self.base_value, shape=self._shape, dtype=int), dtype=self.dtype)
        elif hasattr(self.base_value, "lazily_evaluate"):
            x = self.base_value.lazily_evaluate(shape=self._shape)
        elif isinstance(self.base_value, VectorizedIterable):
            x = self.base_value.next(self.size)
            if x.shape != self._shape:
                x = x.reshape(self._shape)
        elif have_scipy and sparse.issparse(self.base_value):  # For sparse matrices
            if empty_val!=0:
                x = self.base_value.toarray((sparse.csc_matrix))
                x = np.where(x, x, np.nan)
            else:
                x = self.base_value.toarray((sparse.csc_matrix))
        elif isinstance(self.base_value, collections.Iterator):
            x = np.fromiter(self.base_value, dtype=self.dtype or float, count=self.size)
            if x.shape != self._shape:
                x = x.reshape(self._shape)
        else:
            raise ValueError("invalid base value for array")
        return self._apply_operations(x, simplify=simplify)

    def __call__(self, arg):
        if callable(self.base_value):
            if isinstance(arg, larray):
                new_map = deepcopy(arg)
            elif callable(arg):
                new_map = larray(arg)
            else:
                raise Exception("Argument must be either callable or an larray.")
            new_map.operations.append((self.base_value, None))
            new_map.operations.extend(self.operations)
            return new_map
        else:
            raise Exception("larray is not callable")

    __iadd__ = lazy_inplace_operation('add')
    __isub__ = lazy_inplace_operation('sub')
    __imul__ = lazy_inplace_operation('mul')
    __idiv__ = lazy_inplace_operation('div')
    __ipow__ = lazy_inplace_operation('pow')

    __add__ = lazy_operation('add')
    __radd__ = __add__
    __sub__ = lazy_operation('sub')
    __rsub__ = lazy_operation('sub', reversed=True)
    __mul__ = lazy_operation('mul')
    __rmul__ = __mul__
    __div__ = lazy_operation('div')
    __rdiv__ = lazy_operation('div', reversed=True)
    __truediv__ = lazy_operation('truediv')
    __rtruediv__ = lazy_operation('truediv', reversed=True)
    __pow__ = lazy_operation('pow')

    __lt__ = lazy_operation('lt')
    __gt__ = lazy_operation('gt')
    __le__ = lazy_operation('le')
    __ge__ = lazy_operation('ge')

    __neg__ = lazy_unary_operation('neg')
    __pos__ = lazy_unary_operation('pos')
    __abs__ = lazy_unary_operation('abs')


class VectorizedIterable(object):
    """
    Base class for any class which has a method `next(n)`, i.e., where you
    can choose how many values to return rather than just returning one at a
    time.
    """
    pass


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


def _build_ufunc_2nd_arg(func):
    """Return a ufunc taking a second, non-array argument, that works with lazy arrays"""
    def larray_compatible_ufunc2(x1, x2):
        if not isinstance(x2, numbers.Number):
            raise TypeError("lazyarry ufuncs do not accept an array as the second argument")
        if isinstance(x1, larray):
            def partial(x):
                return func(x, x2)
            y = deepcopy(x1)
            y.apply(partial)
            return y
        else:
            return func(x1, x2)
    return larray_compatible_ufunc2


# build lazy-array compatible versions of NumPy ufuncs
namespace = globals()
for name in dir(np):
    obj = getattr(np, name)
    if isinstance(obj, np.ufunc) and name not in namespace:
        if name in ("power", "fmod", "arctan2, hypot, ldexp, maximum, minimum"):
            namespace[name] = _build_ufunc_2nd_arg(obj)
        else:
            namespace[name] = _build_ufunc(obj)
