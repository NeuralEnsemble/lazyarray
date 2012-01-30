"""
Unit tests for ``larray`` class

Copyright Andrew P. Davison, 2012
"""

from lazyarray import larray
import numpy
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
import operator


#class MockRNG(random.WrappedRNG):
#    rng = None
#    
#    def __init__(self, parallel_safe):
#        random.WrappedRNG.__init__(self, parallel_safe=parallel_safe)
#        self.start = 0.0
#    
#    def _next(self, distribution, n, parameters):
#        s = self.start
#        self.start += n*0.1
#        return numpy.arange(s, s+n*0.1, 0.1)


# test larray
def test_create_with_int():
    A = larray(3, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3

def test_create_with_float():
    A = larray(3.0, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3.0

def test_create_with_list():
    A = larray([1,2,3], shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), numpy.array([1,2,3]))

def test_create_with_array():
    A = larray(numpy.array([1,2,3]), shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), numpy.array([1,2,3]))

def test_create_with_generator():
    def plusone():
        i = 0
        while True:
            yield i
            i += 1
    A = larray(plusone(), shape=(5, 11))
    assert_array_equal(A.evaluate(),
                       numpy.arange(55).reshape((5, 11)))

def test_create_with_function1D():
    A = larray(lambda i: 99-i, shape=(3,))
    assert_array_equal(A.evaluate(),
                       numpy.array([99, 98, 97]))

def test_create_with_function2D():
    A = larray(lambda i,j: 3*j-2*i, shape=(2, 3))
    assert_array_equal(A.evaluate(),
                       numpy.array([[0, 3, 6],
                                    [-2, 1, 4]]))

def test_create_inconsistent():
    assert_raises(AssertionError, larray, [1,2,3], shape=4)

def test_create_with_string():
    assert_raises(AssertionError, larray, "123", shape=3)
    
#def test_columnwise_iteration_with_flat_array():
#    m = larray(5, shape=(4,3)) # 4 rows, 3 columns
#    cols = [col for col in m.by_column()]
#    assert_equal(cols, [5, 5, 5])
#
#def test_columnwise_iteration_with_structured_array():
#    input = numpy.arange(12).reshape((4,3))
#    m = larray(input, shape=(4,3)) # 4 rows, 3 columns
#    cols = [col for col in m.by_column()]    
#    assert_array_equal(cols[0], input[:,0])
#    assert_array_equal(cols[2], input[:,2])
#    
#def test_columnwise_iteration_with_function():
#    input = lambda i,j: 2*i + j
#    m = larray(input, shape=(4,3))
#    cols = [col for col in m.by_column()]
#    assert_array_equal(cols[0], numpy.array([0, 2, 4, 6]))
#    assert_array_equal(cols[1], numpy.array([1, 3, 5, 7]))
#    assert_array_equal(cols[2], numpy.array([2, 4, 6, 8]))
#    
#def test_columnwise_iteration_with_flat_array_and_mask():
#    m = larray(5, shape=(4,3)) # 4 rows, 3 columns
#    mask = numpy.array([True, False, True])
#    cols = [col for col in m.by_column(mask=mask)]
#    assert_equal(cols, [5, 5])
#    
#def test_columnwise_iteration_with_structured_array_and_mask():
#    input = numpy.arange(12).reshape((4,3))
#    m = larray(input, shape=(4,3)) # 4 rows, 3 columns
#    mask = numpy.array([False, True, True])
#    cols = [col for col in m.by_column(mask=mask)]    
#    assert_array_equal(cols[0], input[:,1])
#    assert_array_equal(cols[1], input[:,2])

def test_size_related_properties():
    m1 = larray(1, shape=(9,7))
    m2 = larray(1, shape=(13,))
    m3 = larray(1)
    assert_equal(m1.nrows, 9)
    assert_equal(m1.ncols, 7)
    assert_equal(m1.size, 63)
    assert_equal(m2.nrows, 13)
    assert_equal(m2.ncols, 1)
    assert_equal(m2.size, 13)
    assert_raises(ValueError, lambda: m3.nrows)
    assert_raises(ValueError, lambda: m3.ncols)
    assert_raises(ValueError, lambda: m3.size)

def test_evaluate_with_flat_array():
    m = larray(5, shape=(4,3))
    assert_array_equal(m.evaluate(), 5*numpy.ones((4,3)))

def test_evaluate_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m = larray(input, shape=(4,3))
    assert_array_equal(m.evaluate(), input)

def test_evaluate_with_functional_array():
    input = lambda i,j: 2*i + j
    m = larray(input, shape=(4,3))
    assert_array_equal(m.evaluate(),
                        numpy.array([[0, 1, 2],
                                     [2, 3, 4],
                                     [4, 5, 6],
                                     [6, 7, 8]]))

def test_iadd_with_flat_array():
    m = larray(5, shape=(4,3))
    m += 2
    assert_array_equal(m.evaluate(), 7*numpy.ones((4,3)))
    assert_equal(m.base_value, 5)
    assert_equal(m.evaluate(simplify=True), 7)

def test_add_with_flat_array():
    m0 = larray(5, shape=(4,3))
    m1 = m0 + 2
    assert_equal(m1.evaluate(simplify=True), 7)
    assert_equal(m0.evaluate(simplify=True), 5)

def test_lt_with_flat_array():
    m0 = larray(5, shape=(4,3))
    m1 = m0 < 10
    assert_equal(m1.evaluate(simplify=True), True)
    assert_equal(m0.evaluate(simplify=True), 5)
    
def test_lt_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    m1 = m0 < 5
    assert_array_equal(m1.evaluate(simplify=True), input < 5)
    
def test_structured_array_lt_array():
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    comparison = 5*numpy.ones((4,3))
    m1 = m0 < comparison
    assert_array_equal(m1.evaluate(simplify=True), input < comparison)

def test_rsub_with_structured_array():
    m = larray(numpy.arange(12).reshape((4, 3)))
    assert_array_equal((11 - m).evaluate(),
                       numpy.arange(11, -1, -1).reshape((4, 3)))

def test_inplace_mul_with_structured_array():
    m = larray((3*x for x in range(4)), shape=(4,))
    m *= 7
    assert_array_equal(m.evaluate(),
                       numpy.arange(0, 84, 21))

def test_abs_with_structured_array():
    m = larray(lambda i,j: i-j, shape=(3,4))
    assert_array_equal(abs(m).evaluate(),
                       numpy.array([[0, 1, 2, 3],
                                    [1, 0, 1, 2],
                                    [2, 1, 0, 1]]))

def test_multiple_operations_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    m1 = (m0 + 2) < 5
    m2 = (m0 < 5) + 2
    assert_array_equal(m1.evaluate(simplify=True), (input+2)<5)
    assert_array_equal(m2.evaluate(simplify=True), (input<5)+2)
    assert_array_equal(m0.evaluate(simplify=True), input)

def test_apply_function_to_constant_array():
    f = lambda m: 2*m + 3
    m0 = larray(5, shape=(4,3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_equal(m1.evaluate(simplify=True), 13)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m1.operations, [(operator.mul, 2), (operator.add, 3)])

def test_apply_function_to_structured_array():
    f = lambda m: 2*m + 3
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_array_equal(m1.evaluate(simplify=True), input*2 + 3)

def test_apply_function_to_functional_array():
    input = lambda i,j: 2*i + j
    m0 = larray(input, shape=(4,3))
    f = lambda m: 2*m + 3
    m1 = f(m0)
    assert_array_equal(m1.evaluate(),
                        numpy.array([[3, 5, 7],
                                     [7, 9, 11],
                                     [11, 13, 15],
                                     [15, 17, 19]]))

def test_add_two_constant_arrays():
    m0 = larray(5, shape=(4,3))
    m1 = larray(7, shape=(4,3))
    m2 = m0 + m1
    assert_equal(m2.evaluate(simplify=True), 12)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m2.base_value, m0.base_value)
    assert_equal(m2.operations, [(operator.add, m1)])
    
def test_add_incommensurate_arrays():
    m0 = larray(5, shape=(4,3))
    m1 = larray(7, shape=(5,3))
    assert_raises(ValueError, m0.__add__, m1)
    
def test_getitem_from_2D_constant_array():
    m = larray(3, shape=(4,3))
    assert m[0,0] == m[3,2] == m[-1,2] == m[-4,2] == m[2,-3] == 3
    assert_raises(IndexError, m.__getitem__, (4, 0))
    assert_raises(IndexError, m.__getitem__, (2, -4))

def test_getitem_from_1D_constant_array():
    m = larray(3, shape=(43,))
    assert m[0] == m[42] == 3

def test_getitem__with_slice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[:3, 0],
                       numpy.array([3, 3, 3]))

def test_getitem__with_thinslice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_equal(m[2:3, 0:1], 3)

def test_getitem__with_mask_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[1, (0, 2)],
                       numpy.array([3, 3]))    

def test_getslice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[:2],
                       numpy.array([[3, 3, 3],
                                    [3, 3, 3]]))

def test_getitem_from_structured_array():
    m = larray(3*numpy.ones((4,3)), shape=(4,3))
    assert m[0,0] == m[3,2] == m[-1,2] == m[-4,2] == m[2,-3] == 3
    assert_raises(IndexError, m.__getitem__, (4,0))
    assert_raises(IndexError, m.__getitem__, (2,-4))

def test_getitem_from_2D_functional_array():
    m = larray(lambda i,j: 2*i + j, shape=(6,5))
    assert_equal(m[5, 4], 14)

def test_getitem_from_1D_functional_array():
    m = larray(lambda i: i**3, shape=(6,))
    assert_equal(m[5], 125)

def test_getitem_from_3D_functional_array():
    m = larray(lambda i,j,k: i+j+k, shape=(2,3,4))
    assert_raises(NotImplementedError, m.__getitem__, (0,1,2))

def test_getitem_with_slice_from_2D_functional_array():
    m = larray(lambda i,j: 2*i + j, shape=(6,5))
    assert_array_equal(m[2:5, 3:],
                       numpy.array([[7, 8],
                                    [9, 10],
                                    [11, 12]]))

def test_getitem_with_slice_from_2D_functional_array_2():
    def test_function(i, j):
        return i*i + 2*i*j + 3
    m = larray(test_function, shape=(3,15))
    assert_array_equal(m[:, 3:14:3],
                       numpy.fromfunction(test_function, shape=(3,15))[:, 3:14:3])

def test_getitem_with_mask_from_2D_functional_array():
    m = larray(lambda i,j: 2*i + j, shape=(6,5))
    assert_array_equal(m[[2, 3, 4], [3, 4]],
                       numpy.array([[7, 8],
                                    [9, 10],
                                    [11, 12]]))

def test_getitem_with_mask_from_1D_functional_array():
    m = larray(lambda i: numpy.sqrt(i), shape=(10,))
    assert_array_equal(m[[0, 1, 4, 9]],
                       numpy.array([0, 1, 2, 3]))

def test_getslice_from_2D_functional_array():
    m = larray(lambda i,j: 2*i + j, shape=(6,5))
    assert_array_equal(m[1:3],
                       numpy.array([[2, 3, 4, 5, 6],
                                    [4, 5, 6, 7, 8]]))

def test_getitem_from_iterator_array():
    m = larray(iter([1, 2, 3]), shape=(3,))
    assert_raises(NotImplementedError, m.__getitem__, 2)

def test_getitem_from_array_with_operations():
    a1 = numpy.array([[1, 3, 5], [7, 9, 11]])
    m1 = larray(a1)
    f = lambda i,j: numpy.sqrt(i*i + j*j)
    a2 = numpy.fromfunction(f, shape=(2, 3))
    m2 = larray(f, shape=(2, 3))
    a3 = 3*a1 + a2
    m3 = 3*m1 + m2
    assert_array_equal(a3[:,(0,2)],
                       m3[:,(0,2)])

def test_evaluate_with_invalid_base_value():
    m = larray(range(5))
    m.base_value = "foo"
    assert_raises(ValueError, m.evaluate)
    
def test_partially_evaluate_with_invalid_base_value():
    m = larray(range(5))
    m.base_value = "foo"
    assert_raises(ValueError, m._partially_evaluate, 3)

def test_check_bounds_with_invalid_address():
    m = larray([[1, 3, 5], [7, 9, 11]])
    assert_raises(TypeError, m.check_bounds, (object(), 1))