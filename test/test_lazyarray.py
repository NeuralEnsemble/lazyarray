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
    assert A.value == 3

def test_create_with_float():
    A = larray(3.0, shape=(5,))
    assert A.shape == (5,)
    assert A.value == 3.0

def test_create_with_list():
    A = larray([1,2,3], shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.value, numpy.array([1,2,3]))

def test_create_with_array():
    A = larray(numpy.array([1,2,3]), shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.value, numpy.array([1,2,3]))

def test_create_inconsistent():
    assert_raises(AssertionError, larray, [1,2,3], shape=4)

def test_create_with_string():
    assert_raises(AssertionError, larray, "123", shape=3)
    
def test_setitem_nonexpanded_same_value():
    A = larray(3, shape=(5,))
    assert A.value == 3
    A[0] = 3
    assert A.value == 3

def test_setitem_invalid_value():
    A = larray(3, shape=(5,))
    assert_raises(TypeError, A, "abc")

def test_setitem_nonexpanded_different_value():
    A = larray(3, shape=(5,))
    assert A.value == 3
    A[0] = 4; A[4] = 5
    assert_array_equal(A.value, numpy.array([4, 3, 3, 3, 5]))

def test_columnwise_iteration_with_flat_array():
    m = larray(5, shape=(4,3)) # 4 rows, 3 columns
    cols = [col for col in m.by_column()]
    assert_equal(cols, [5, 5, 5])

def test_columnwise_iteration_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m = larray(input, shape=(4,3)) # 4 rows, 3 columns
    cols = [col for col in m.by_column()]    
    assert_array_equal(cols[0], input[:,0])
    assert_array_equal(cols[2], input[:,2])

#def test_columnwise_iteration_with_random_array_parallel_safe_no_mask():
#    random.mpi_rank=0; random.num_processes=2
#    input = random.RandomDistribution(rng=MockRNG(parallel_safe=True))
#    copy_input = random.RandomDistribution(rng=MockRNG(parallel_safe=True))
#    m = larray(input, shape=(4,3))
#    cols = [col for col in m.by_column()]
#    assert_array_equal(cols[0], copy_input.next(4, mask_local=False))
#    assert_array_equal(cols[1], copy_input.next(4, mask_local=False))
#    assert_array_equal(cols[2], copy_input.next(4, mask_local=False))
    
def test_columnwise_iteration_with_function():
    input = lambda i,j: 2*i + j
    m = larray(input, shape=(4,3))
    cols = [col for col in m.by_column()]
    assert_array_equal(cols[0], numpy.array([0, 2, 4, 6]))
    assert_array_equal(cols[1], numpy.array([1, 3, 5, 7]))
    assert_array_equal(cols[2], numpy.array([2, 4, 6, 8]))
    
def test_columnwise_iteration_with_flat_array_and_mask():
    m = larray(5, shape=(4,3)) # 4 rows, 3 columns
    mask = numpy.array([True, False, True])
    cols = [col for col in m.by_column(mask=mask)]
    assert_equal(cols, [5, 5])
    
def test_columnwise_iteration_with_structured_array_and_mask():
    input = numpy.arange(12).reshape((4,3))
    m = larray(input, shape=(4,3)) # 4 rows, 3 columns
    mask = numpy.array([False, True, True])
    cols = [col for col in m.by_column(mask=mask)]    
    assert_array_equal(cols[0], input[:,1])
    assert_array_equal(cols[1], input[:,2])

#def test_columnwise_iteration_with_random_array_parallel_safe_with_mask():
#    random.mpi_rank=0; random.num_processes=2
#    input = random.RandomDistribution(rng=MockRNG(parallel_safe=True))
#    copy_input = random.RandomDistribution(rng=MockRNG(parallel_safe=True))
#    m = larray(input, shape=(4,3))
#    mask = numpy.array([False, False, True])
#    cols = [col for col in m.by_column(mask=mask)]
#    assert_equal(len(cols), 1)
#    assert_array_almost_equal(cols[0], copy_input.next(12, mask_local=False)[8:], 15)

def test_as_array_with_flat_array():
    m = larray(5, shape=(4,3))
    assert_array_equal(m.as_array(), 5*numpy.ones((4,3)))

def test_as_array_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m = larray(input, shape=(4,3))
    assert_array_equal(m.as_array(), input)

def test_as_array_with_functional_array():
    input = lambda i,j: 2*i + j
    m = larray(input, shape=(4,3))
    assert_array_equal(m.as_array(),
                        numpy.array([[0, 1, 2],
                                     [2, 3, 4],
                                     [4, 5, 6],
                                     [6, 7, 8]]))

def test_iadd_with_flat_array():
    m = larray(5, shape=(4,3))
    m += 2
    assert_array_equal(m.as_array(), 7*numpy.ones((4,3)))
    assert_equal(m.base_value, 5)
    assert_equal(m.value, 7)

def test_add_with_flat_array():
    m0 = larray(5, shape=(4,3))
    m1 = m0 + 2
    assert_equal(m1.value, 7)
    assert_equal(m0.value, 5)

def test_lt_with_flat_array():
    m0 = larray(5, shape=(4,3))
    m1 = m0 < 10
    assert_equal(m1.value, True)
    assert_equal(m0.value, 5)
    
def test_lt_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    m1 = m0 < 5
    assert_array_equal(m1.value, input < 5)
    
def test_structured_array_lt_array():
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    comparison = 5*numpy.ones((4,3))
    m1 = m0 < comparison
    assert_array_equal(m1.value, input < comparison)

def test_multiple_operations_with_structured_array():
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    m1 = (m0 + 2) < 5
    m2 = (m0 < 5) + 2
    assert_array_equal(m1.value, (input+2)<5)
    assert_array_equal(m2.value, (input<5)+2)
    assert_array_equal(m0.value, input)

def test_apply_function_to_constant_array():
    f = lambda m: 2*m + 3
    m0 = larray(5, shape=(4,3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_equal(m1.value, 13)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m1.operations, [(operator.mul, 2), (operator.add, 3)])

def test_apply_function_to_structured_array():
    f = lambda m: 2*m + 3
    input = numpy.arange(12).reshape((4,3))
    m0 = larray(input, shape=(4,3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_array_equal(m1.value, input*2 + 3)

def test_apply_function_to_functional_array():
    input = lambda i,j: 2*i + j
    m0 = larray(input, shape=(4,3))
    f = lambda m: 2*m + 3
    m1 = f(m0)
    assert_array_equal(m1.as_array(),
                        numpy.array([[3, 5, 7],
                                     [7, 9, 11],
                                     [11, 13, 15],
                                     [15, 17, 19]]))

def test_add_two_constant_arrays():
    m0 = larray(5, shape=(4,3))
    m1 = larray(7, shape=(4,3))
    m2 = m0 + m1
    assert_equal(m2.value, 12)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m2.base_value, m0.base_value)
    assert_equal(m2.operations, [(operator.add, m1)])
    
def test_add_incommensurate_arrays():
    m0 = larray(5, shape=(4,3))
    m1 = larray(7, shape=(5,3))
    assert_raises(ValueError, m0.__add__, m1)
    
def test_getitem_from_constant_array():
    m = larray(3, shape=(4,3))
    assert m[0,0] == m[3,2] == m[-1,2] == m[-4,2] == m[2,-3] == 3
    assert_raises(IndexError, m.__getitem__, (4,0))
    assert_raises(IndexError, m.__getitem__, (2,-4))
    
def test_getitem_from_constant_array():
    m = larray(3*numpy.ones((4,3)), shape=(4,3))
    assert m[0,0] == m[3,2] == m[-1,2] == m[-4,2] == m[2,-3] == 3
    assert_raises(IndexError, m.__getitem__, (4,0))
    assert_raises(IndexError, m.__getitem__, (2,-4))
    
