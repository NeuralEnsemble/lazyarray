"""
Unit tests for ``larray``-compatible ufuncs

Copyright Andrew P. Davison, 2012
"""

from lazyarray import larray, sqrt, cos
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_sqrt_from_array():
    A = larray(numpy.array([1, 4, 9, 16, 25]))
    assert_array_equal(sqrt(A).evaluate(),
                       numpy.arange(1, 6))
            
            
def test_sqrt_from_iterator():
    A = larray(iter([1, 4, 9, 16, 25]), shape=(5,))
    assert_array_equal(sqrt(A).evaluate(),
                       numpy.arange(1, 6))
    
def test_sqrt_from_func():
    A = larray(lambda x: (x + 1)**2, shape=(5,))
    assert_array_equal(sqrt(A).evaluate(),
                       numpy.arange(1, 6))
    
def test_sqrt_normal_array():
    A = numpy.array([1, 4, 9, 16, 25])
    assert_array_equal(sqrt(A),
                       numpy.arange(1, 6))
    
def test_cos_from_generator():
    def clock():
        for x in numpy.arange(0, 2*numpy.pi, numpy.pi/2):
            yield x
    A = larray(clock(), shape=(2, 2))
    assert_array_almost_equal(cos(A).evaluate(),
                              numpy.array([[1.0, 0.0],
                                           [-1.0, 0.0]]),
                              decimal=15)