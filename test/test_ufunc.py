"""
Unit tests for ``larray``-compatible ufuncs

Copyright Andrew P. Davison, 2012-2017
"""

from lazyarray import larray, sqrt, cos
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_sqrt_from_array():
    A = larray(np.array([1, 4, 9, 16, 25]))
    assert_array_equal(sqrt(A).evaluate(),
                       np.arange(1, 6))


def test_sqrt_from_iterator():
    A = larray(iter([1, 4, 9, 16, 25]), shape=(5,))
    assert_array_equal(sqrt(A).evaluate(),
                       np.arange(1, 6))


def test_sqrt_from_func():
    A = larray(lambda x: (x + 1) ** 2, shape=(5,))
    assert_array_equal(sqrt(A).evaluate(),
                       np.arange(1, 6))


def test_sqrt_normal_array():
    A = np.array([1, 4, 9, 16, 25])
    assert_array_equal(sqrt(A),
                       np.arange(1, 6))


def test_cos_from_generator():
    def clock():
        for x in np.arange(0, 2 * np.pi, np.pi / 2):
            yield x
    A = larray(clock(), shape=(2, 2))
    assert_array_almost_equal(cos(A).evaluate(),
                              np.array([[1.0, 0.0],
                                           [-1.0, 0.0]]),
                              decimal=15)
