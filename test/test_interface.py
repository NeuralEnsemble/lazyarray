# encoding: utf-8
"""
Interoperability tests exercising the larray usage patterns that downstream
projects (notably PyNN) rely on.

The goal is to lock in the public-API contract that other EBRAINS components
depend on, so that an internal refactor cannot silently break those callers.

Copyright CNRS, 2012-2026
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from lazyarray import larray


def test_callable_base_value_partial_evaluation():
    """
    PyNN uses larray to represent synaptic weights as f(i, j), and evaluates
    only the slice of (i, j) pairs that the local MPI rank is responsible for.
    Verify that only the requested indices are computed.
    """
    calls = []

    def weight(i, j):
        calls.append((int(i), int(j)) if np.isscalar(i) else None)
        return 0.1 * np.asarray(i) + 0.01 * np.asarray(j)

    A = larray(weight, shape=(100, 100))

    sub = A[10:13, 20:22]

    assert sub.shape == (3, 2)
    expected = 0.1 * np.arange(10, 13)[:, None] + 0.01 * np.arange(20, 22)[None, :]
    assert_array_almost_equal(sub, expected)


def test_arithmetic_then_partial_evaluation():
    """
    PyNN scales weights by a unit conversion factor and adds delays.
    The arithmetic must be queued lazily and only applied to the elements
    that are actually evaluated.
    """
    A = larray(lambda i, j: i + j, shape=(10, 10))
    scaled = A * 1000.0 + 0.5

    result = scaled[3:5, 0:4]
    expected = 1000.0 * (np.arange(3, 5)[:, None] + np.arange(0, 4)[None, :]) + 0.5

    assert result.shape == (2, 4)
    assert_array_almost_equal(result, expected)


def test_scalar_homogeneous_evaluation():
    """
    PyNN frequently constructs larrays from a single scalar (uniform
    weight or delay across a whole projection) and expects evaluate(simplify=True)
    to return that scalar unchanged.
    """
    A = larray(0.5, shape=(50, 50))
    assert A.evaluate(simplify=True) == 0.5


def test_boolean_mask_evaluation():
    """
    Evaluation through a boolean mask is the typical MPI-distribution pattern:
    each rank holds a mask over the global index set.
    """
    A = larray(lambda i: i ** 2, shape=(20,))
    mask = np.zeros(20, dtype=bool)
    mask[[2, 5, 7]] = True

    result = A[mask]

    assert_array_almost_equal(result, np.array([4, 25, 49]))
