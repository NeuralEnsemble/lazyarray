# encoding: utf-8
"""
Unit tests for ``larray`` class

Copyright Andrew P. Davison, Joël Chavas, Elodie Legouée (CNRS) and Ankur Sinha (UCL), 2012-2022
"""

from lazyarray import larray, VectorizedIterable, sqrt, partial_shape
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import operator
from copy import deepcopy
import pytest
from scipy.sparse import bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dia_matrix, dok_matrix, lil_matrix



class MockRNG(VectorizedIterable):

    def __init__(self, start, delta):
        self.start = start
        self.delta = delta

    def next(self, n):
        s = self.start
        self.start += n * self.delta
        return s + self.delta * np.arange(n)


# test larray
def test_create_with_int():
    A = larray(3, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3


def test_create_with_int_and_dtype():
    A = larray(3, shape=(5,), dtype=float)
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3


def test_create_with_float():
    A = larray(3.0, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3.0


def test_create_with_list():
    A = larray([1, 2, 3], shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), np.array([1, 2, 3]))


def test_create_with_array():
    A = larray(np.array([1, 2, 3]), shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), np.array([1, 2, 3]))


def test_create_with_array_and_dtype():
    A = larray(np.array([1, 2, 3]), shape=(3,), dtype=int)
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), np.array([1, 2, 3]))


def test_create_with_generator():
    def plusone():
        i = 0
        while True:
            yield i
            i += 1
    A = larray(plusone(), shape=(5, 11))
    assert_array_equal(A.evaluate(),
                       np.arange(55).reshape((5, 11)))


def test_create_with_function1D():
    A = larray(lambda i: 99 - i, shape=(3,))
    assert_array_equal(A.evaluate(),
                       np.array([99, 98, 97]))


def test_create_with_function1D_and_dtype():
    A = larray(lambda i: 99 - i, shape=(3,), dtype=float)
    assert_array_equal(A.evaluate(),
                       np.array([99.0, 98.0, 97.0]))


def test_create_with_function2D():
    A = larray(lambda i, j: 3 * j - 2 * i, shape=(2, 3))
    assert_array_equal(A.evaluate(),
                       np.array([[0, 3, 6],
                                    [-2, 1, 4]]))


def test_create_inconsistent():
    pytest.raises(ValueError, larray, [1, 2, 3], shape=4)


def test_create_with_string():
    pytest.raises(TypeError, larray, "123", shape=3)


def test_create_with_larray():
    A = 3 + larray(lambda i: 99 - i, shape=(3,))
    B = larray(A, shape=(3,), dtype=int)
    assert_array_equal(B.evaluate(),
                       np.array([102, 101, 100]))


## For sparse matrices
def test_create_with_sparse_array():
    row = np.array([0, 2, 2, 0, 1, 2])
    col = np.array([0, 0, 1, 2, 2, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    bsr = larray(bsr_matrix((data, (row, col)), shape=(3, 3))) # For bsr_matrix
    coo = larray(coo_matrix((data, (row, col)), shape=(3, 3))) # For coo_matrix
    csc = larray(csc_matrix((data, (row, col)), shape=(3, 3))) # For csc_matrix
    csr = larray(csr_matrix((data, (row, col)), shape=(3, 3))) # For csr_matrix
    data_dia = np.array([[1, 2, 3, 4]]).repeat(3, axis=0) # For dia_matrix
    offsets_dia = np.array([0, -1, 2]) # For dia_matrix
    dia = larray(dia_matrix((data_dia, offsets_dia), shape=(4, 4))) # For dia_matrix
    dok = larray(dok_matrix(((row, col)), shape=(3, 3))) # For dok_matrix
    lil = larray(lil_matrix(data, shape=(3, 3))) # For lil_matrix
    assert bsr.shape == (3, 3)
    assert coo.shape == (3, 3)
    assert csc.shape == (3, 3)
    assert csr.shape == (3, 3)
    assert dia.shape == (4, 4)
    assert dok.shape == (2, 6)
    assert lil.shape == (1, 6)

    def test_evaluate_with_sparse_array():
        assert_array_equal(bsr.evaluate(), bsr_matrix((data, (row, col))).toarray()) # For bsr_matrix
        assert_array_equal(coo.evaluate(), coo_matrix((data, (row, col))).toarray()) # For coo_matrix
        assert_array_equal(csc.evaluate(), csc_matrix((data, (row, col))).toarray()) # For csc_matrix
        assert_array_equal(csr.evaluate(), csr_matrix((data, (row, col))).toarray()) # For csr_matrix
        assert_array_equal(dia.evaluate(), dia_matrix((data_dia, (row, col))).toarray()) # For dia_matrix
        assert_array_equal(dok.evaluate(), dok_matrix((data, (row, col))).toarray()) # For dok_matrix
        assert_array_equal(lil.evaluate(), lil_matrix((data, (row, col))).toarray()) # For lil_matrix

    def test_multiple_operations_with_sparse_array():
        # For bsr_matrix
        bsr0 = bsr /100.0
        bsr1 = 0.2 + bsr0
        assert_array_almost_equal(bsr0.evaluate(), np.array([[0.01, 0., 0.04], [0., 0., 0.05], [0.02, 0.03, 0.06]]))
        assert_array_almost_equal(bsr0.evaluate(), np.array([[0.21, 0.2, 0.24], [0.2, 0.2, 0.25], [0.22, 0.23, 0.26]]))
        # For coo_matrix
        coo0 = coo /100.0
        coo1 = 0.2 + coo0
        assert_array_almost_equal(coo0.evaluate(), np.array([[0.01, 0., 0.04], [0., 0., 0.05], [0.02, 0.03, 0.06]]))
        assert_array_almost_equal(coo0.evaluate(), np.array([[0.21, 0.2, 0.24], [0.2, 0.2, 0.25], [0.22, 0.23, 0.26]]))
        # For csc_matrix
        csc0 = csc /100.0
        csc1 = 0.2 + csc0
        assert_array_almost_equal(csc0.evaluate(), np.array([[0.01, 0., 0.04], [0., 0., 0.05], [0.02, 0.03, 0.06]]))
        assert_array_almost_equal(csc0.evaluate(), np.array([[0.21, 0.2, 0.24], [0.2, 0.2, 0.25], [0.22, 0.23, 0.26]]))
        # For csr_matrix
        csr0 = csr /100.0
        csr1 = 0.2 + csr0
        assert_array_almost_equal(csc0.evaluate(), np.array([[0.01, 0., 0.04], [0., 0., 0.05], [0.02, 0.03, 0.06]]))
        assert_array_almost_equal(csc0.evaluate(), np.array([[0.21, 0.2, 0.24], [0.2, 0.2, 0.25], [0.22, 0.23, 0.26]]))
        # For dia_matrix
        dia0 = dia /100.0
        dia1 = 0.2 + dia0
        assert_array_almost_equal(dia0.evaluate(), np.array([[0.01, 0.02, 0.03, 0.04]]))
        assert_array_almost_equal(dia1.evaluate(), np.array([[0.21, 0.22, 0.23, 0.24]]))
         # For dok_matrix
        dok0 = dok /100.0
        dok1 = 0.2 + dok0
        assert_array_almost_equal(dok0.evaluate(), np.array([[0., 0.02, 0.02, 0., 0.01, 0.02], [0., 0., 0.01, 0.02, 0.02, 0.02]]))
        assert_array_almost_equal(dok1.evaluate(), np.array([[0.2, 0.22, 0.22, 0.2, 0.21, 0.22], [0.2, 0.2, 0.21, 0.22, 0.22, 0.22]]))
         # For lil_matrix
        lil0 = lil /100.0
        lil1 = 0.2 + lil0
        assert_array_almost_equal(lil0.evaluate(), np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]))
        assert_array_almost_equal(lil1.evaluate(), np.array([[0.21, 0.22, 0.23, 0.24, 0.25, 0.26]]))


    def test_getitem_from_2D_sparse_array():
        pytest.raises(IndexError, bsr.__getitem__, (3, 0))
        pytest.raises(IndexError, coo.__getitem__, (3, 0))
        pytest.raises(IndexError, csc.__getitem__, (3, 0))
        pytest.raises(IndexError, csr.__getitem__, (3, 0))
        pytest.raises(IndexError, dia.__getitem__, (3, 0))
        pytest.raises(IndexError, dok.__getitem__, (3, 0))
        pytest.raises(IndexError, lil.__getitem__, (3, 0))


# def test_columnwise_iteration_with_flat_array():
# m = larray(5, shape=(4,3)) # 4 rows, 3 columns
#    cols = [col for col in m.by_column()]
#    assert cols == [5, 5, 5]
#
# def test_columnwise_iteration_with_structured_array():
#    input = np.arange(12).reshape((4,3))
# m = larray(input, shape=(4,3)) # 4 rows, 3 columns
#    cols = [col for col in m.by_column()]
#    assert_array_equal(cols[0], input[:,0])
#    assert_array_equal(cols[2], input[:,2])
#
# def test_columnwise_iteration_with_function():
#    input = lambda i,j: 2*i + j
#    m = larray(input, shape=(4,3))
#    cols = [col for col in m.by_column()]
#    assert_array_equal(cols[0], np.array([0, 2, 4, 6]))
#    assert_array_equal(cols[1], np.array([1, 3, 5, 7]))
#    assert_array_equal(cols[2], np.array([2, 4, 6, 8]))
#
# def test_columnwise_iteration_with_flat_array_and_mask():
# m = larray(5, shape=(4,3)) # 4 rows, 3 columns
#    mask = np.array([True, False, True])
#    cols = [col for col in m.by_column(mask=mask)]
#    assert cols == [5, 5]
#
# def test_columnwise_iteration_with_structured_array_and_mask():
#    input = np.arange(12).reshape((4,3))
# m = larray(input, shape=(4,3)) # 4 rows, 3 columns
#    mask = np.array([False, True, True])
#    cols = [col for col in m.by_column(mask=mask)]
#    assert_array_equal(cols[0], input[:,1])
#    assert_array_equal(cols[1], input[:,2])


def test_size_related_properties():
    m1 = larray(1, shape=(9, 7))
    m2 = larray(1, shape=(13,))
    m3 = larray(1)
    assert m1.nrows == 9
    assert m1.ncols == 7
    assert m1.size == 63
    assert m2.nrows == 13
    assert m2.ncols == 1
    assert m2.size == 13
    pytest.raises(ValueError, lambda: m3.nrows)
    pytest.raises(ValueError, lambda: m3.ncols)
    pytest.raises(ValueError, lambda: m3.size)


def test_evaluate_with_flat_array():
    m = larray(5, shape=(4, 3))
    assert_array_equal(m.evaluate(), 5 * np.ones((4, 3)))


def test_evaluate_with_structured_array():
    input = np.arange(12).reshape((4, 3))
    m = larray(input, shape=(4, 3))
    assert_array_equal(m.evaluate(), input)


def test_evaluate_with_functional_array():
    input = lambda i, j: 2 * i + j
    m = larray(input, shape=(4, 3))
    assert_array_equal(m.evaluate(),
                       np.array([[0, 1, 2],
                                    [2, 3, 4],
                                    [4, 5, 6],
                                    [6, 7, 8]]))


def test_evaluate_with_vectorized_iterable():
    input = MockRNG(0, 1)
    m = larray(input, shape=(7, 3))
    assert_array_equal(m.evaluate(),
                       np.arange(21).reshape((7, 3)))


def test_evaluate_twice_with_vectorized_iterable():
    input = MockRNG(0, 1)
    m1 = larray(input, shape=(7, 3)) + 3
    m2 = larray(input, shape=(7, 3)) + 17
    assert_array_equal(m1.evaluate(),
                       np.arange(3, 24).reshape((7, 3)))
    assert_array_equal(m2.evaluate(),
                       np.arange(38, 59).reshape((7, 3)))


def test_evaluate_structured_array_size_1_simplify():
    m = larray([5.0], shape=(1,))
    assert m.evaluate(simplify=True) == 5.0
    n = larray([2.0], shape=(1,))
    assert (m/n).evaluate(simplify=True) == 2.5


def test_iadd_with_flat_array():
    m = larray(5, shape=(4, 3))
    m += 2
    assert_array_equal(m.evaluate(), 7 * np.ones((4, 3)))
    assert m.base_value == 5
    assert m.evaluate(simplify=True) == 7


def test_add_with_flat_array():
    m0 = larray(5, shape=(4, 3))
    m1 = m0 + 2
    assert m1.evaluate(simplify=True) == 7
    assert m0.evaluate(simplify=True) == 5


def test_lt_with_flat_array():
    m0 = larray(5, shape=(4, 3))
    m1 = m0 < 10
    assert m1.evaluate(simplify=True) is True
    assert m0.evaluate(simplify=True) == 5


def test_lt_with_structured_array():
    input = np.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    m1 = m0 < 5
    assert_array_equal(m1.evaluate(simplify=True), input < 5)


def test_structured_array_lt_array():
    input = np.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    comparison = 5 * np.ones((4, 3))
    m1 = m0 < comparison
    assert_array_equal(m1.evaluate(simplify=True), input < comparison)


def test_rsub_with_structured_array():
    m = larray(np.arange(12).reshape((4, 3)))
    assert_array_equal((11 - m).evaluate(),
                       np.arange(11, -1, -1).reshape((4, 3)))


def test_inplace_mul_with_structured_array():
    m = larray((3 * x for x in range(4)), shape=(4,))
    m *= 7
    assert_array_equal(m.evaluate(),
                       np.arange(0, 84, 21))


def test_abs_with_structured_array():
    m = larray(lambda i, j: i - j, shape=(3, 4))
    assert_array_equal(abs(m).evaluate(),
                       np.array([[0, 1, 2, 3],
                                    [1, 0, 1, 2],
                                    [2, 1, 0, 1]]))


def test_multiple_operations_with_structured_array():
    input = np.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    m1 = (m0 + 2) < 5
    m2 = (m0 < 5) + 2
    assert_array_equal(m1.evaluate(simplify=True), (input + 2) < 5)
    assert_array_equal(m2.evaluate(simplify=True), (input < 5) + 2)
    assert_array_equal(m0.evaluate(simplify=True), input)


def test_multiple_operations_with_functional_array():
    m = larray(lambda i: i, shape=(5,))
    m0 = m / 100.0
    m1 = 0.2 + m0
    assert_array_almost_equal(m0.evaluate(), np.array([0.0, 0.01, 0.02, 0.03, 0.04]), decimal=12)
    assert_array_almost_equal(m1.evaluate(), np.array([0.20, 0.21, 0.22, 0.23, 0.24]), decimal=12)
    assert m1[0] == 0.2


def test_operations_combining_constant_and_structured_arrays():
    m0 = larray(10, shape=(5,))
    m1 = larray(np.arange(5))
    m2 = m0 + m1
    assert_array_almost_equal(m2.evaluate(), np.arange(10, 15))


def test_apply_function_to_constant_array():
    f = lambda m: 2 * m + 3
    m0 = larray(5, shape=(4, 3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert m1.evaluate(simplify=True) == 13
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert m1.operations == [(operator.mul, 2), (operator.add, 3)]


def test_apply_function_to_structured_array():
    f = lambda m: 2 * m + 3
    input = np.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_array_equal(m1.evaluate(simplify=True), input * 2 + 3)


def test_apply_function_to_functional_array():
    input = lambda i, j: 2 * i + j
    m0 = larray(input, shape=(4, 3))
    f = lambda m: 2 * m + 3
    m1 = f(m0)
    assert_array_equal(m1.evaluate(),
                       np.array([[3, 5, 7],
                                    [7, 9, 11],
                                    [11, 13, 15],
                                    [15, 17, 19]]))


def test_add_two_constant_arrays():
    m0 = larray(5, shape=(4, 3))
    m1 = larray(7, shape=(4, 3))
    m2 = m0 + m1
    assert m2.evaluate(simplify=True) == 12
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert m2.base_value == m0.base_value
    assert m2.operations == [(operator.add, m1)]


def test_add_incommensurate_arrays():
    m0 = larray(5, shape=(4, 3))
    m1 = larray(7, shape=(5, 3))
    pytest.raises(ValueError, m0.__add__, m1)


def test_getitem_from_2D_constant_array():
    m = larray(3, shape=(4, 3))
    assert m[0, 0] == m[3, 2] == m[-1, 2] == m[-4, 2] == m[2, -3] == 3
    pytest.raises(IndexError, m.__getitem__, (4, 0))
    pytest.raises(IndexError, m.__getitem__, (2, -4))


def test_getitem_from_1D_constant_array():
    m = larray(3, shape=(43,))
    assert m[0] == m[42] == 3


def test_getitem__with_slice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[:3, 0],
                       np.array([3, 3, 3]))


def test_getitem__with_thinslice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert m[2:3, 0:1] == 3


def test_getitem__with_mask_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[1, (0, 2)],
                       np.array([3, 3]))


def test_getitem_with_numpy_integers_from_2D_constant_array():
    if not hasattr(np, "int64"):
        pytest.skip("test requires a 64-bit system")
    m = larray(3, shape=(4, 3))
    assert m[np.int64(0), np.int32(0)] == 3


def test_getslice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[:2],
                       np.array([[3, 3, 3],
                                    [3, 3, 3]]))


def test_getslice_past_bounds_from_constant_array():
    m = larray(3, shape=(5,))
    assert_array_equal(m[2:10],
                       np.array([3, 3, 3]))


def test_getitem_from_structured_array():
    m = larray(3 * np.ones((4, 3)), shape=(4, 3))
    assert m[0, 0] == m[3, 2] == m[-1, 2] == m[-4, 2] == m[2, -3] == 3
    pytest.raises(IndexError, m.__getitem__, (4, 0))
    pytest.raises(IndexError, m.__getitem__, (2, -4))


def test_getitem_from_2D_functional_array():
    m = larray(lambda i, j: 2 * i + j, shape=(6, 5))
    assert m[5, 4] == 14


def test_getitem_from_1D_functional_array():
    m = larray(lambda i: i ** 3, shape=(6,))
    assert m[5] == 125


def test_getitem_from_3D_functional_array():
    m = larray(lambda i, j, k: i + j + k, shape=(2, 3, 4))
    pytest.raises(NotImplementedError, m.__getitem__, (0, 1, 2))


def test_getitem_from_vectorized_iterable():
    input = MockRNG(0, 1)
    m = larray(input, shape=(7,))
    m3 = m[3]
    assert isinstance(m3, (int, np.integer))
    assert m3 == 0
    assert m[0] == 1


def test_getitem_with_slice_from_2D_functional_array():
    m = larray(lambda i, j: 2 * i + j, shape=(6, 5))
    assert_array_equal(m[2:5, 3:],
                       np.array([[7, 8],
                                    [9, 10],
                                    [11, 12]]))


def test_getitem_with_slice_from_2D_functional_array_2():
    def test_function(i, j):
        return i * i + 2 * i * j + 3
    m = larray(test_function, shape=(3, 15))
    assert_array_equal(m[:, 3:14:3],
                       np.fromfunction(test_function, shape=(3, 15))[:, 3:14:3])


def test_getitem_with_mask_from_2D_functional_array():
    a = np.arange(30).reshape((6, 5))
    m = larray(lambda i, j: 5 * i + j, shape=(6, 5))
    assert_array_equal(a[[2, 3], [3, 4]],
                       np.array([13, 19]))
    assert_array_equal(m[[2, 3], [3, 4]],
                       np.array([13, 19]))


def test_getitem_with_mask_from_1D_functional_array():
    m = larray(lambda i: np.sqrt(i), shape=(10,))
    assert_array_equal(m[[0, 1, 4, 9]],
                       np.array([0, 1, 2, 3]))


def test_getitem_with_boolean_mask_from_1D_functional_array():
    m = larray(lambda i: np.sqrt(i), shape=(10,))
    assert_array_equal(m[np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)],
                       np.array([0, 1, 2, 3]))


def test_getslice_from_2D_functional_array():
    m = larray(lambda i, j: 2 * i + j, shape=(6, 5))
    assert_array_equal(m[1:3],
                       np.array([[2, 3, 4, 5, 6],
                                    [4, 5, 6, 7, 8]]))


def test_getitem_from_iterator_array():
    m = larray(iter([1, 2, 3]), shape=(3,))
    pytest.raises(NotImplementedError, m.__getitem__, 2)


def test_getitem_from_array_with_operations():
    a1 = np.array([[1, 3, 5], [7, 9, 11]])
    m1 = larray(a1)
    f = lambda i, j: np.sqrt(i * i + j * j)
    a2 = np.fromfunction(f, shape=(2, 3))
    m2 = larray(f, shape=(2, 3))
    a3 = 3 * a1 + a2
    m3 = 3 * m1 + m2
    assert_array_equal(a3[:, (0, 2)],
                       m3[:, (0, 2)])


def test_evaluate_with_invalid_base_value():
    m = larray(range(5))
    m.base_value = "foo"
    pytest.raises(ValueError, m.evaluate)


def test_partially_evaluate_with_invalid_base_value():
    m = larray(range(5))
    m.base_value = "foo"
    pytest.raises(ValueError, m._partially_evaluate, 3)


def test_check_bounds_with_invalid_address():
    m = larray([[1, 3, 5], [7, 9, 11]])
    pytest.raises(TypeError, m.check_bounds, (object(), 1))


def test_check_bounds_with_invalid_address2():
    m = larray([[1, 3, 5], [7, 9, 11]])
    pytest.raises(ValueError, m.check_bounds, ([], 1))


def test_partially_evaluate_constant_array_with_one_element():
    m = larray(3, shape=(1,))
    a = 3 * np.ones((1,))
    m1 = larray(3, shape=(1, 1))
    a1 = 3 * np.ones((1, 1))
    m2 = larray(3, shape=(1, 1, 1))
    a2 = 3 * np.ones((1, 1, 1))
    assert a[0] == m[0]
    assert a.shape == m.shape
    assert a[:].shape == m[:].shape
    assert a == m.evaluate()
    assert a1.shape == m1.shape
    assert a1[0, :].shape == m1[0, :].shape
    assert a1[:].shape == m1[:].shape
    assert a1 == m1.evaluate()
    assert a2.shape == m2.shape
    assert a2[:, 0, :].shape == m2[:, 0, :].shape
    assert a2[:].shape == m2[:].shape
    assert a2 == m2.evaluate()


def test_partially_evaluate_constant_array_with_boolean_index():
    m = larray(3, shape=(4, 5))
    a = 3 * np.ones((4, 5))
    addr_bool = np.array([True, True, False, False, True])
    addr_int = np.array([0, 1, 4])
    assert a[::2, addr_bool].shape == a[::2, addr_int].shape
    assert a[::2, addr_int].shape == m[::2, addr_int].shape
    assert a[::2, addr_bool].shape == m[::2, addr_bool].shape


def test_partially_evaluate_constant_array_with_all_boolean_indices_false():
    m = larray(3, shape=(3,))
    a = 3 * np.ones((3,))
    addr_bool = np.array([False, False, False])
    assert a[addr_bool].shape == m[addr_bool].shape


def test_partially_evaluate_constant_array_with_only_one_boolean_indice_true():
    m = larray(3, shape=(3,))
    a = 3 * np.ones((3,))
    addr_bool = np.array([False, True, False])
    assert a[addr_bool].shape == m[addr_bool].shape
    assert m[addr_bool][0] == a[0]


def test_partially_evaluate_constant_array_with_boolean_indice_as_random_valid_ndarray():
    m = larray(3, shape=(3,))
    a = 3 * np.ones((3,))
    addr_bool = np.random.rand(3) > 0.5
    while not addr_bool.any():
        # random array, but not [False, False, False]
        addr_bool = np.random.rand(3) > 0.5
    assert a[addr_bool].shape == m[addr_bool].shape
    assert m[addr_bool][0] == a[addr_bool][0]


def test_partially_evaluate_constant_array_size_one_with_boolean_index_true():
    m = larray(3, shape=(1,))
    a = np.array([3])
    addr_bool = np.array([True])
    m1 = larray(3, shape=(1, 1))
    a1 = 3 * np.ones((1, 1))
    addr_bool1 = np.array([[True]], ndmin=2)
    assert m[addr_bool][0] == a[0]
    assert m[addr_bool] == a[addr_bool]
    assert m[addr_bool].shape == a[addr_bool].shape
    assert m1[addr_bool1][0] == a1[addr_bool1][0]
    assert m1[addr_bool1].shape == a1[addr_bool1].shape


def test_partially_evaluate_constant_array_size_two_with_boolean_index_true():
    m2 = larray(3, shape=(1, 2))
    a2 = 3 * np.ones((1, 2))
    addr_bool2 = np.ones((1, 2), dtype=bool)
    assert m2[addr_bool2][0] == a2[addr_bool2][0]
    assert m2[addr_bool2].shape == a2[addr_bool2].shape


def test_partially_evaluate_constant_array_size_one_with_boolean_index_false():
    m = larray(3, shape=(1,))
    m1 = larray(3, shape=(1, 1))
    a = np.array([3])
    a1 = np.array([[3]], ndmin=2)
    addr_bool = np.array([False])
    addr_bool1 = np.array([[False]], ndmin=2)
    addr_bool2 = np.array([False])
    assert m[addr_bool].shape == a[addr_bool].shape
    assert m1[addr_bool1].shape == a1[addr_bool1].shape


def test_partially_evaluate_constant_array_size_with_empty_boolean_index():
    m = larray(3, shape=(1,))
    a = np.array([3])
    addr_bool = np.array([], dtype='bool')
    assert m[addr_bool].shape == a[addr_bool].shape
    assert m[addr_bool].shape == (0,)


def test_partially_evaluate_functional_array_with_boolean_index():
    m = larray(lambda i, j: 5 * i + j, shape=(4, 5))
    a = np.arange(20.0).reshape((4, 5))
    addr_bool = np.array([True, True, False, False, True])
    addr_int = np.array([0, 1, 4])
    assert a[::2, addr_bool].shape == a[::2, addr_int].shape
    assert a[::2, addr_int].shape == m[::2, addr_int].shape
    assert a[::2, addr_bool].shape == m[::2, addr_bool].shape


def test_getslice_with_vectorized_iterable():
    input = MockRNG(0, 1)
    m = larray(input, shape=(7, 3))
    assert_array_equal(m[::2, (0, 2)],
                       np.arange(8).reshape((4, 2)))


def test_equality_with_lazyarray():
    m1 = larray(42.0, shape=(4, 5)) / 23.0 + 2.0
    m2 = larray(42.0, shape=(4, 5)) / 23.0 + 2.0
    assert m1 == m2


def test_equality_with_number():
    m1 = larray(42.0, shape=(4, 5))
    m2 = larray([42, 42, 42])
    m3 = larray([42, 42, 43])
    m4 = larray(42.0, shape=(4, 5)) + 2
    assert m1 == 42.0
    assert m2 == 42
    assert m3 != 42
    pytest.raises(Exception, m4.__eq__, 44.0)


def test_equality_with_array():
    m1 = larray(42.0, shape=(4, 5))
    target = 42.0 * np.ones((4, 5))
    pytest.raises(TypeError, m1.__eq__, target)


def test_deepcopy():
    m1 = 3 * larray(lambda i, j: 5 * i + j, shape=(4, 5)) + 2
    m2 = deepcopy(m1)
    m1.shape = (3, 4)
    m3 = deepcopy(m1)
    assert m1.shape == m3.shape == (3, 4)
    assert m2.shape == (4, 5)
    assert_array_equal(m1.evaluate(), m3.evaluate())


def test_deepcopy_with_ufunc():
    m1 = sqrt(larray([x ** 2 for x in range(5)]))
    m2 = deepcopy(m1)
    m1.base_value[0] = 49
    assert_array_equal(m1.evaluate(), np.array([7, 1, 2, 3, 4]))
    assert_array_equal(m2.evaluate(), np.array([0, 1, 2, 3, 4]))


def test_set_shape():
    m = larray(42) + larray(lambda i: 3 * i)
    assert m.shape is None
    m.shape = (5,)
    assert_array_equal(m.evaluate(), np.array([42, 45, 48, 51, 54]))


def test_call():
    A = larray(np.array([1, 2, 3]), shape=(3,)) - 1
    B = 0.5 * larray(lambda i: 2 * i, shape=(3,))
    C = B(A)
    assert_array_equal(C.evaluate(), np.array([0, 1, 2]))
    assert_array_equal(A.evaluate(), np.array([0, 1, 2]))  # A should be unchanged


def test_call2():
    positions = np.array(
        [[0.,  2.,  4.,  6.,  8.],
         [0.,  0.,  0.,  0.,  0.],
         [0.,  0.,  0.,  0.,  0.]])

    def position_generator(i):
        return positions.T[i]

    def distances(A, B):
        d = A - B
        d **= 2
        d = np.sum(d, axis=-1)
        np.sqrt(d, d)
        return d

    def distance_generator(f, g):
        def distance_map(i, j):
            return distances(f(i), g(j))
        return distance_map
    distance_map = larray(distance_generator(position_generator, position_generator),
                          shape=(4, 5))
    f_delay = 1000 * larray(lambda d: 0.1 * (1 + d), shape=(4, 5))
    assert_array_almost_equal(
        f_delay(distance_map).evaluate(),
        np.array([[100, 300, 500, 700, 900],
                     [300, 100, 300, 500, 700],
                     [500, 300, 100, 300, 500],
                     [700, 500, 300, 100, 300]], dtype=float),
        decimal=12)
    # repeat, should be idempotent
    assert_array_almost_equal(
        f_delay(distance_map).evaluate(),
        np.array([[100, 300, 500, 700, 900],
                     [300, 100, 300, 500, 700],
                     [500, 300, 100, 300, 500],
                     [700, 500, 300, 100, 300]], dtype=float),
        decimal=12)


def test__issue4():
    # In order to avoid the errors associated with version changes of numpy, mask1 and mask2 no longer contain boolean values ​​but integer values
    a = np.arange(12).reshape((4, 3))
    b = larray(np.arange(12).reshape((4, 3)))
    mask1 = (slice(None), int(True))
    mask2 = (slice(None), np.array([int(True)]))
    assert b[mask1].shape == partial_shape(mask1, b.shape) == a[mask1].shape
    assert b[mask2].shape == partial_shape(mask2, b.shape) == a[mask2].shape


def test__issue3():
    a = np.arange(12).reshape((4, 3))
    b = larray(a)
    c = larray(lambda i, j: 3*i + j, shape=(4, 3))
    assert_array_equal(a[(1, 3), :][:, (0, 2)], b[(1, 3), :][:, (0, 2)])
    assert_array_equal(b[(1, 3), :][:, (0, 2)], c[(1, 3), :][:, (0, 2)])
    assert_array_equal(a[(1, 3), (0, 2)], b[(1, 3), (0, 2)])
    assert_array_equal(b[(1, 3), (0, 2)], c[(1, 3), (0, 2)])


def test_partial_shape():
    a = np.arange(12).reshape((4, 3))
    test_cases = [
        (slice(None), (4, 3)),
        ((slice(None), slice(None)), (4, 3)),
        (slice(1, None, 2), (2, 3)),
        (1, (3,)),
        ((1, slice(None)), (3,)),
        ([0, 2, 3], (3, 3)),
        (np.array([0, 2, 3]), (3, 3)),
        ((np.array([0, 2, 3]), slice(None)), (3, 3)),
        (np.array([True, False, True, True]), (3, 3)),
        #(np.array([True, False]), (1, 3)),  # not valid with NumPy 1.13
        (np.array([[True, False, False], [False, False, False], [True, True, False], [False, True, False]]), (4,)),
        #(np.array([[True, False, False], [False, False, False], [True, True, False]]), (3,)),  # not valid with NumPy 1.13
        ((3, 1), tuple()),
        ((slice(None), 1), (4,)),
        ((slice(None), slice(1, None, 3)), (4, 1)),
        ((np.array([0, 3]), 2), (2,)),
        ((np.array([0, 3]), np.array([1, 2])), (2,)),
        ((slice(None), np.array([2])), (4, 1)),
        (((1, 3), (0, 2)), (2,)),
        (np.array([], bool), (0, 3)),
    ]
    for mask, expected_shape in test_cases:
        assert partial_shape(mask, a.shape) == a[mask].shape
        assert partial_shape(mask, a.shape) == expected_shape
    b = np.arange(5)
    test_cases = [
        (np.arange(5), (5,))
    ]
    for mask, expected_shape in test_cases:
        assert partial_shape(mask, b.shape) == b[mask].shape
        assert partial_shape(mask, b.shape) == expected_shape

def test_is_homogeneous():
    m0 = larray(10, shape=(5,))
    m1 = larray(np.arange(1, 6))
    m2 = m0 + m1
    m3 = 9 + m0 / m1
    assert m0.is_homogeneous
    assert not m1.is_homogeneous
    assert not m2.is_homogeneous
    assert not m3.is_homogeneous
