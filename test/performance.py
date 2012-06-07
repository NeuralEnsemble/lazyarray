

import numpy as np
from numpy.testing import assert_array_equal
from lazyarray import larray
from timeit import repeat

def test_function(i, j):
    return i*i + 2*i*j + 3

def array_from_function_full(f, shape):
    return np.fromfunction(f, shape)
    
def larray_from_function_full(f, shape):
    return larray(f, shape).evaluate()
    
def array_from_function_slice(f, shape):
    return np.fromfunction(f, shape)[:, 0:-1:10]

def larray_from_function_slice(f, shape):
    return larray(f, shape)[:, 0:shape[1]-1:10]

    
if __name__ == "__main__":
    assert_array_equal(array_from_function_full(test_function, (5000, 5000)),
                       larray_from_function_full(test_function, (5000, 5000)))
    
    print "Array from function: full array"
    print(repeat('array_from_function_full(test_function, (5000, 5000))',
                 setup='from __main__ import array_from_function_full, test_function',
                 number=1, repeat=5))
    
    
    print "Lazy array from function: full array"
    print(repeat('larray_from_function_full(test_function, (5000, 5000))',
                 setup='from __main__ import larray_from_function_full, test_function',
                 number=1, repeat=5))
    
    assert_array_equal(array_from_function_slice(test_function, (5000, 5000)),
                       larray_from_function_slice(test_function, (5000, 5000)))
    print "Array from function: slice"
    print(repeat('array_from_function_slice(test_function, (5000, 5000))',
                 setup='from __main__ import array_from_function_slice, test_function',
                 number=1, repeat=5))
    
    print "Lazy array from function: slice"
    print(repeat('larray_from_function_slice(test_function, (5000, 5000))',
                 setup='from __main__ import larray_from_function_slice, test_function',
                 number=1, repeat=5))