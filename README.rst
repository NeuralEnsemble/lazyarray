
=========
lazyarray
=========

lazyarray is a Python package that provides a lazily-evaluated numerical array
class, ``larray``, based on and compatible with NumPy arrays.

Lazy evaluation means that any operations on the array (potentially including
array construction) are not performed immediately, but are delayed until
evaluation is specifically requested. Evaluation of only parts of the array is
also possible.

Use of an ``larray`` can potentially save considerable computation time
and memory in cases where:

* arrays are used conditionally (i.e. there are cases in which the array is
  never used)
* only parts of an array are used (for example in distributed computation,
  in which each MPI node operates on a subset of the elements of the array)


.. image:: https://readthedocs.org/projects/lazyarray/badge/?version=latest
   :target: http://lazyarray.readthedocs.io/en/latest/

.. image:: https://github.com/NeuralEnsemble/lazyarrays/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralEnsemble/lazyarray/actions

.. image:: https://coveralls.io/repos/github/NeuralEnsemble/lazyarray/badge.svg?branch=master
   :target: https://coveralls.io/github/NeuralEnsemble/lazyarray?branch=master
