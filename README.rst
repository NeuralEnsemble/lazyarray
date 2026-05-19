
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

.. image:: https://github.com/NeuralEnsemble/lazyarray/actions/workflows/test.yml/badge.svg
   :target: https://github.com/NeuralEnsemble/lazyarray/actions

.. image:: https://coveralls.io/repos/github/NeuralEnsemble/lazyarray/badge.svg?branch=master
   :target: https://coveralls.io/github/NeuralEnsemble/lazyarray?branch=master


Documentation
=============

Full documentation is available at https://lazyarray.readthedocs.io,
including a tutorial, API reference, and a developers' guide.


Communication
=============

* General questions and discussion: the
  `NeuralEnsemble Google Group <http://groups.google.com/group/neuralensemble>`_.
* Bug reports and feature requests: the
  `GitHub issue tracker <https://github.com/NeuralEnsemble/lazyarray/issues>`_.

Contributions are welcome — see the ``CONTRIBUTING.md`` file at the root
of the repository and the
`developers' guide <https://lazyarray.readthedocs.io/en/latest/developers.html>`_
for details. All contributors are expected to follow the
`Code of Conduct <https://github.com/NeuralEnsemble/lazyarray/blob/master/CODE_OF_CONDUCT.md>`_.


License
=======

lazyarray is released under the BSD-3-Clause license; see the ``LICENSE``
file for the full text.
