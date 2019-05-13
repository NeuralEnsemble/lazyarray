# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name='lazyarray',
    version='0.3.3',
    py_modules=['lazyarray'],
    license='Modified BSD',
    author="Andrew P. Davison",
    author_email="andrew.davison@unic.cnrs-gif.fr",
    url="http://github.com/NeuralEnsemble/lazyarray/",
    description="a Python package that provides a lazily-evaluated numerical array class, larray, based on and compatible with NumPy arrays.",
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.8",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ]
)
