# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name='lazyarray',
    version='0.1.0',
    py_modules=['lazyarray'],
    license='Modified BSD',
    author="Andrew P. Davison",
    author_email="andrew.davison@unic.cnrs-gif.fr",
    url="http://bitbucket.org/apdavison/lazyarray/",
    description="a Python package that provides a lazily-evaluated numerical array class, larray, based on and compatible with NumPy arrays.",
    long_description=open('README').read(),
    install_requires=[
        "numpy >= 1.5"
    ],   
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ]
)

