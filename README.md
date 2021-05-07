# Lattice
A python interface for some lattice models

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://mit-license.org)

## Features
This library provides a simple interface to get matrix elements associated with some simple,
fermionic lattice models:
  - Hubbard models with Peierls phase
  - Anderson model

## Warning\!
  - The scope of this library is limited
  - No effort has been made to optimize performance
  - Testing is limited, some code may not behave as expected
  - In particular, the provided full-CI code will not work for general applications
  - This use of library is not recommended for most applications

## Tests
The tests we do have can be run as follows:
  - Individually from the `lattice/tests` subdirectory
  - All at once by running `python test_suites.py` from `lattice/tests`
  - All at once by running `python -m unittest test.py`
