#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

import numpy as np
from functools import wraps


# shortcut for vectorisation.
# >>> arr = np.ones((10,), dtype=int)
# >>> print(arr.shape)
#     (10,)
# >>> print(arr[:, np.newaxis].shape)
#     (10, 1)
# >>> print(arr[:, _].shape)
#     (10, 1)
_ = np.newaxis


def frozen(array: np.ndarray) -> np.ndarray:
  """
    Freeze a vector inplace and return it.

    Example
    -------

    >>> arr = np.zeros((10,), dtype=int)
    >>> print(arr[0])
        0
    >>> arr[0] = 1
    >>> print(arr[0])
        1
    >>> arr = np.zeros((10,), dtype=int)
    >>> arr = frozen(arr)
    >>> arr[0] = 1
        ERROR

    Both in and out of place will work.
    >>> arr = np.zeros((10,), dtype=int)
    >>> frozen(arr)
    >>> arr[0] = 1
        ERROR
  """
  array = np.asarray(array)
  array.flags.writeable = False
  return array


def freeze(fn):
  """
    Decorator that freezes the returned array inplace.

    Example
    -------

    def multiply(arr, val):
      return val * arr

    >>> arr = np.ones((5,), dtype=int)
    >>> new_arr = multiply(arr, 2)
    >>> print(new_arr)
        [2, 2, 2, 2, 2]
    >>> new_arr[0] = 10
    >>> print(new_arr)
        [10, 2, 2, 2, 2]

    @freeze
    def multiply(arr, val):
      return val * arr

    >>> arr = np.ones((5,), dtype=int)
    >>> new_arr = multiply(arr, 2)
    >>> print(new_arr)
        [2, 2, 2, 2, 2]
    >>> new_arr[0] = 10
        ERROR
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    return frozen(fn(*args, **kwargs))
  return wrapper
