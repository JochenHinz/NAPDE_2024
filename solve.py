#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from util import np
from scipy import sparse
from scipy.sparse import linalg


class NanVec(np.ndarray):
  """
     Vector of dtype float initilized to np.nan.
     Used in `solve_with_dirichlet_data`.
  """

  @classmethod
  def from_indices_data(cls, length, indices, data):
    'Instantiate NanVec x of length ``length`` satisfying x[indices] = data.'
    vec = cls(length)
    vec[np.asarray(indices)] = np.asarray(data)
    return vec

  def __new__(cls, length):
    vec = np.empty(length, dtype=float).view(cls)
    vec[:] = np.nan
    return vec

  @property
  def where(self):
    """
      Return boolean mask ``mask`` of shape self.shape with mask[i] = True
      if self[i] != np.nan and False if self[i] == np.nan.
      >>> vec = NanVec(5)
      >>> vec[[1, 2, 3]] = 7
      >>> vec.where
          [False, True, True, True, False]
    """
    return ~np.isnan(self.view(np.ndarray))

  def __ior__(self, other):
    """
      Set self[~self.where] to other[~self.where].
      Other is either an array-like of self.shape or a scalar.
      If it is a scalar set self[~self.where] = other
      >>> vec = NanVec(5)
      >>> vec[[0, 4]] = 5
      >>> vec |= np.array([0, 1, 2, 3, 4])
      >>> print(vec)
          [5, 1, 2, 3, 5]

      >>> vec = NanVec(5)
      >>> vec[[0, 4]] = 5
      >>> vec |= 0
      >>> print(vec)
          [5, 0, 0, 0, 5]
    """
    wherenot = ~self.where
    self[wherenot] = other if np.isscalar(other) else other[wherenot]
    return self

  def __or__(self, other):
    """
      Same as self.__ior__ but the result is cast into a new vector, i.e.,
      z = self | other.
    """
    return self.copy().__ior__(other)


def solve_sparse_linear(A, rhs, method='direct', solverargs=None):
  """
    Solve sparse linear system.

    Parameters
    ----------

    A: `sparse.spmatrix`
      The linear operator of shape (N, N).
    rhs: `np.ndarray`
      Right-hand-side of shape (N,).
    method: `str`
      Linear solver method. Must be any of 'direct', 'cg', 'gmres', 'bicgstab'.
    solverargs: `dict` or None
      Additional keyword arguments that are forwarded to the linear solver routine.
  """

  spsolver = {'cg': linalg.cg,
              'gmres': linalg.gmres,
              'direct': linalg.spsolve,
              'bicgstab': linalg.bicgstab}.get(method, None)

  if spsolver is None:
    raise AssertionError("Unknown linear solver method '{method}'.")

  solverargs = dict(solverargs or {})

  sol = spsolver(A, rhs, **solverargs)

  # some of the solvers return a tuple containing the solution vector
  # and additional stuff that is ignored for now.
  # test if sol is an array or sparse matrix and if not, discard all but
  # the first entry.
  if not isinstance(sol, (np.ndarray, sparse.spmatrix)):
    sol, *ignore = sol

  return sol


def solve_with_dirichlet_data(A, rhs, freezeindices, data, **kwargs):
  """
    Solve sparse linear system with given Dirichlet data.

    Parameters
    ----------

    A: sparse square matrix of shape (N, N) representing the linear operator
       WITHOUT any Dirichlet data. Can be either of ``numpy.ndarray`` or
       any ``scipy.sparse.spmatrix`` type. If o ``numpy.ndarray`` type,
       it will be converted to ``scipy.sparse.spmatrix``.
    rhs: array-like representing the right-hand-side vector of shape (N,)
         without Dirichlet data.
    freezeindices: array-like of indices containing the degrees of freedom
                    that are fixed from the Dirichlet data. The shape is
                    (n,) with n <= N.
    data: array-like containing the dirichlet data, i.e., the solution vector
          index freeze_indices[i] will be equal to data[i].
          Can also be a scalar, in which case the scalar is repeated to match
          the length of `freezeindices`.
          data.shape == (n,)

    **kwargs: forwarded to ``solve_sparse_linear``

    Returns
    -------

    x: np.ndarray
      solution vector of shape (N,) with x[freeze_indices] = data

    Example
    -------
    Finite difference discretisation of the Poisson problem -Δu = f over Ω = (0, 1),
    with f = 10 and homogeneous Dirichlet data on δΩ = {0, 1}.

    >>> N = 21
    >>> h = 1 / (N - 1)
    >>> A = 1 / (h**2) sparse.diags(diagonals=[-np.ones(N-1),
                                               2 * np.ones(N),
                                               -np.ones(N-1)],
                                    offsets=[-1, 0, 1], format='lil')
    >>> rhs = 10 * np.ones(N)
    >>> freezeindices = [0, N-1]  # 0-based indexing
    >>> data = [0, 0]

    >>> sol = solve_with_dirichlet_data(A, rhs, freezeindices, data)

  """

  if not sparse.isspmatrix(A):
    A = sparse.csr_matrix(A)

  N = len(rhs)
  assert A.shape == (N, N)

  # convert the full rhs vector (i.e., the one that is tested against ALL test functions)
  # to an np.ndarray if not already.
  rhs = np.asarray(rhs)
  
  # create a vector that contains the Dirichlet data on the entries that are frozen
  # from the Dirichlet boundary condition and NaN else.
  cons = NanVec.from_indices_data(N, freezeindices, data)
  
  # cons.where gives True where cons is not None and False else
  # ~cons.where gives the opposite, i.e., False where cons is not None and True else
  # the indices that are degrees of freedom in the linear problem subject to
  # Dirichlet data are hence given by dofindices = ~cons.where
  dofindices = ~cons.where

  # this step corresponds to the step f -> f_0 - B u_D in the document
  # given cons = [5, 6, nan, 7, nan, nan, ...], cons|0 gives
  # cons|0 = [5, 6, 0, 7, 0, 0, ...] so that A @ (cons|0) represents
  # [B, D]^T u_D in the terminology of the document.
  # Now we subtract rhs - A @ (cons|0) which represents
  # f - [B, D]^T u_D and restricting to (...)[dofindices] is equivalent to
  # taking the subset ([f_0, f_D]^T - [B, D]^T u_D)[slice] = f_0 - B u_D
  rhs = (rhs - A @ (cons|0))[dofindices]

  # convert to lil-format to enable slicing
  # this step restricts A to what corresponds to \tilde{A} in the document.
  # This is done by restricting the rows and columns to only the dofindices
  A = A.tolil()[dofindices][:, dofindices].tocsr()
  
  # solve for u_0
  sol = solve_sparse_linear(A, rhs, **kwargs)

  # cons is equal to the imposed Dirichlet data on the freezeindices
  # setting cons[dofindices] = sol will create a vector that is equal to
  # the Dirichlet data on the freezeindices and to u_0 on the dofindices.
  cons[dofindices] = sol

  # return the result as an ordinary np.ndarray
  return cons.view(np.ndarray)


if __name__ == '__main__':
  from matplotlib import pyplot as plt

  N = 21
  h = 1 / (N - 1)
  x = np.arange(0, 1+h, h)
  A = 1 / (h**2) * sparse.diags(diagonals=[-np.ones(N-1),
                                           2 * np.ones(N),
                                           -np.ones(N-1)],
                                offsets=[-1, 0, 1], format='lil')

  rhs = 10 * np.ones(N)
  indices = [0, N-1]
  data = [0, 0]

  sol = solve_with_dirichlet_data(A, rhs, indices, data)

  plt.plot(x, sol)
  plt.show()
