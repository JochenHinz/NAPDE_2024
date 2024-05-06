#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from util import np
from solve import solve_with_dirichlet_data
from mesh import Triangulation

from scipy import sparse

import sys
from typing import Callable
from functools import partial, lru_cache


BAR_LENGTH = 100


class ProgressBar:
  """ For monitoring the progress of a loop. """

  def __init__(self, prefix='Task', suffix='completed'):
    self.prefix = str(prefix).strip()
    self.suffix = str(suffix).strip()

  def __call__(self, factor):
    filled_up_Length = int(round(BAR_LENGTH * factor))
    percentage = round(100.0 * factor, 1)
    bar = '=' * filled_up_Length + '-' * (BAR_LENGTH - filled_up_Length)
    sys.stdout.write('%s [%s] %s%s %s.\r' % (self.prefix, bar, percentage, '%', self.suffix))
    sys.stdout.flush()

  def __del__(self):
    sys.stdout.write('\n')


def thetamethod(M: sparse.spmatrix,
                A: sparse.spmatrix,
                timestep: Callable,
                lhs0: np.ndarray,
                T: float,
                *,
                theta: float = None,
                ft: Callable = None,
                freezeindices: np.ndarray = None,
                data: Callable = None,
                solverargs: dict = None):
  """
    Approximately solve a problem of the form:
    ∂t u(t) = F(u(t), ∂x u(t), ∂x^2 u(t)) + f(t)
    subject to spatial BCs and an initial condition u(0) = u^0,
    using the thetamethod.

    Here, F(., ., .) is linear in all its arguments and
    can hence be represented by a linear operator `A` over a
    finite-dimensional FEM basis.

    With 0 <= theta <= 1, the spatial and temporal discretisation
    can be cast into the form

    M (x^{n+1} - x^n) / dt^n = (1 - theta) * (A x^n + f^n) + theta * (A x^{n+1} + f^{n+1})
    (subject to initial and boundary conditions)

    Parameters
    ----------
    M : :class:`sparse.spmatrix` or :class: `np.ndarray`
      The sparse linear operator of shape (N, N) that represents the inertia in above equation
      :class:`int` N represents the number of unknowns >>including<< the ones that are being frozen
      due to the Dirichlet data.
    A : :class:`sparse.spmatrix` or :class: `np.ndarray`
      The sparse linear operator of shape (N, N) that represents the part of F(., ., .)
      that is linear in u(t) and its spatial derivatives.
    timestep : :class:`Callable` or :class:`Numbers.real`
      A function that returns a positive timestep as a function of all previous solution vectors
      [u^0, u^1, ..., u^{n-1}], or a positive real number that represents a constant timestep.
    lhs0 : :class:`np.ndarray`
      Array-like of shape (N,) representing the initial condition.
    T : :class:`Numbers.real`
      A positive real number that represents the time-instance after which the recursion is terminated.
    theta : :class:`Numbers.real`
      A real number 0 <= theta <= 1 that represents the theta parameter in the thetamethod.
    ft : :class:`Callable` or :class:`np.ndarray`, optional.
      A callable source-term as a function of time. Represents f(t) in the problem's classical form.
      If not of type :type:`Callable` it must either be a :class:`np.ndarray` of shape (N,), representing
      a constant source term or `None` in which case it is converted into an array of zeros.
    freezeindices : :class:`np.ndarray`
      Array-like of datatype (int, np.int_, ...) that contains the indices of the DOFs that are frozen
      by the Dirichlet data. If there's no Dirichlet boundary, it can be taken equal to `None`.
    data : :class:`Callable`
      If `freezeindices` is not None, data may not be None (and vice-versa). A :class:`Callable` that
      generates a :class:`np.ndarray` of shape freezeindices.shape as a function of time, representing
      the Dirichlet data. Can be a constant :class:`np.ndarray` of shape freezeindices.shape in case the
      Dirichlet data is not a function of time.
    solverargs : :class:`dict`
      Dictionary containing additional keyword arguments that are forwarded to the
      `solver.solve_with_dirichlet_data` routine that is called in every iteration of the recursion.
      For instance, solverargs = {'method': 'cg'} will ensure that the linear operator is inverted
      using the conjugate-gradient method in every iteration.

    Returns
    -------
    List of solution vectors and list of time-instances that correspond to the solution vectors.
  """
  assert theta is not None, 'Please specify the 0 <= theta <= 1 parameter.'
  assert 0 <= theta <= 1
  assert T > 0

  # convert to csr_matrix, if not already
  M, A = map(sparse.csr_matrix, (M, A))

  lhs0 = np.asarray(lhs0)
  assert len(lhs0.shape) == 1
  assert A.shape == M.shape
  assert A.shape[1:] == lhs0.shape

  # timestep is a scalar => convert to function that returns that scalar
  if np.isscalar(timestep):
    assert timestep > 0
    _timestep = timestep
    timestep = lambda solutions: _timestep

  # source function ft is None => source function is zero
  if ft is None:
    ft = np.zeros(lhs0.shape)

  # source function not a callable function => assert that it is a constant array instead
  if not issubclass(type(ft), Callable):
    # if the source function is not an array-like, np.asarray(.) will raise an error
    _ft = np.asarray(ft)
    # source function must have the same shape as lhs0
    assert _ft.shape == lhs0.shape
    # convert to Callable that returns constant array
    ft = lambda t: _ft

  # create cached version of the source term
  # to remember ft at t^n
  ft = lru_cache(maxsize=1)(ft)

  if freezeindices is not None:
    # there is a Dirichlet boundary => make sure that data is either a Callable or an array
    if not issubclass(type(data), Callable):
      # not a Callable => make sure it's an array-like
      _data = np.asarray(data)  # this will fail if not array-like
      # convert to Callable
      data = lambda t: _data
  else:
    # no Dirichlet boundary => freezeindices is empty and data returns np.array([])
    freezeindices = np.array([], dtype=int)
    assert data is None
    data = lambda t: np.array([], dtype=float)

  # create cached version of data array function
  # to remember data at t^n
  data = lru_cache(maxsize=1)(data)

  solverargs = dict(solverargs or {})

  # make a local function that creates the system matrix
  # S = M - dt * theta * A
  # remember the system matrix as a function of dt so it is
  # not recomputed each time in case dt is constant.
  @lru_cache(maxsize=5)
  def _S(dt):
    return M - dt * theta * A

  solutions = [lhs0]
  tn = 0
  ts = [tn]
  progressbar = ProgressBar(prefix='Time stepping scheme', suffix='completed')

  while True:

    # print progress to stdout
    progressbar(tn / T)

    # exit the loop in case t^n >= T
    if tn >= T:
      break

    # compute timestep as a function of all previous solution vectors
    dt = timestep(solutions)
    assert dt > 0

    # update the time instance
    tnp = min(tn + dt, T)

    # compute f^n and f^{n+1}
    # f^n is known from the previous iteration and remembered through caching
    fn, fnp = ft(tn), ft(tnp)

    # the Dirichlet data at time instance t = (1 - theta) * t^n + theta * t^{n+1}
    dn = (1 - theta) * data(tn) + theta * data(tnp)

    # create the system matrix. Use the _S(dt) function defined above such that
    # S is not recomputed each time in case dt is constant
    S = _S(dt)

    # create the right-hand-side vector
    # rhs = M @ x^n + dt * ((1 - theta) * (f^n + A @ x^n) + theta * f^{n+1})
    rhs = M @ solutions[-1] + dt * ((1 - theta) * (fn + A @ solutions[-1]) + theta * fnp)

    # find x^{n+1} by solving Sx = rhs with the specified Dirichlet data
    # forward the solverargs to the solve_with_dirichlet_data routine
    solutions.append(solve_with_dirichlet_data(S, rhs, freezeindices, dn, **solverargs))

    # keep track of the time-instance that corresponds to solutions[-1]
    ts.append(tnp)

    # update time instance
    tn = tnp

  return solutions, ts


# define various different time-stepping schemes by specifying
# the theta-parameter.
# This creates a new function with the same signature as `thetamethod`
# but without the theta parameter.
forward_euler = partial(thetamethod, theta=0)
implicit_euler = partial(thetamethod, theta=1)
crank_nicolson = partial(thetamethod, theta=.5)


def make_video(mesh: Triangulation, solutions, time_instances=None, filename: str = 'animation.mp4', dpi=400):
  """
    Create a video from a sequence of solution vectors that are compatible with `mesh`.

    Parameters
    ----------
    mesh : :class:`mesh.Triangulation`
      Represents the mesh on which the solutions are plotted.
    solutions : :class:`list` or :class:`tuple` or any other container-type.
      Container-like containing the solution vectors all of shape (mesh.points.shape[0],).
    time_instances : :class:`list` or `tuple` or np.array or ...
      Container-like containing the time-instances that correspond to the solution vectors.
      Optional. If not passed, defaults to np.arange(len(solutions)).
    filename : :class:`str`
      The filename under which the video is stored. Must end with `.mp4`.
    dpi : :class:`float`
      The quality of the video.
  """
  assert filename.endswith('.mp4')
  from matplotlib import animation
  from matplotlib import pyplot as plt
  from matplotlib.tri import Triangulation as pltTriangulation

  if time_instances is None:
    time_instances = np.arange(len(solutions))

  interval = 2

  vmin = min(map(np.min, solutions))
  vmax = max(map(np.max, solutions))

  (minx, miny), (maxx, maxy) = (op(mesh.points, axis=0) for op in (np.min, np.max))

  width = maxx - minx
  xcenter = (maxx + minx) / 2
  height = maxy - miny
  ycenter = (maxy + miny) / 2

  fig = plt.figure()
  ax = plt.axes(xlim=(xcenter - .6 * width, xcenter + .6 * width),
                ylim=(ycenter - .6 * height, ycenter + .6 * height))
  ax.set_axis_off()
  ax.set_aspect('equal')

  label = ax.text(xcenter - 0.2 * width,
                  ycenter + .6 * height,
                  r'$t^n = {:.3g}$'.format(0),
                  ha='left', va='center', fontsize=20, color="Red")
  tri = pltTriangulation(*mesh.points.T, mesh.triangles)
  artist = ax.tripcolor(tri, solutions[0], shading='flat',
                                           edgecolor='k',
                                           vmin=vmin,
                                           vmax=vmax)

  plt.colorbar(artist)

  artist.set_animated(True)

  progressbar = ProgressBar(prefix='Video creation', suffix='completed')

  def init():
    return artist,

  @lru_cache(maxsize=1)
  def animate(i):
    artist.set_array((solutions[i][mesh.triangles]).sum(1) / 3)
    label.set_text(r'$t^n = {:.3g}$'.format(time_instances[i]))
    progressbar(i/(float(len(solutions) - 1)))
    return artist,

  anim = animation.FuncAnimation(fig,
                                 lambda index: animate(index//interval),
                                 init_func=init,
                                 frames=len(solutions) * interval,
                                 interval=interval,
                                 blit=True)

  anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'], dpi=dpi)


if __name__ == '__main__':
  from integrate import mass_with_reaction_iter, stiffness_with_diffusivity_iter, \
                                                 assemble_matrix_from_iterables
  from quad import seven_point_gauss_6
  quadrule = seven_point_gauss_6()

  xi = np.linspace(0, 2 * np.pi, 21)[:-1]
  circle = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  mesh = Triangulation.from_polygon(circle, mesh_size=0.1)

  # make the mass and (negative) stiffness matrix
  M = assemble_matrix_from_iterables(mesh, mass_with_reaction_iter(mesh, quadrule))
  A = -.2 * assemble_matrix_from_iterables(mesh, stiffness_with_diffusivity_iter(mesh, quadrule))

  # make a zero initial condition
  lhs0 = np.zeros(mesh.points.shape[0])

  # get boundary indices
  bindices = mesh.boundary_indices
  
  # set data to sin(2πt) on the boundary
  data = lambda t: np.sin(2 * np.pi * t) * np.ones(bindices.shape)

  # make sure the first iterate satisfies the boundary condition
  lhs0[bindices] = data(0)

  solutions, time_instances = crank_nicolson(M, A, timestep=0.025,
                                                   lhs0=lhs0,
                                                   T=3,
                                                   freezeindices=bindices,
                                                   data=data,
                                                   solverargs={'method': 'cg'})  # solver: conjugate gradient

  make_video(mesh, solutions, time_instances)