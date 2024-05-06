#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from util import np, _, freeze
import pygmsh
import meshio
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation as pltTriangulation
from typing import Callable
from scipy import sparse

# A class' member function tagged as cached property
# will remember whatever is returned the first time it's called.
# The second time it's called it will not be computed again.
# We have to make sure that whatever is returned is immutable though, meaning
# that it cannot be changed from outside of the class.
# If an array is returned, we can freeze that array for example (see below).
from functools import cached_property
from itertools import count


def abs_tuple(tpl):
  """
    [5, 6] -> (5, 6)
    [6, 5] -> (5, 6)
    (6, 5) -> (5, 6)
  """
  a, b = tpl
  if a > b: return b, a
  return tuple(tpl)


class Triangulation:

  def _refine(self):
    """
      Uniformly refine the entire mesh once.
            i1                             i1
            / \                            / \
           /   \                          /   \
          /     \         becomes      i01 ---- i12
         /       \                      / \   /  \
        /         \                    /   \ /    \
      i0 --------- i2                i0 -- i20 --- i2

      Returns
      -------
      The refined mesh of class `Triangulation`
      T: the linear operator that prolongs a vector of weights
         w.r.t. the old basis to a vector of weights w.r.t. the refined basis.
    """
    points = self.points
    slices = np.array([[0, 1], [1, 2], [2, 0]])
    all_edges = list(set(map(abs_tuple, np.concatenate(self.triangles[:, slices]))))
    newpoints = points[np.array(all_edges)].sum(1) / 2
    map_edge_number = dict(zip(all_edges, count(len(points))))

    m = len(points)
    n = m + len(map_edge_number)

    # prolongation matrix
    T = sparse.lil_matrix((n, m))

    triangles = []
    for tri in self.triangles:
      i01, i12, i20 = [map_edge_number[edge] for edge in map(abs_tuple, tri[slices])]
      i0, i1, i2 = tri
      triangles.extend([
          [i0, i01, i20],
          [i01, i1, i12],
          [i01, i12, i20],
          [i20, i12, i2]
      ])
      T[tri, tri] = 1
      T[[i01, i01, i12, i12, i20, i20], [i0, i1, i1, i2, i2, i0]] = .5

    triangles = np.array(triangles, dtype=int)

    lines = []
    for line in self.lines:
      i12 = map_edge_number[abs_tuple(line)]
      i0, i1 = line
      lines.extend([[i0, i12], [i12, i1]])

    lines = np.array(lines, dtype=int)

    points = np.concatenate([points, newpoints])

    cells = {**self.mesh.cells_dict, **{'line': lines, 'triangle': triangles}}
    mesh = self.mesh.__class__(points=np.concatenate([points, np.zeros(points.shape[:1] + (1,))], axis=1),
                               cells=cells)
    return self.__class__(mesh), T.tocsr()

  @staticmethod
  def from_polygon(*args, **kwargs):
    return mesh_from_polygon(*args, **kwargs)

  @classmethod
  def from_file(cls, filename):
    """ Load mesh from gmsh file. """
    from meshio import gmsh
    return cls(gmsh.main.read(filename))

  mesh: meshio._mesh.Mesh

  def __init__(self, mesh):
    assert isinstance(mesh, meshio._mesh.Mesh)

    simplex_names = ('line', 'triangle', 'vertex')
    if not set(mesh.cells_dict.keys()).issubset(set(simplex_names)):
      raise NotImplementedError("Expected the mesh to only contain the simplices:"
                                " '{}' but found '{}'."
                                .format(simplex_names, tuple(mesh.cells_dict.keys())))

    self.mesh = mesh

  def refine(self, n=1):
    """
      Refine the mesh `n` times and return the corresponding prolongation operator.
    """
    assert n >= 0
    ret, T = self, sparse.eye(len(self.points))
    for i in range(n):
      ret, _T = ret._refine()
      T = _T @ T
    return ret, T.tocsr()

  @property
  @freeze
  def triangles(self):
    return self.mesh.cells_dict['triangle']

  @property
  @freeze
  def lines(self):
    """ Return array ``x`` of shape (nboundaryelements, 2) where x[i] contains
        the indices of the vertices that fence-off the i-th boundary element. """
    return self.mesh.cells_dict['line']

  @cached_property
  @freeze
  def normals(self):
    # get all forward tangents
    ts = (self.points[self.lines] * np.array([-1, 1])[_, :, _]).sum(1)

    # normal is tangent[::-1] * [1, -1]
    ns = ts[:, ::-1] * np.array([[1, -1]])

    # normalise
    return ns / np.linalg.norm(ns, ord=2, axis=1)[:, _]

  @property
  @freeze
  def points(self):
    # mesh.points.shape == (npoints, 3) by default, with zeros on the last axis.
    # => ignore that part in R^2.
    return self.mesh.points[:, :2]

  def points_iter(self):
    """
      An iterator that returns the three vertices of each element.

      Example
      -------

      for (a, b, c) in mesh.points_iter():
        # do stuff with vertices a, b and c

    """
    for tri in self.triangles:
      yield self.points[tri]

  def plot(self):
    plot_mesh(self)

  @cached_property
  @freeze
  def BK(self):
    """
      Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BK[i, :, :] or, in short, mesh.BK[i] gives
      the Jacobi matrix corresponding to the i-th element.

      Example
      -------

      for i, BK in enumerate(mesh.BK):
        # do stuff with the Jacobi matrix BK corresponding to the i-th element.
    """
    a, b, c = self.points[self.triangles.T]
    # freeze the array to avoid accidentally overwriting stuff
    return np.stack([b - a, c - a], axis=2)

  @cached_property
  @freeze
  def detBK(self):
    """ Jacobian determinant (measure) per element. """
    # the np.linalg.det function returns of an array ``x`` of shape
    # (n, m, m) the determinant taken along the last two axes, i.e.,
    # in this case an array of shape (nelems,) where the i-th entry is the
    # determinant of self.BK[i]
    return np.abs(np.linalg.det(self.BK))

  @cached_property
  @freeze
  def detBK_boundary(self):
    """ Measure per boundary edge. """
    a, b = self.points[self.lines.T]
    return np.linalg.norm(b - a, ord=2, axis=1)

  @cached_property
  @freeze
  def BKinv(self):
    """
      Inverse of the Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BKinv[i, :, :] or, in short, mesh.BKinv[i] gives
      the nverse Jacobi matrix corresponding to the i-th element of shape (2, 2).
    """
    (a, b), (c, d) = self.BK.T
    return np.rollaxis(np.stack([[d, -b], [-c, a]], axis=1), -1) / self.detBK[:, _, _]

  @cached_property
  @freeze
  def boundary_indices(self):
    """ Return the sorted indices of all vertices that lie on the boundary. """
    return np.sort(np.unique(self.lines.ravel()))

  @cached_property
  def element_neighbors(self):
    """
       Return a tuple of tuples `A` where A[i] contains the indices of elements that share at
       least one vertex with the i-th element.
    """
    map_index_element = {}
    for ielem, trindices in enumerate(self.triangles):
      for index in trindices:
        map_index_element.setdefault(index, []).append(ielem)

    element_neighbors = [set() for _i in range(len(self.triangles))]
    for index, values in map_index_element.items():
      for i, val in enumerate(values):
        element_neighbors[val].update(set(values[:i] + values[i+1:]))

    return tuple(map(lambda x: tuple(sorted(x)), element_neighbors))

  def tripcolor(self, z, title=None, show=True):
    """ Plot discrete data ``z`` on the vertices of the mesh.
        Data is linearly interpolated between the vertices. """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    triang = pltTriangulation(*self.points.T, self.triangles)
    tpc = ax.tripcolor(triang, z, shading='flat', edgecolor='k')
    fig.colorbar(tpc)
    if title is not None:
      ax.set_title(title)
    if show: plt.show()
    return fig, ax


def mesh_from_polygon(points: np.ndarray, mesh_size=0.05) -> Triangulation:
  """
    create :class: ``Triangulation`` mesh from ordered set of boundary
    points.

    parameters
    ----------
    points: Array-like of shape points.shape == (npoints, 2) of boundary
            points ordered in counter-clockwise direction.
            The first point need not be repeated.
    mesh_size: Numeric value determining the density of cells.
               Smaller values => denser mesh.
      Can alternatively be a function of the form
        mesh_size = lambda dim, tag, x, y, z, _: target mesh size as function of x and y.
      For instance, mesh_size = lambda ... : 0.1 - 0.05 * np.exp(-20 * ((x - .5)**2 + (y - .5)**2))
      creates a denser mesh close to the point (x, y) = (.5, .5).
  """

  if np.isscalar(mesh_size):
    _mesh_size = mesh_size
    mesh_size = lambda *args, **kwargs: _mesh_size

  assert isinstance(type(mesh_size), Callable)

  points = np.asarray(points)
  assert points.shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(points)
    geom.set_mesh_size_callback(mesh_size)
    mesh = geom.generate_mesh(algorithm=5)

  return Triangulation(mesh)


def plot_mesh(mesh: Triangulation):
  """ Plot a mesh of type ``Triangulation``. """

  points = mesh.points
  triangles = mesh.triangles

  fig, ax = plt.subplots()

  ax.set_aspect('equal')

  ax.triplot(*points.T, triangles)
  plt.show()


def plot_function_on_mesh(mesh: Triangulation, func: Callable):
  """
    Plot scalar function ``func`` on ``mesh``.

    parameters
    ----------
    mesh: :class: ``Triangulation`` generated by, for instance ``mesh_from_polygon``.
    func: Callable func = func(x, y) wherein x and y are the mesh vertices.
          Needs to be vectorized, i.e, z = f(x, y) with x.shape == y.shape == (N,)
          means z.shape == (N,).
  """

  points = mesh.points

  # triangles representing the mesh
  # triangles is an array of shape (npoints, 3), where triangles[i] = [i0, i1, i2]
  # means that the i-th cell's vertices are, in counter-clockwise direction,
  # given by points[ [i0, i1, i2] ].
  triangles = mesh.triangles

  # create the function value to be put on the z-axis
  z = func(*points.T)

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.plot_trisurf(*points.T, z, triangles=triangles,
                                linewidth=0.5,
                                antialiased=True,
                                alpha=.8,
                                cmap=plt.cm.autumn, edgecolors='k')

  plt.show()


if __name__ == '__main__':
  xi = np.linspace(0, 2*np.pi, 41)[:-1]
  circle = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  mesh = mesh_from_polygon(circle)
  mesh.plot()

  func = lambda x, y: np.exp(-(x**2 + y**2))

  plot_function_on_mesh(mesh, func)
