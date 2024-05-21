#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jochen Hinz
"""

from util import np
from quad import QuadRule, seven_point_gauss_6, univariate_gauss
from integrate import stiffness_with_diffusivity_iter, assemble_matrix_from_iterables, \
                      assemble_rhs_from_iterables, poisson_rhs_iter, \
                      assemble_neumann_rhs, mass_with_reaction_iter, transport_matrix_iter, streamline_diffusion_stabilisation_iter
from solve import solve_with_dirichlet_data
from mesh import Triangulation


def homogeneous_poisson_with_f1(mesh: Triangulation, quadrule: QuadRule):
  """
    P1 FEM solution of the poisson-type problem

      -∆u = 1    in  Ω
        u = 0    on ∂Ω

    parameters
    ----------
    mesh: `Triangulation`
      The mesh that represents the domain Ω
    quadrule: `QuadRule`
      quadrature scheme used to assemble the system matrices and right-hand-side.
  """

  f = lambda x: np.array([1])
  A = assemble_matrix_from_iterables(mesh, stiffness_with_diffusivity_iter(mesh, quadrule))
  rhs = assemble_rhs_from_iterables(mesh, poisson_rhs_iter(mesh, quadrule, f))

  bindices = mesh.boundary_indices
  data = np.zeros(bindices.shape, dtype=float)

  solution = solve_with_dirichlet_data(A, rhs, bindices, data)

  mesh.tripcolor(solution)


def homogeneous_poisson_with_g1_bottom_unit_square(quadrule1D: QuadRule,
                                                   quadrule2D: QuadRule,
                                                   mesh_size=0.05):
  """
    P1 FEM solution of the poisson-type problem

     - ∆u = 0   in Ω
     ∂n u = 1   on the bottom part of ∂Ω
        u = 0   everywhere else on ∂Ω

    where Ω = (0, 1)^2.

    parameters
    ----------
    quadrule1D: `QuadRule`
      quadrature scheme used to integrate over the mesh's boundary elements.
      quadrule.simplex_type == 'line'
    quadrule2D: `QuadRule`
      quadrature scheme used to integrate over the mesh's triangular elements.
      quadrule.simplex_type == 'triangle'
    mesh_size: float value 0 < mesh_size < 1 tuning the mesh density.
               Smaller value => denser mesh.
  """

  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=mesh_size)

  # integrate over the bottom boundary
  # line segment bottom: a ---- b
  # the y-coordinates of both a and b have to be 0
  # return True if abs(y) are both below 10^-13
  selecter = lambda a, b: abs(a[1]) < 1e-13 and abs(b[1]) < 1e-13

  # all boundary indices
  bindices = mesh.boundary_indices

  # corresponding boundary points
  bpoints = mesh.points[bindices]

  # select only the indices of bindices that lie on the top, left or right boundaries
  dindices = sorted( set(bindices[ np.abs(bpoints[:, 0] - 0) < 1e-13 ]) |  # left
                     set(bindices[ np.abs(bpoints[:, 1] - 1) < 1e-13 ]) |  # top
                     set(bindices[ np.abs(bpoints[:, 0] - 1) < 1e-13 ]) )  # right

  data = np.zeros((len(dindices),), dtype=float)

  A = assemble_matrix_from_iterables(mesh, stiffness_with_diffusivity_iter(mesh, quadrule2D))
  rhs = assemble_neumann_rhs(mesh, quadrule1D, lambda x: np.array([1]), selecter)

  # solve with homogeneous Dirichlet data imposed on the indices in dindices
  solution = solve_with_dirichlet_data(A, rhs, dindices, data)

  mesh.tripcolor(solution)


def homogeneous_poisson_with_sine_rhs(quadrule: QuadRule, mesh_size=0.025, k=1):
  """
    P1 FEM solution of the poisson-type problem

      -∆u = sin(2 k π x) sin(2 k π y)    in  Ω
        u = 0                            on ∂Ω

    where Ω = (0, 1)^2.

    parameters
    ----------
    quadrule: `QuadRule`
      quadrature scheme used to assemble the system matrices and right-hand-side.
    mesh_size: float value 0 < mesh_size < 1 tuning the mesh density.
               Smaller value => denser mesh.
    k: `int`
      the k-value of the right-hand-side

  """

  assert isinstance(k, (int, np.int_))

  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=mesh_size)

  # all boundary indices
  bindices = mesh.boundary_indices

  # the data is zero on the whole boundary
  data = np.zeros(bindices.shape, dtype=float)

  # assemble the stiffness matrix
  Aiter = stiffness_with_diffusivity_iter(mesh, quadrule)
  A = assemble_matrix_from_iterables(mesh, Aiter)

  # assemble the right-hand-side vector
  rhs_iter = poisson_rhs_iter(mesh, quadrule, lambda x: np.sin(2 * np.pi * k * x[:, 0]) * np.sin(2 * np.pi * k * x[:, 1]))
  rhs = assemble_rhs_from_iterables(mesh, rhs_iter)

  # solve with Dirichlet data
  solution = solve_with_dirichlet_data(A, rhs, bindices, data)

  # plot
  mesh.tripcolor(solution, title=r'The numerical solution of $-\Delta u = \sin(2 k \pi x) \sin(2 k \pi y)$ with $k = {}$'.format(k))


def session09_exercise01(quadrule2D: QuadRule,
                         mesh_size=0.025):

  # XXX: docstring

  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=mesh_size)

  # all boundary indices
  bindices = mesh.boundary_indices

  # corresponding boundary points
  bpoints = mesh.points[bindices]

  # data array of same length containing (initially) only zeros
  data = np.zeros(bindices.shape, dtype=float)

  # set data array to one on the left boundary
  # on the left boundary the x-coordinate of the points is machine precision 0
  left_mask = np.abs(bpoints[:, 0]) < 1e-13
  data[left_mask] = 1

  # set data array to one on the bottom boundary
  # on the bottom boundary the y-coordinate of the points is machine precision 0
  bottom_mask = np.abs(bpoints[:, 1]) < 1e-13
  data[bottom_mask] = 1

  beta = lambda x: np.array([[-1e3, -1e3]])

  Miter = mass_with_reaction_iter(mesh, quadrule2D)
  Aiter = stiffness_with_diffusivity_iter(mesh, quadrule2D)
  Biter = transport_matrix_iter(mesh, quadrule2D, beta)

  S = assemble_matrix_from_iterables(mesh, Miter, Aiter, Biter)
  rhs = np.zeros(S.shape[:1])

  solution = solve_with_dirichlet_data(S, rhs, bindices, data)

  mesh.tripcolor(solution, title='The solution without stabilisation')

  Miter = mass_with_reaction_iter(mesh, quadrule2D)
  Aiter = stiffness_with_diffusivity_iter(mesh, quadrule2D)
  Biter = transport_matrix_iter(mesh, quadrule2D, beta)
  Stabiter = streamline_diffusion_stabilisation_iter(mesh, quadrule2D, beta, gamma=5)

  S = assemble_matrix_from_iterables(mesh, Miter, Aiter, Biter, Stabiter)

  solution = solve_with_dirichlet_data(S, rhs, bindices, data)

  mesh.tripcolor(solution, title='The solution with stabilisation')


if __name__ == '__main__':
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=0.05)
  mesh.plot()
  quadrule2D = seven_point_gauss_6()
  quadrule1D = univariate_gauss(4)

  # homogeneous_poisson_with_f1(mesh, quadrule2D)
  # homogeneous_poisson_with_g1_bottom_unit_square(quadrule1D, quadrule2D)
  # homogeneous_poisson_with_sine_rhs(quadrule2D)
  session09_exercise01(quadrule2D)
