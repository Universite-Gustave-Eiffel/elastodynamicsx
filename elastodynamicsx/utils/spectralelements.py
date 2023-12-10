# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

# See:
# https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_lagrange_variants.py
# https://github.com/FEniCS/dolfinx/issues/2537


import typing
import basix
import basix.ufl
from dolfinx.mesh import CellType


# ## ### ###  ###
# ## Elements ###
# ## ### ###  ###

_cell_type_mesh2basix = {CellType.point        : basix.CellType.point,
                         CellType.interval     : basix.CellType.interval,
                         CellType.triangle     : basix.CellType.triangle,
                         CellType.quadrilateral: basix.CellType.quadrilateral,
                         CellType.tetrahedron  : basix.CellType.tetrahedron,
                         CellType.prism        : basix.CellType.prism,
                         CellType.pyramid      : basix.CellType.pyramid,
                         CellType.hexahedron   : basix.CellType.hexahedron
                         }


def _suitable_cell_type_format(cell_type):
    if   type(cell_type) == basix.CellType:
        return cell_type
    elif type(cell_type) == CellType:
        return _cell_type_mesh2basix[cell_type]
    elif type(cell_type) == str:
        return basix.cell.string_to_type(cell_type)
    else:
        raise TypeError("Unknown cell type: {0:s}".format(cell_type))


def GLL_element(cell_type,
                degree: int,
                shape: typing.Optional[typing.Tuple[int, ...]] = None) -> basix.ufl._BasixElement:
    """Element defined using the Gauss-Lobatto-Legendre points"""
    cell_type = _suitable_cell_type_format(cell_type)
    element = basix.ufl.element(basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.gll_warped, shape=shape)
    return element


def GL_element(cell_type,
               degree: int,
               shape: typing.Optional[typing.Tuple[int, ...]] = None) -> basix.ufl._BasixElement:
    """(discontinuous) Element defined using the Gauss-Legendre points"""
    cell_type = _suitable_cell_type_format(cell_type)
    element   = basix.ufl.element(basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.gl_warped, True)
    return element


def Legendre_element(cell_type,
                     degree: int,
                     shape: typing.Optional[typing.Tuple[int, ...]] = None) -> basix.ufl._BasixElement:
    """(discontinuous) Element whose basis functions are the orthonormal Legendre polynomials"""
    cell_type = _suitable_cell_type_format(cell_type)
    element   = basix.ufl.element(basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.legendre, True)
    return element


# ## ### ### ### ###
# ## Quadrature  ###
# ## ### ### ### ###

def GLL_quadrature(degree: int) -> dict:
    """Returns the dolfinx quadrature rule for use with GLL elements of the given degree."""
    if degree == 2:
        return {"quadrature_rule": "GLL", "quadrature_degree": 3}
    else:
        return {"quadrature_rule": "GLL", "quadrature_degree": 2 * (degree - 1)}


def GL_quadrature(degree: int) -> dict:
    """Returns the dolfinx quadrature rule for use with GL elements of the given degree."""
    return {"quadrature_rule": "GL",  "quadrature_degree": 2 * degree}  # 2 * degree + 1 ?


def Legendre_quadrature(degree: int) -> dict:
    """Returns the dolfinx quadrature rule for use with Legendre elements of the given degree."""
    return {"quadrature_rule": "GL",  "quadrature_degree": 2 * degree}


# ## ### ### ### ###
# ## ALL IN ONE  ###
# ## ### ### ### ###

def spectral_element(name: str,
                     cell_type,
                     degree: int,
                     shape: typing.Optional[typing.Tuple[int, ...]] = None) -> basix.ufl._BasixElement:
    """
    A spectral element that can be used in a dolfinx.fem.FunctionSpace

    Args:
        name: One of ("GLL", "GL", "Legendre")
        cell_type: Elements can be defined from any type, but diagonal mass matrices
            can only be obtained using GLL / GL quadratures that require cell types
            interval (1D) / quadrilateral (2D) / hexahedron (3D)
        degree: The maximum degree of basis functions

    Example:
        .. highlight:: python
        .. code-block:: python

          from mpi4py import MPI
          from dolfinx import mesh, fem
          from elastodynamicsx.utils import spectral_element, spectral_quadrature
          #
          degree = 4
          specFE = spectral_element("GLL", mesh.CellType.quadrilateral, degree)
          specmd = spectral_quadrature("GLL", degree)
          V = fem.FunctionSpace( mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=mesh.CellType.quadrilateral), specFE )

          #######
          # Compile mass matrices using pure fenicsx code
          import ufl
          u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

          # here we do not specify quadrature information -> non-diagonal mass matrix
          a1_non_diag = fem.form(ufl.inner(u,v) * ufl.dx)
          M1_non_diag = fem.petsc.assemble_matrix(a1_non_diag)
          M1_non_diag.assemble()

          # here we specify the quadrature metadata -> the matrix becomes diagonal
          a2_diag     = fem.form(ufl.inner(u,v) * ufl.dx(metadata=specmd))
          M2_diag     = fem.petsc.assemble_matrix(a2_diag)
          M2_diag.assemble()

          #######
          # Compile mass matrices using elastodynamicsx.pde classes
          from elastodynamicsx.pde import PDE, material

          # here we do not specify quadrature information -> non-diagonal mass matrix
          mat3          = material(V, 'scalar', 1, 1)
          pde3_non_diag = PDE(V, [mat3])
          M3_non_diag   = pde3_non_diag.M()

          # elastodynamicsx offers two ways to specify the quadrature metadata

          # 1. by passing the metadata as kwargs to each material / boundary condition / bodyforce

          mat4          = material(V, 'scalar', 1, 1, metadata=specmd)
          pde4_diag     = PDE(V, [mat4])
          M4_diag       = pde4_diag.M()

          # 2. by passing the metadata to the PDE.default_metadata, which is being used by
          # all materials / boundary conditions / bodyforces at __init__ call

          PDE.default_metadata = specmd # default metadata for all forms in the pde package (materials, BCs, ...)

          mat5          = material(V, 'scalar', 1, 1)
          pde5_diag     = PDE(V, [mat5])
          M5_diag       = pde5_diag.M()

          # spy mass matrices
          from elastodynamicsx.plot import spy_petscMatrix
          import matplotlib.pyplot as plt
          for i,M in enumerate([M1_non_diag, M2_diag, M3_non_diag, M4_diag, M5_diag]):
              fig = plt.figure()
              fig.suptitle('M'+str(i+1))
              spy_petscMatrix(M)
          plt.show()
    """
    if   name.lower() == "gll":
        return GLL_element(cell_type, degree, shape)
    elif name.lower() == "gl":
        return GL_element(cell_type, degree, shape)
    elif name.lower() == "legendre":
        return Legendre_element(cell_type, degree, shape)
    else:
        raise TypeError("Unknown element name: {0:s}".format(name))


def spectral_quadrature(name: str, degree: int) -> dict:
    """
    A quadrature metadata to build diagonal mass matrices when used with
    corresponding spectral elements.

    Args:
        name: One of ("GLL", "GL", "Legendre")
        degree: The maximum degree of basis functions

    Example:
        See doc of 'spectral_element'.
    """
    if   name.lower() == "gll":
        return GLL_quadrature(degree)
    elif name.lower() == "gl":
        return GL_quadrature(degree)
    elif name.lower() == "legendre":
        return Legendre_quadrature(degree)
    else:
        raise TypeError("Unknown element name: {0:s}".format(name))
