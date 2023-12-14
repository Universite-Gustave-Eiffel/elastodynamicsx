# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
.. role:: python(code)
  :language: python

The *pde* module contains tools for building a Partial Differential Equation
from material and damping laws, as well as boundary conditions.

The PDE is of the form:
    | M*a + C*v + K(u) = b,
    | + Boundary conditions.

The module also contains tools to formulate a time domain problem using implicit
or explicit time schemes, although the preferred way to use these tools is through
the :python:`elastodynamicsx.solvers.timestepper` function.

--- --- ---

Material laws:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.pde import material
      mat = material( (function_space, cell_tags, marker), type_, *args, **kwargs)

Body forces:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.pde import BodyForce
      bf  = BodyForce( (function_space, cell_tags, marker), value)

Boundary conditions:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.pde import BoundaryCondition
      bc  = BoundaryCondition( (function_space, facet_tags, marker), type_, value)

Assembling a PDE:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.pde import PDE
      pde = PDE(materials=[mat1, mat2, ...], bodyforces=[bf1, bf2, ...], bcs=[bc1, bc2, ...])

Getting the forms:
    .. highlight:: python
    .. code-block:: python

      # Python functions
      m = pde.m
      # Use as:
      # u = ufl.TrialFunction(function_space)
      # v = ufl.TestFunction(function_space)
      # m_ufl = m(u,v)

      # Compiled dolfinx forms
      m_form = pde.m_form

      # PETSc Matrices
      M = pde.M()
"""

from .boundarycondition import *
from .pde import *
from .materials import *
from .bodyforce import *
from .timeschemes import *

# from . import boundarycondition
# from . import bodyforce
# from . import timeschemes
# from . import pde
# from . import materials


#__all__ = ["bodyforce", "boundarycondition", "materials", "pde", "timeschemes",
#           "default_metadata", "build_mpc"]
