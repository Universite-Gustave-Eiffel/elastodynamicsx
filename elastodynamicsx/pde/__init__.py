# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
Tools for building a Partial Differential Equation from material laws. The
package also provides tools for building Boundary Conditions.

The PDE is of the form:
    M*a + C*v + K(u) = F.

--- --- ---

Material laws:
    from elastodynamicsx.pde import material
    mat = material( (function_space, cell_tags, marker), type_, *args, **kwargs)

Body forces:
    from elastodynamicsx.pde import BodyForce
    bf  = BodyForce( (function_space, cell_tags, marker), value)

Boundary conditions:
    from elastodynamicsx.pde import BoundaryCondition
    bc  = BoundaryCondition( (function_space, facet_tags, marker), type_, value)

Assembling a PDE:
    from elastodynamicsx.pde import PDE
    pde = PDE(materials=[mat1, mat2, ...], bodyforces=[bf1, bf2, ...])
"""

from .boundarycondition import *

from .pde import *
from .materials import *
from .bodyforce import *
from .timeschemes import *
