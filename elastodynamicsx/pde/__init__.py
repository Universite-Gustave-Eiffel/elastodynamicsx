"""
Tools for building a Partial Differential Equation of the type:

    M*a + C*v + K(u) = F.

The package also provides tools for building Boundary Conditions.
"""


from .boundarycondition import *

from .pde import *
from .material import *
from .elasticmaterial import *
