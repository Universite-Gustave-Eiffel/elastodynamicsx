"""
Tools for solving PDEs

Time domain:
    from elastodynamicsx.solvers import TimeStepper

Normal modes:
    from elastodynamicsx.solvers import ElasticResonanceSolver

... in the future
Frequency domain:
    from elastodynamicsx.solvers import ...

Guided waves:
    from elastodynamicsx.solvers import ...
"""

from .timestepper import *
from .eigensolver import *

