"""
Tools for solving PDEs

Time domain:
    from elastodynamicsx.solvers import TimeStepper

Normal modes:
    from elastodynamicsx.solvers import ElasticResonanceSolver

Frequency domain:
    from elastodynamicsx.solvers import FrequencyDomainSolver

... in the future
Guided waves:
    from elastodynamicsx.solvers import ...
"""

from .timestepper import *
from .eigensolver import *
from .frequencydomain import *
