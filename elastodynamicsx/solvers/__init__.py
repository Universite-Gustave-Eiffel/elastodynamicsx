# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
The *solvers* module contains tools for solving PDEs

Time domain:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.solvers import TimeStepper

Normal modes:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.solvers import EigenmodesSolver

Frequency domain:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.solvers import FrequencyDomainSolver

... in the future... Guided waves:
    .. highlight:: python
    .. code-block:: python

      from elastodynamicsx.solvers import ...
"""

from .timedomain import TimeStepper, NonlinearTimeStepper, LinearTimeStepper, OneStepTimeStepper
from .eigensolver import EigenmodesSolver
from .frequencydomain import FrequencyDomainSolver

__all__ = ["TimeStepper", "NonlinearTimeStepper", "LinearTimeStepper", "OneStepTimeStepper",
           "EigenmodesSolver",
           "FrequencyDomainSolver"]
