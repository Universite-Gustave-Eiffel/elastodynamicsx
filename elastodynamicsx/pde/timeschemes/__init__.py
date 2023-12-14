# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
.. role:: python(code)
  :language: python

The *timeschemes* module contains tools to construct the weak form of a time-dependent
problem. Several implicit and explicit schemes are supported. The preferred way to use
this module is however through the :python:`elastodynamicsx.solvers.timestepper` function.

List of supported schemes:
    Explicit schemes:
        - 'leapfrog'

    Implicit schemes:
        - 'midpoint'
        - 'linear-acceleration-method'
        - 'newmark'
        - 'hht-alpha'
        - 'generalized-alpha'
"""

from .timescheme import TimeScheme, FEniCSxTimeScheme
from .leapfrog import LeapFrog
from .newmark import GalphaNewmarkBeta, HilberHughesTaylor, NewmarkBeta, MidPoint, LinearAccelerationMethod

all_timeschemes_explicit = [LeapFrog]
all_timeschemes_implicit = [GalphaNewmarkBeta, HilberHughesTaylor, NewmarkBeta, MidPoint, LinearAccelerationMethod]
all_timeschemes = all_timeschemes_explicit + all_timeschemes_implicit

__all__ = ["all_timeschemes_explicit", "all_timeschemes_implicit", "all_timeschemes",
           "TimeScheme", "FEniCSxTimeScheme",
           "LeapFrog",
           "GalphaNewmarkBeta", "HilberHughesTaylor", "NewmarkBeta", "MidPoint", "LinearAccelerationMethod"]
