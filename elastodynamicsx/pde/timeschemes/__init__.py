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

from .timeschemebase import TimeScheme, FEniCSxTimeScheme
from .leapfrog import LeapFrog
from .newmark import GalphaNewmarkBeta, HilberHughesTaylor, NewmarkBeta, MidPoint, LinearAccelerationMethod

all_timeschemes_explicit = [LeapFrog]
all_timeschemes_implicit = [GalphaNewmarkBeta, HilberHughesTaylor, NewmarkBeta, MidPoint, LinearAccelerationMethod]
all_timeschemes = all_timeschemes_explicit + all_timeschemes_implicit


def timescheme(*args, **kwargs):
    """
    Builder method that instanciates the desired time scheme

    Args:
        *args: Passed to the required time scheme

    Keyword Args:
        scheme: (mandatory) String identifier of the time scheme
        **kwargs: Passed to the required material

    Returns:
        An instance of the desired time scheme
    """
    scheme = kwargs.pop('scheme', 'unknown').lower()
    for s_ in all_timeschemes:
        if scheme in s_.labels:
            return s_(*args, **kwargs)

    raise TypeError('unknown scheme: ' + scheme)


__all__ = ["timescheme",
           "all_timeschemes_explicit", "all_timeschemes_implicit", "all_timeschemes",
           "TimeScheme", "FEniCSxTimeScheme",
           "LeapFrog",
           "GalphaNewmarkBeta", "HilberHughesTaylor", "NewmarkBeta", "MidPoint", "LinearAccelerationMethod"]
