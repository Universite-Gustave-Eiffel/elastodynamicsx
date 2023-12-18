# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from typing import Union, Dict


class PDECONFIG:
    """
    Global configuration parameters for classes in the pde module
    """
    default_metadata: Union[Dict, None] = None
    """
    The default metadata used by all measures (dx, ds, dS, ...) in the classes
    of the pde package: Material, BoundaryCondition, BodyForce

    Example:
        Spectral Element Method with GLL elements of degree 6:

        .. highlight:: python
        .. code-block:: python

          from elastodynamicsx.pde import PDECONFIG
          from elastodynamicsx.utils import spectral_quadrature
          specmd = spectral_quadrature("GLL", 6)
          CONFIG.default_metadata = specmd
    """

    default_jit_options: Dict = {}
    """
    The default options for the just-in-time compiler used in the classes
    of the pde package: PDE, FEniCSxTimeScheme

    See:
        https://jsdokken.com/dolfinx-tutorial/chapter4/compiler_parameters.html

    Example:
        .. highlight:: python
        .. code-block:: python

            from elastodynamicsx.pde import PDECONFIG
            PDECONFIG.default_jit_options = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
                                             "cffi_libraries": ["m"]}
    """
