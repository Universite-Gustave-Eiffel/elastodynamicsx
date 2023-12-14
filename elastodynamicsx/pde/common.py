# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT


class PDECONFIG:
    # The default metadata used by all measures (dx, ds, dS, ...) in the classes
    # of the pde package: Material, BoundaryCondition, BodyForce
    # Example: Spectral Element Method with GLL elements of degree 6
    # >>> from elastodynamicsx.pde import PDECONFIG
    # >>> from elastodynamicsx.utils import spectral_quadrature
    # >>> specmd = spectral_quadrature("GLL", 6)
    # >>> CONFIG.default_metadata = specmd
    #
    default_metadata: dict = None

    # The default options for the just-in-time compiler used in the classes
    # of the pde package: PDE, FEniCSxTimeScheme
    # See:
    #     https://jsdokken.com/dolfinx-tutorial/chapter4/compiler_parameters.html
    # Example:
    # >>> from elastodynamicsx.pde import PDECONFIG
    # >>> PDECONFIG.default_jit_options = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
    # >>>                                  "cffi_libraries": ["m"]}
    default_jit_options: dict = {}
