# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""Main module for ElastodynamiCSx"""

from ._version import __version__

from elastodynamicsx import pde, plot, solutions, solvers, utils

__all__ = ["pde", "plot", "solutions", "solvers", "utils"]
