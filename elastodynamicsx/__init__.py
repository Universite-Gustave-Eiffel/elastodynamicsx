# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""Main module for ElastodynamiCSx"""

from ._version import __version__  # noqa

import os
# A simple flag to switch the jupyter_backend when building the docs
_DOCS_CFG = os.environ.get("ELASTODYNAMICSX_DOCS_CFG", "FALSE").upper() == "TRUE"

if _DOCS_CFG:
    import pyvista
    pyvista.set_jupyter_backend('static')

from elastodynamicsx import pde, plot, solutions, solvers, utils  # noqa

__all__ = ["pde", "plot", "solutions", "solvers", "utils"]
