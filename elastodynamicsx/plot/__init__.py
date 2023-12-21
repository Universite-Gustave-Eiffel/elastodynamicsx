# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""The *plot* module contains convenience tools for plotting"""

from .plot import live_plotter, plot_mesh, plotter, CustomScalarPlotter, CustomVectorPlotter, spy_petscMatrix, \
    _get_3D_array_from_nparray

__all__ = ["live_plotter", "plot_mesh", "plotter", "CustomScalarPlotter", "CustomVectorPlotter", "spy_petscMatrix",
           "_get_3D_array_from_nparray"]
