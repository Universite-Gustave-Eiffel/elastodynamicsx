# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""The *utils* module contains various tools that do not fit into the other packages"""

from .tags import make_facet_tags, make_cell_tags, make_tags, _get_functionspace_tags_marker
from .eval import ParallelEvaluator
from .spectralelements import spectral_element, spectral_quadrature

__all__ = ["make_facet_tags", "make_cell_tags", "make_tags", "_get_functionspace_tags_marker",
           "ParallelEvaluator",
           "spectral_element", "spectral_quadrature"]
