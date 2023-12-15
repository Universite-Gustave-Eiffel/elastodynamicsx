# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import numpy as np
from dolfinx.fem import FunctionSpaceBase
from dolfinx.mesh import Mesh, locate_entities, meshtags, MeshTags

from typing import Tuple, Callable, Union


def make_facet_tags(domain: Mesh, boundaries: Tuple[Tuple[int, Callable]]) -> MeshTags:
    """Shortcut for make_tags(domain, locators, type_='boundaries')"""
    return make_tags(domain, boundaries, type_='boundaries')


def make_cell_tags(domain: Mesh, subdomains: Tuple[Tuple[int, Callable]]) -> MeshTags:
    """Shortcut for make_tags(domain, locators, type_='domains')"""
    return make_tags(domain, subdomains, type_='domains')


def make_tags(domain: Mesh, locators: Tuple[Tuple[int, Callable]], type_='unknown') -> MeshTags:
    """
    Args:
        domain: A mesh
        locators: A tuple of tuples of the type (tag, fn), where:
            tag: is an int to be assigned to the cells or facets
            fn: is a boolean function that take 'x' as argument
        type_: either 'domains' or 'boundaries'

    Returns:
        A MeshTags object (dolfinx.mesh)

    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html

    Example:
        .. highlight:: python
        .. code-block:: python

          import numpy as np
          from mpi4py import MPI
          from dolfinx.mesh import create_unit_square
          #
          domain = create_unit_square(MPI.COMM_WORLD, 10, 10)

          boundaries = ((1, lambda x: np.isclose(x[0], 0)),
                        (2, lambda x: np.isclose(x[0], 1)),
                        (3, lambda x: np.isclose(x[1], 0)),
                        (4, lambda x: np.isclose(x[1], 1)))
          facet_tags = make_tags(domain, boundaries, 'boundaries')

          Omegas = ((1, lambda x: x[1] <= 0.5),
                    (2, lambda x: x[1] >= 0.5))
          cell_tags = make_tags(domain, Omegas, 'domains')
    """
    if type_.lower() == 'boundaries':
        fdim = domain.topology.dim - 1
    elif type_.lower() == 'domains':
        fdim = domain.topology.dim
    else:
        raise TypeError("Unknown type: {0:s}".format(type_))

    loc_indices, loc_markers = [], []
    for (marker, locator) in locators:
        loc = locate_entities(domain, fdim, locator)
        loc_indices.append(loc)
        loc_markers.append(np.full_like(loc, marker))

    loc_indices = np.hstack(loc_indices).astype(np.int32)
    loc_markers = np.hstack(loc_markers).astype(np.int32)
    sorted_loc = np.argsort(loc_indices)
    loc_tags = meshtags(domain, fdim, loc_indices[sorted_loc], loc_markers[sorted_loc])

    return loc_tags


def get_functionspace_tags_marker(functionspace_tags_marker:
                                  Union[FunctionSpaceBase, Tuple[FunctionSpaceBase, MeshTags, int]]
                                  ) -> Tuple[FunctionSpaceBase, MeshTags, int]:
    """
    This is a convenience function for several classes/functions of other packages.
    It is not intended to be used in other context.

    Example:
        .. highlight:: python
        .. code-block:: python

          function_space, tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)

        where functionspace_tags_marker can be:

        .. highlight:: python
        .. code-block:: python

            functionspace_tags_marker = (function_space, facet_tags, marker)
            functionspace_tags_marker = (function_space, cell_tags, marker)
            functionspace_tags_marker = function_space #means tags=None and marker=None
    """
    if isinstance(functionspace_tags_marker, FunctionSpaceBase):
        return functionspace_tags_marker, None, None
    else:
        return functionspace_tags_marker
