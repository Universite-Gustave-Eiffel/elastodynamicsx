# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

import ufl  # type: ignore

from .common import PDECONFIG
from elastodynamicsx.utils import get_functionspace_tags_marker

# just for types
import numpy as np
from dolfinx.fem.function import Constant, Function


class BodyForce:
    """
    Representation of the rhs term (the 'b' term) of a pde such as defined
    in the PDE class. An instance represents a single source.

    Args:
        functionspace_tags_marker: The function space, cell tags and marker(s) where the linear form is defined
        value: The value of the body force as a ufl-compatible object

    Keyword Args:
        metadata: dict, compiler-specific parameters, passed to ufl.Measure(..., metadata=metadata)

    Example:
        .. highlight:: python
        .. code-block:: python

          # V is a vector function space, dim=2

          # Gaussian source distribution
          X0_src = np.array([length/2, height/2, 0])  # center
          R0_src = 0.1  # radius
          F0_src = default_scalar_type([1,0])  # amplitude of the source

          def src_x(x):  # source(x): Gaussian
              nrm = 1/(2*np.pi*R0_src**2)  # normalize to int[src_x(x) dx]=1
              r = np.linalg.norm(x-X0_src[:,np.newaxis], axis=0)
              return nrm * np.exp(-1/2*(r/R0_src)**2, dtype=default_scalar_type)

          value = fem.Function(V)
          value.interpolate(lambda x: F0_src[:,np.newaxis] * src_x(x)[np.newaxis,:])

          bf = BodyForce(V, value)  # defined in the entire domain
          bf = BodyForce((V, cell_tags, (1, 4)), value)  # definition restricted to tags 1 and 4
    """

    def __init__(self, functionspace_tags_marker, value: typing.Union[Function, Constant, np.ndarray], **kwargs):
        self._value = value
        function_space, cell_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)
        md = kwargs.get('metadata', PDECONFIG.default_metadata)
        # md = kwargs.get('metadata', None)
        self._dx = ufl.Measure("dx", domain=function_space.mesh, subdomain_data=cell_tags, metadata=md)(marker)

    @property
    def L(self) -> typing.Callable:
        """The linear form function"""
        return lambda v: ufl.inner(self._value, v) * self._dx

    @property
    def value(self) -> typing.Union[Function, Constant, np.ndarray]:
        return self._value
