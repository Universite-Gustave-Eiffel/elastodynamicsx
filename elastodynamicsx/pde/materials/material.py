# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker
from elastodynamicsx.pde import PDECONFIG


class Material:
    """
    .. role:: python(code)
       :language: python

    Base class for a representation of a material law.

    Args:
        functionspace_tags_marker: Available possibilities are:

            - :python:`(function_space, cell_tags, marker)`: meaning application of
                the material in the cells whose tag correspond to marker
            - :python:`function_space`: in this case :python:`cell_tags=None`
                and :python:`marker=None`, meaning application of the material in the entire domain

        rho: Density
        is_linear: True for linear, False for hyperelastic

    Keyword Args:
        metadata: (default=None) The metadata used by the ufl measures (dx, dS).
            If set to :python:`None`, uses the :python:`PDECONFIG.default_metadata`
    """

    # ## --------------------------
    # ## --------- static ---------
    # ## --------------------------

    labels: typing.List[str]

    # ## --------------------------
    # ## ------- non-static -------
    # ## --------------------------

    def __init__(self, functionspace_tags_marker, rho, is_linear, **kwargs):
        self._rho = rho
        self._is_linear = is_linear

        function_space, cell_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)

        domain = function_space.mesh
        md = kwargs.get('metadata', PDECONFIG.default_metadata)
        # also valid if cell_tags or marker are None
        self._dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags, metadata=md)(marker)
        self._dS = ufl.Measure("dS", domain=domain, subdomain_data=cell_tags, metadata=md)(marker)
        self._function_space = function_space

        e = function_space.element.basix_element
        if e.discontinuous is True:  # True for discontinuous Galerkin
            print('Material: Using discontinuous elements -> DG formulation')
            self._k = self.k_DG
        else:
            # print('Material: Using continuous elements -> CG formulation')
            self._k = self.k_CG

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def m(self) -> typing.Callable:
        """(bilinear) mass form function"""
        return lambda u, v: self._rho * ufl.inner(u, v) * self._dx

    @property
    def c(self) -> typing.Callable:
        """(bilinear) damping form function"""
        return None

    @property
    def k(self) -> typing.Callable:
        """(bilinear) Stiffness form function"""
        return self._k

    @property
    def k_CG(self) -> typing.Callable:
        """Stiffness form function for a Continuous Galerkin formulation"""
        print("supercharge me")
        raise NotImplementedError

    @property
    def k_DG(self) -> typing.Callable:
        """Stiffness form function for a Disontinuous Galerkin formulation"""
        print("supercharge me")
        raise NotImplementedError

    @property
    def DG_numerical_flux(self) -> typing.Callable:
        """Numerical flux for a Disontinuous Galerkin formulation"""
        print("supercharge me")
        raise NotImplementedError

    @property
    def rho(self):
        """Density"""
        return self._rho
