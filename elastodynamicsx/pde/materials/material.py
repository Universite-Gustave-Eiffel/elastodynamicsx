# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

import ufl  # type: ignore

from elastodynamicsx.utils import _get_functionspace_tags_marker
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

        function_space, cell_tags, marker = _get_functionspace_tags_marker(functionspace_tags_marker)

        domain = function_space.mesh
        md = kwargs.get('metadata', PDECONFIG.default_metadata)
        # also valid if cell_tags or marker are None
        self._dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags, metadata=md)(marker)
        self._dS = ufl.Measure("dS", domain=domain, subdomain_data=cell_tags, metadata=md)(marker)
        self._function_space = function_space

        e = function_space.element.basix_element
        if e.discontinuous is True:  # True for discontinuous Galerkin
            print('Material: Using discontinuous elements -> DG formulation')
            self._K_fn = self.K_fn_DG
        else:
            # print('Material: Using continuous elements -> CG formulation')
            self._K_fn = self.K_fn_CG

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    def M_fn(self, u, v):
        """(bilinear) mass form function"""
        return self._rho * ufl.inner(u, v) * self._dx

    def C_fn(self, u, v):
        """(bilinear) damping form function"""
        return None

    def K_fn(self, u, v):
        """(bilinear) Stiffness form function"""
        return self._K_fn(u, v)

    def K_fn_CG(self, u, v):
        """Stiffness form function for a Continuous Galerkin formulation"""
        print("supercharge me")
        raise NotImplementedError

    def K0_fn_CG(self, u, v):
        """K0 stiffness form function for a Continuous Galerkin formulation (waveguides)"""
        print("supercharge me")
        raise NotImplementedError

    def K1_fn_CG(self, u, v):
        """K1 stiffness form function for a Continuous Galerkin formulation (waveguides)"""
        print("supercharge me")
        raise NotImplementedError

    def K2_fn_CG(self, u, v):
        """K2 stiffness form function for a Continuous Galerkin formulation (waveguides)"""
        print("supercharge me")
        raise NotImplementedError

    def K_fn_DG(self, u, v):
        """Stiffness form function for a Disontinuous Galerkin formulation"""
        print("supercharge me")
        raise NotImplementedError

    def DG_numerical_flux(self, u, v):
        """Numerical flux for a Disontinuous Galerkin formulation"""
        print("supercharge me")
        raise NotImplementedError

    @property
    def rho(self):
        """Density"""
        return self._rho
