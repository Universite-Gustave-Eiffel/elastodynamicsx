# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

from dolfinx import fem, default_scalar_type
import ufl

from .elasticmaterial import ElasticMaterial
from elastodynamicsx.utils import get_functionspace_tags_marker


class ScalarLinearMaterial(ElasticMaterial):
    """
    A scalar linear material (e.g. 2D-SH or fluid)

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        mu : Shear modulus (or rho*c**2 for a fluid, with c the sound velocity)

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['scalar', '2d-sh', 'fluid']

    def __init__(self, functionspace_tags_marker, rho, mu, **kwargs):
        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements == 0, 'ScalarLinearMaterial requires a scalar function space'

        C11 = C22 = C33 = mu
        C12 = C13 = C23 = mu
        C_21 = [0] * 21
        C_21[0], C_21[6], C_21[11] = C11, C22, C33
        C_21[1], C_21[2], C_21[7] = C12, C13, C23
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)

    @property
    def mu(self):
        """The shear modulus"""
        return self._C_21[0]

    @property
    def Z(self):
        """The mechanical impedance :math:`\\rho c`"""
        return ufl.sqrt(self.rho * self.mu)

    @property
    def P_modulus(self):
        """P-wave modulus, mu stiffness constant"""
        return self.mu

    def sigma(self, u):
        """Stress function (matrix representation): sigma(u)"""
        return self.mu * self._epsilon(u)

    def sigmaVoigt(self, u):
        """Stress function (Voigt representation): sigma(u)"""
        return self.mu * self._epsilonVoigt(u)

    @property
    def k1_CG(self) -> typing.Callable:
        return lambda u, v: ufl.inner(self.mu * self._L_crosssection(u), self._L_crosssection(v)) * self._dx

    @property
    def k2_CG(self) -> typing.Callable:
        return lambda u, v: fem.Constant(u.ufl_function_space().mesh, default_scalar_type(0)) \
            * ufl.inner(u, v) * self._dx

    @property
    def k3_CG(self) -> typing.Callable:
        return lambda u, v: ufl.inner(self.mu * self._L_onaxis(u), self._L_onaxis(v)) * self._dx

    @property
    def DG_numerical_flux_SIPG(self) -> typing.Callable:
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V)  # works!
        h_avg = (h('+') + h('-')) / 2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma

        def k_int_facets(u, v):
            return -inner(avg(sigma(u)), jump(v, n)) * dS \
                - inner(jump(u, n), avg(sigma(v))) * dS \
                + R_ / h_avg * inner(jump(u), jump(v)) * dS

        return k_int_facets

    @property
    def DG_numerical_flux_NIPG(self) -> typing.Callable:
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V)  # works!
        h_avg = (h('+') + h('-')) / 2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma

        def k_int_facets(u, v):
            return -inner(avg(sigma(u)), jump(v, n)) * dS \
                + inner(jump(u, n), avg(sigma(v))) * dS \
                + R_ / h_avg * inner(jump(u), jump(v)) * dS

        return k_int_facets

    @property
    def DG_numerical_flux_IIPG(self) -> typing.Callable:
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V)  # works!
        h_avg = (h('+') + h('-')) / 2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma

        def k_int_facets(u, v):
            return -inner(avg(sigma(u)), jump(v, n)) * dS + R_ / h_avg * inner(jump(u), jump(v)) * dS

        return k_int_facets


class IsotropicMaterial(ElasticMaterial):
    """
    An isotropic linear elastic material

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        lambda_: Lame's first parameter
        mu: Lame's second parameter (shear modulus)

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['isotropic']

    def __init__(self, functionspace_tags_marker, rho, lambda_, mu, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'IsotropicMaterial requires a vector function space'

        C11 = C22 = C33 = lambda_ + 2 * mu
        C12 = C13 = C23 = lambda_
        C44 = C55 = C66 = mu
        C_21 = [0] * 21
        C_21[0], C_21[6], C_21[11] = C11, C22, C33
        C_21[1], C_21[2], C_21[7] = C12, C13, C23
        C_21[15], C_21[18], C_21[20] = C44, C55, C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)

    @property
    def lambda_(self):
        """Lame's first parameter"""
        return self._C_21[1]

    @property
    def mu(self):
        """Lame's second parameter (shear modulus)"""
        return self._C_21[15]

    @property
    def P_modulus(self):
        """P-wave modulus, C11 stiffness constant"""
        return self._C_21[0]

    @property
    def Z_N(self):
        """P-wave mechanical impedance :math:`\\rho c_L`"""
        return ufl.sqrt(self.rho * self.P_modulus)

    @property
    def Z_T(self):
        """S-wave mechanical impedance :math:`\\rho c_S`"""
        return ufl.sqrt(self.rho * self.mu)

    def sigma(self, u):  # TODO: is this a speed up? otherwise: remove?
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * self.mu * self.epsilon(u)
