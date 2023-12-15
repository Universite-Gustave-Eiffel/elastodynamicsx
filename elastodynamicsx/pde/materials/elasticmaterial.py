# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

from dolfinx import fem, default_scalar_type
import ufl
import numpy as np

from .material import Material
from .kinematics import get_epsilon_function, get_epsilonVoigt_function, get_L_operators
from .dampings import NoDamping, RayleighDamping
from elastodynamicsx.utils import get_functionspace_tags_marker


class ElasticMaterial(Material):
    """
    Base class for linear elastic materials, supporting full anisotropy

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C_21: list of the 21 independent elastic constants

    Keyword Args:
        damping: (default=NoDamping()) An instance of a subclass of Damping
    """

    # ## ### ### ## #
    # ## static  ## #
    # ## ### ### ## #

    eq_IJ: np.ndarray = np.zeros((6, 6), dtype='int')

    # fill eq_IJ
    _cpt = 0
    for _i in range(6):
        for _j in range(6 - _i):
            eq_IJ[_i, _i + _j] = _cpt
            _cpt += 1
    for _i in range(6):
        for _j in range(_i):
            eq_IJ[_i, _j] = eq_IJ[_j, _i]

    ijkm: np.ndarray = np.zeros((21, 4), dtype='int')
    ijkm[0, :] = 0, 0, 0, 0  # C11 <-> 0
    ijkm[1, :] = 0, 0, 1, 1  # C12 <-> 1
    ijkm[2, :] = 0, 0, 2, 2  # C13 <-> 2
    ijkm[3, :] = 0, 0, 1, 2  # C14 <-> 3
    ijkm[4, :] = 0, 0, 0, 2  # C15 <-> 4
    ijkm[5, :] = 0, 0, 0, 1  # C16 <-> 5
    ijkm[6, :] = 1, 1, 1, 1  # C22 <-> 6
    ijkm[7, :] = 1, 1, 2, 2  # C23 <-> 7
    ijkm[8, :] = 1, 1, 1, 2  # C24 <-> 8
    ijkm[9, :] = 1, 1, 0, 2  # C25 <-> 9
    ijkm[10, :] = 1, 1, 0, 1  # C26 <-> 10
    ijkm[11, :] = 2, 2, 2, 2  # C33 <-> 11
    ijkm[12, :] = 2, 2, 1, 2  # C34 <-> 12
    ijkm[13, :] = 2, 2, 0, 2  # C35 <-> 13
    ijkm[14, :] = 2, 2, 0, 1  # C36 <-> 14
    ijkm[15, :] = 1, 2, 1, 2  # C44 <-> 15
    ijkm[16, :] = 1, 2, 0, 2  # C45 <-> 16
    ijkm[17, :] = 1, 2, 0, 1  # C46 <-> 17
    ijkm[18, :] = 0, 2, 0, 2  # C55 <-> 18
    ijkm[19, :] = 0, 2, 0, 1  # C56 <-> 19
    ijkm[20, :] = 0, 1, 0, 1  # C66 <-> 20

    def Cij(C_21: typing.List, i: int, j: int):
        """
        Returns the :math:`C_{ij}` coefficient from the C_21 list

        i, j = 0..5
        """
        return C_21[ElasticMaterial.eq_IJ[i, j]]

    def Cijkm(C_21: typing.List, i: int, j: int, k: int, m: int):
        """
        Returns the :math:`C_{ijkm}` coefficient from the C_21 list

        i, j, k, m = 0..2
        """
        ij = i if i == j else 6 - i - j
        km = k if k == m else 6 - k - m
        return ElasticMaterial.Cij(C_21, ij, km)

    # ## ### ### ### ## #
    # ## non-static  ## #
    # ## ### ### ### ## #

    def __init__(self, functionspace_tags_marker, rho, C_21: typing.List, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        dim = function_space.mesh.geometry.dim  # space dimension
        nbcomps = function_space.num_sub_spaces  # number of components (0 for scalar)

        # Cij coefficients, in (6x6) or (3x3x3x3) representations
        self._C_21 = C_21
        self._Cij_6x6 = ufl.as_matrix([[ElasticMaterial.Cij(C_21, i, j) for j in range(6)] for i in range(6)])
        self._Cijkm_3x3x3x3 = ufl.as_tensor([[[[ElasticMaterial.Cijkm(C_21, i, j, k, m) for m in range(3)]
                                             for k in range(3)] for j in range(3)] for i in range(3)])

        # TODO: rotate with Euler angles

        # Cij coefficients, in (3x3) or (2x2x2x2) representations if nbcomps == 2
        if nbcomps == 2:  # Cij -> 3x3 matrix; Cijkm -> 2x2x2x2 tensor
            self._Cij = ufl.as_matrix([[self._Cij_6x6[i, j] for j in (0, 1, 3)] for i in (0, 1, 3)])
            self._Cijkm = ufl.as_tensor([[[[self._Cijkm_3x3x3x3[i, j, k, m] for m in range(2)] for k in range(2)]
                                          for j in range(2)] for i in range(2)])
        else:
            self._Cij = self._Cij_6x6
            self._Cijkm = self._Cijkm_3x3x3x3

        # ###
        # Kinematics
        # Strain operator, matrix representation
        self._epsilon = get_epsilon_function(dim, nbcomps)

        # Strain operator, Voigt representation
        self._epsilonVoigt = get_epsilonVoigt_function(dim, nbcomps)

        # L operators (waveguides): Lxy and Ls for 2D cross sections, Lx and Ls for 1D cross sections
        self._L_crosssection, self._L_onaxis = get_L_operators(dim, nbcomps)
        # ###

        self._DGvariant = kwargs.pop('DGvariant', 'SIPG')
        self._damping = kwargs.pop('damping', NoDamping())
        if isinstance(self._damping, RayleighDamping) and (self._damping.host_material is None):
            self._damping.link_material(self)

        super().__init__(functionspace_tags_marker, rho, is_linear=True, **kwargs)

    # ## ### ### ### ### ## #
    # ## Stress, strain  ## #
    # ## ### ### ### ### ## #

    @property
    def epsilon(self):
        """Strain function (matrix representation) :math:`\\boldsymbol{\epsilon}(\mathbf{u})`"""
        return self._epsilon

    @property
    def epsilonVoigt(self):
        """Strain function (Voigt representation) :math:`\\boldsymbol{\epsilon}(\mathbf{u})`"""
        return self._epsilonVoigt

    def sigma(self, u):
        """Stress function (matrix representation) :math:`\\boldsymbol{\sigma}(\mathbf{u})`"""
        return self._Cijkm * self._epsilon(u)

    def sigmaVoigt(self, u):
        """Stress function (Voigt representation) :math:`\\boldsymbol{\sigma}(\mathbf{u})`"""
        return self._Cij * self._epsilonVoigt(u)

    def sigma_n(self, u, n):
        """Stress in the 'n' direction :math:`\\boldsymbol{\sigma}(\mathbf{u}) \cdot \mathbf{n}`"""
        sig = self.sigma(u)
        dim = len(n)
        col = sig.ufl_shape[1]

        if col > dim:
            n = ufl.as_vector([n[i] for i in range(dim)] + [0 for i in range(col - dim)])

        return ufl.dot(sig, n)

    def diamond(self, v1, v2):
        """
        Returns the diamond product of vectors v1 and v2.

        Returns:
            :math:`\mathbf{d}`, such as :math:`d_{jk} = C_{ijkm}  v_{1,i}  v_{2,m}`
        """
        v1 = ufl.as_vector(v1)
        v2 = ufl.as_vector(v2)
        i, j, k, m = ufl.indices(4)
        return ufl.as_matrix(self._Cijkm[i, j, k, m] * v1[i] * v2[m], (j, k))

    # ## ### ### ### ### ### ### ### ## #
    # ## Damping and stiffness forms ## #
    # ## ### ### ### ### ### ### ### ## #

    @property
    def c(self) -> typing.Callable:
        """Damping form function"""
        return self._damping.c

    @property
    def k_CG(self) -> typing.Callable:
        """Stiffness form function for a Continuous Galerkin formulation"""
        # return lambda u, v: ufl.inner(self.sigma(u), self.epsilon(v)) * self._dx
        return lambda u, v: ufl.inner(self.sigmaVoigt(u), self._epsilonVoigt(v)) * self._dx

    @property
    def k1_CG(self) -> typing.Callable:
        return lambda u, v: ufl.inner(self._Cij * self._L_crosssection(u), self._L_crosssection(v)) * self._dx

    # @property
    # def k2_CG(self) -> typing.Callable:  # This is K2 - K2.T
    #     return lambda u, v: ( ufl.inner(self._Cij * self._L_onaxis(u), self._L_crosssection(v)) \
    #         - ufl.inner(self._L_crosssection(u), self._Cij * self._L_onaxis(v)) )* self._dx

    @property
    def k2_CG(self) -> typing.Callable:
        return lambda u, v: ufl.inner(self._Cij * self._L_onaxis(u), self._L_crosssection(v)) * self._dx

    @property
    def k3_CG(self) -> typing.Callable:
        return lambda u, v: ufl.inner(self._Cij * self._L_onaxis(u), self._L_onaxis(v)) * self._dx

    # ## ### ### ### ### ### ### ### ### ### ## #
    # ## Discontinuous Galerkin formulation  ## #
    # ## ### ### ### ### ### ### ### ### ### ## #

    @property
    def k_DG(self) -> typing.Callable:
        """Stiffness form function for a Discontinuous Galerkin formulation"""
        return lambda u, v: self.k_CG(u, v) + self.DG_numerical_flux(u, v)

    def select_DG_numerical_flux(self, variant: str = 'SIPG') -> typing.Callable:
        if variant.upper() == 'SIPG':
            return self.DG_numerical_flux_SIPG
        elif variant.upper() == 'NIPG':
            return self.DG_numerical_flux_NIPG
        elif variant.upper() == 'IIPG':
            return self.DG_numerical_flux_IIPG
        else:
            raise TypeError('Unknown DG variant ' + variant)

    @property
    def DG_numerical_flux(self) -> typing.Callable:
        """Numerical flux for a Disontinuous Galerkin formulation"""
        return self.DG_numerical_flux_SIPG

    def DG_SIPG_regularization_parameter(self) -> fem.Constant:
        """Regularization parameter for the Symmetric Interior Penalty Galerkin methods (SIPG)"""
        degree = self._function_space.ufl_element().degree()
        # +1 otherwise blows with elements of degree 1
        gamma = fem.Constant(self._function_space.mesh, default_scalar_type(degree * (degree + 1) + 1))
        # gamma = fem.Constant(self._function_space.mesh, default_scalar_type(160))
        P_mod = self.P_modulus
        R_ = gamma * P_mod
        return R_

    @property
    def DG_numerical_flux_SIPG(self) -> typing.Callable:
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V)  # works!
        h_avg = (h('+') + h('-')) / 2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n

        def k_int_facets(u, v):
            return -inner(avg(sig_n(u, n)), jump(v)) * dS \
                - inner(jump(u), avg(sig_n(v, n))) * dS \
                + R_ / h_avg * inner(jump(u), jump(v)) * dS

        return k_int_facets

    @property
    def DG_numerical_flux_NIPG(self) -> typing.Callable:
        """WARNING, instable for elasticity"""
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V)  # works!
        h_avg = (h('+') + h('-')) / 2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n

        def k_int_facets(u, v):
            return -inner(avg(sig_n(u, n)), jump(v)) * dS \
                + inner(jump(u), avg(sig_n(v, n))) * dS \
                + R_ / h_avg * inner(jump(u), jump(v)) * dS

        return k_int_facets

    @property
    def DG_numerical_flux_IIPG(self) -> typing.Callable:
        """WARNING, instable for elasticity"""
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V)  # works!
        h_avg = (h('+') + h('-')) / 2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n

        def k_int_facets(u, v):
            return -inner(avg(sig_n(u, n)), jump(v)) * dS \
                + R_ / h_avg * inner(jump(u), jump(v)) * dS

        return k_int_facets

    # ## ### ### ### ### ### ## #
    # ## material constants  ## #
    # ## ### ### ### ### ### ## #

    @property
    def P_modulus(self):
        """
        P-wave modulus

        Returns:

            - :math:`\\rho c_{max}^2` where :math:`c_{max}` is the highest wave velocity
            - :math:`\lambda + 2 \mu` for isotropic materials
            - :math:`\mu` for scalar materials
        """
        raise NotImplementedError('ElasticMaterial::P_modulus -> supercharge me')

    def Cij_xyz_frame(self, i, j):
        """Cij stiffness constant in the (xyz) coordinate frame"""
        return self._Cij_6x6[i, j]

    def Cijkm_xyz_frame(self, i, j, k, m):
        """Cijkm stiffness constant in the (xyz) coordinate frame"""
        return self._Cijkm_3x3x3x3[i, j, k, m]

    def Cij_mat_frame(self, i, j):
        """Cij stiffness constant in the material coordinate frame"""
        return ElasticMaterial.Cij(self._C_21, i, j)

    def Cijkm_mat_frame(self, i, j, k, m):
        """Cijkm stiffness constant in the material coordinate frame"""
        return ElasticMaterial.Cijkm(self._C_21, i, j, k, m)

    @property
    def C11(self):
        """C11 stiffness constant in the material coordinate frame"""
        return self._C_21[0]

    @property
    def C12(self):
        """C12 stiffness constant in the material coordinate frame"""
        return self._C_21[1]

    @property
    def C13(self):
        """C13 stiffness constant in the material coordinate frame"""
        return self._C_21[2]

    @property
    def C14(self):
        """C14 stiffness constant in the material coordinate frame"""
        return self._C_21[3]

    @property
    def C15(self):
        """C15 stiffness constant in the material coordinate frame"""
        return self._C_21[4]

    @property
    def C16(self):
        """C16 stiffness constant in the material coordinate frame"""
        return self._C_21[5]

    @property
    def C22(self):
        """C22 stiffness constant in the material coordinate frame"""
        return self._C_21[6]

    @property
    def C23(self):
        """C23 stiffness constant in the material coordinate frame"""
        return self._C_21[7]

    @property
    def C24(self):
        """C24 stiffness constant in the material coordinate frame"""
        return self._C_21[8]

    @property
    def C25(self):
        """C25 stiffness constant in the material coordinate frame"""
        return self._C_21[9]

    @property
    def C26(self):
        """C26 stiffness constant in the material coordinate frame"""
        return self._C_21[10]

    @property
    def C33(self):
        """C33 stiffness constant in the material coordinate frame"""
        return self._C_21[11]

    @property
    def C34(self):
        """C34 stiffness constant in the material coordinate frame"""
        return self._C_21[12]

    @property
    def C35(self):
        """C35 stiffness constant in the material coordinate frame"""
        return self._C_21[13]

    @property
    def C36(self):
        """C36 stiffness constant in the material coordinate frame"""
        return self._C_21[14]

    @property
    def C44(self):
        """C44 stiffness constant in the material coordinate frame"""
        return self._C_21[15]

    @property
    def C45(self):
        """C45 stiffness constant in the material coordinate frame"""
        return self._C_21[16]

    @property
    def C46(self):
        """C46 stiffness constant in the material coordinate frame"""
        return self._C_21[17]

    @property
    def C55(self):
        """C55 stiffness constant in the material coordinate frame"""
        return self._C_21[18]

    @property
    def C56(self):
        """C56 stiffness constant in the material coordinate frame"""
        return self._C_21[19]

    @property
    def C66(self):
        """C66 stiffness constant in the material coordinate frame"""
        return self._C_21[20]
