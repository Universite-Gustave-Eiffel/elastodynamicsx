# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker
from elastodynamicsx.pde import PDE


def material(functionspace_tags_marker, type_, *args, **kwargs):
    """
    Builder method that instanciates the desired material type

    Args:
        functionspace_tags_marker: Available possibilities are
            (function_space, cell_tags, marker)  # Meaning application of the
                material in the cells whose tag correspond to marker
            function_space #In this case cell_tags=None and marker=None, meaning
                application of the material in the entire domain
        type_: Available options are:
                 'scalar'
                 'isotropic'
        args:   Passed to the required material
        kwargs: Passed to the required material

    Returns:
        An instance of the desired material

    Examples of use:
        aluminum = material((function_space, cell_tags, 1), 'isotropic',
                             rho=2.8, lambda_=58, mu=26) #restricted to subdomain number 1
        aluminum = material( function_space, 'isotropic',
                             rho=2.8, lambda_=58, mu=26) #entire domain

        mat = material( function_space, 'isotropic'  , rho, C12, C44)
        mat = material( function_space, 'cubic'      , rho, C11, C12, C44)
        mat = material( function_space, 'hexagonal'  , rho, C11, C13, C33, C44, C66)
        mat = material( function_space, 'trigonal'   , rho, C11, C12, C13, C14, C25, C33, C44)
        mat = material( function_space, 'tetragonal' , rho, C11, C12, C13, C16, C33, C44, C66)
        mat = material( function_space, 'orthotropic', rho, C11, C12, C13, C22, C23, C33, C44, C55, C66)
        mat = material( function_space, 'monoclinic' , rho, C11, C12, C13, C15, C22, C23, C25, C33, C35, C44, C46, C55, C66)
        mat = material( function_space, 'triclinic'  , rho, C11, C12, C13, C14, C15, C16, C22, C23, C24, C25, C26, C33, C34, C35, C36, C44, C45, C46, C55, C56, C66)
    """
    for Mat in all_materials:
        if type_.lower() in Mat.labels:
            return Mat(functionspace_tags_marker, *args, **kwargs)
    #
    raise TypeError('unknown material type: '+type_)



class Material():
    """
    Base class for a representation of a PDE of the kind:

        M*a + C*v + K(u) = 0

    i.e. the lhs of a pde such as defined in the PDE class. An instance represents
    a single (possibly arbitrarily space-dependent) material.
    """


    ### --------------------------
    ### --------- static ---------
    ### --------------------------

    labels = ['supercharge me']

    def build(*args, **kwargs):
        """
        Same as material(*args, **kwargs).
        """
        return material(*args, **kwargs)


    ### --------------------------
    ### ------- non-static -------
    ### --------------------------

    def __init__(self, functionspace_tags_marker, rho, is_linear, **kwargs):
        """
        Args:
            functionspace_tags_marker: Available possibilities are
                (function_space, cell_tags, marker) #Meaning application of the
                    material in the cells whose tag correspond to marker
                function_space #In this case cell_tags=None and marker=None, meaning
                    application of the material in the entire domain
            rho:   Density
            is_linear: True for linear, False for hyperelastic
            kwargs:
                metadata: (default=None) The metadata used by the ufl measures (dx, dS)
                    If set to None, uses the PDE.default_metadata
        """
        self._rho = rho
        self._is_linear = is_linear

        function_space, cell_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)

        domain = function_space.mesh
        md     = kwargs.get('metadata', PDE.default_metadata)
        self._dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags, metadata=md)(marker) #also valid if cell_tags or marker are None
        self._dS = ufl.Measure("dS", domain=domain, subdomain_data=cell_tags, metadata=md)(marker)
        self._function_space = function_space

        e = function_space.element.basix_element
        if e.discontinuous == True: #True for discontinuous Galerkin
            print('Material: Using discontinuous elements -> DG formulation')
            self._k = self.k_DG
        else:
            # print('Material: Using continuous elements -> CG formulation')
            self._k = self.k_CG


    @property
    def is_linear(self) -> bool:
        return self._is_linear

    @property
    def m(self) -> 'function':
        """(bilinear) mass form function"""
        return lambda u,v: self._rho* ufl.inner(u, v) * self._dx

    @property
    def c(self) -> 'function':
        """(bilinear) damping form function"""
        return None

    @property
    def k(self) -> 'function':
        """Stiffness form function"""
        return self._k

    @property
    def k_CG(self) -> 'function':
        """Stiffness form function for a Continuous Galerkin formulation"""
        print("supercharge me")

    @property
    def k_DG(self) -> 'function':
        """Stiffness form function for a Disontinuous Galerkin formulation"""
        print("supercharge me")

    @property
    def DG_numerical_flux(self) -> 'function':
        """Numerical flux for a Disontinuous Galerkin formulation"""
        print("supercharge me")

    @property
    def rho(self):
        """Density"""
        return self._rho


# -----------------------------------------------------
# Import subclasses -- must be done at the end to avoid loop imports
# -----------------------------------------------------
from .elasticmaterial import ScalarLinearMaterial, IsotropicMaterial
from .anisotropicmaterials import CubicMaterial, HexagonalMaterial, TrigonalMaterial, TetragonalMaterial, OrthotropicMaterial, MonoclinicMaterial, TriclinicMaterial
from .hyperelasticmaterial import DummyIsotropicMaterial, Murnaghan, StVenantKirchhoff, MooneyRivlinIncompressible, MooneyRivlinCompressible

all_linear_materials    = [ScalarLinearMaterial, IsotropicMaterial, CubicMaterial, HexagonalMaterial,
                          TrigonalMaterial, TetragonalMaterial, OrthotropicMaterial, MonoclinicMaterial, TriclinicMaterial]
all_nonlinear_materials = [DummyIsotropicMaterial, Murnaghan, StVenantKirchhoff, MooneyRivlinIncompressible, MooneyRivlinCompressible]
all_materials           = all_linear_materials + all_nonlinear_materials
