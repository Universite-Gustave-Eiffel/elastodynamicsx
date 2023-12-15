# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
.. role:: python(code)
  :language: python

The *materials* module contains classes for a representation of a PDE of the kind:

    M*a + C*v + K(u) = 0

i.e. the lhs of a PDE such as defined in the PDE class. An instance represents
a single (possibly arbitrarily space-dependent) material.

| The preferred way to call a material law is using the :python:`material` function:

.. highlight:: python
.. code-block:: python

  from elastodynamicsx.pde import material
  mat = material( (function_space, cell_tags, marker), type_, *args, **kwargs)
"""

# import base classes
from .material import Material
from .elasticmaterial import ElasticMaterial
from .hyperelasticmaterials import HyperelasticMaterial

# import builder
from .dampings import damping

# import submodules
from . import isotropicmaterials
from . import anisotropicmaterials
from . import hyperelasticmaterials
from . import dampings

# import material classes (without including them in __all__)
from .isotropicmaterials import ScalarLinearMaterial, IsotropicMaterial
from .anisotropicmaterials import CubicMaterial, HexagonalMaterial, TrigonalMaterial, TetragonalMaterial, \
    OrthotropicMaterial, MonoclinicMaterial, TriclinicMaterial
from .hyperelasticmaterials import Murnaghan, DummyIsotropicMaterial, StVenantKirchhoff, \
    MooneyRivlinIncompressible, MooneyRivlinCompressible


all_linear_materials = [ScalarLinearMaterial, IsotropicMaterial, CubicMaterial, HexagonalMaterial,
                        TrigonalMaterial, TetragonalMaterial, OrthotropicMaterial, MonoclinicMaterial,
                        TriclinicMaterial]
all_nonlinear_materials = [DummyIsotropicMaterial, Murnaghan, StVenantKirchhoff, MooneyRivlinIncompressible,
                           MooneyRivlinCompressible]
all_materials = all_linear_materials + all_nonlinear_materials


def material(functionspace_tags_marker, type_, *args, **kwargs) -> Material:
    """
    .. role:: python(code)
       :language: python

    Builder method that instanciates the desired material type

    Args:
        functionspace_tags_marker: Available possibilities are:

            - :python:`(function_space, cell_tags, marker)`: meaning application
                of the material in the cells whose tag correspond to marker
            - :python:`function_space`: in this case :python:`cell_tags=None`
                and :python:`marker=None`, meaning application of the material in the entire domain

        type_: Available options are:

            - *linear (scalar):* 'scalar'
            - *linear:* 'isotropic', 'cubic', 'hexagonal', 'trigonal', 'tetragonal',
                'orthotropic', 'monoclinic', 'triclinic'
            - *nonlinear, hyperelastic:* 'murnaghan', 'saintvenant-kirchhoff', 'mooney-rivlin-incomp'

        *args: Passed to the required material

    Keyword Args:
        **kwargs: Passed to the required material

    Returns:
        An instance of the desired material

    Example:
        .. highlight:: python
        .. code-block:: python

          # ###
          # Constant / variable material parameter
          # ###
          rho = 2.8
          rho = fem.Constant(function_space.mesh, default_scalar_type(2.8))
          rho = fem.Function(scalar_function_space); rho.interpolate(lambda x: 2.8 * np.ones(len(x)))

          # ###
          # Subdomain(s) or entire domain?
          # ###

          # restricted to subdomain number 1
          aluminum = material((function_space, cell_tags, 1), 'isotropic',
                               rho=2.8, lambda_=58, mu=26)

          # restricted to subdomains number 1, 2, 5
          aluminum = material((function_space, cell_tags, (1,2,5)), 'isotropic',
                               rho=2.8, lambda_=58, mu=26)

          # entire domain
          aluminum = material( function_space, 'isotropic',
                               rho=2.8, lambda_=58, mu=26)

          # ###
          # Available laws
          # ###

          # Linear elasticity
          mat = material( function_space, 'isotropic'  , rho, C12, C44)
          mat = material( function_space, 'cubic'      , rho, C11, C12, C44)
          mat = material( function_space, 'hexagonal'  , rho, C11, C13, C33, C44, C66)
          mat = material( function_space, 'trigonal'   , rho, C11, C12, C13, C14, C25,
                                                              C33, C44)
          mat = material( function_space, 'tetragonal' , rho, C11, C12, C13, C16, C33, C44, C66)
          mat = material( function_space, 'orthotropic', rho, C11, C12, C13, C22, C23, C33, C44,
                                                              C55, C66)
          mat = material( function_space, 'monoclinic' , rho, C11, C12, C13, C15, C22, C23, C25,
                                                              C33, C35, C44, C46, C55, C66)
          mat = material( function_space, 'triclinic'  , rho, C11, C12, C13, C14, C15, C16, C22,
                                                              C23, C24, C25, C26, C33, C34, C35,
                                                              C36, C44, C45, C46, C55, C56, C66)

          # Hyperelasticity
          mat = material( function_space, 'saintvenant-kirchhoff', rho, C12, C44)
          mat = material( function_space, 'murnaghan', rho, C12, C44, l, m, n)
          mat = material( function_space, 'mooney-rivlin-incomp', rho, C1, C2)
    """
    for Mat in all_materials:
        if type_.lower() in Mat.labels:
            return Mat(functionspace_tags_marker, *args, **kwargs)
    #
    raise TypeError('unknown material type: ' + type_)


__all__ = ["material", "damping",
           "Material", "ElasticMaterial", "HyperelasticMaterial",
           "isotropicmaterials", "anisotropicmaterials", "hyperelasticmaterials", "dampings"]
