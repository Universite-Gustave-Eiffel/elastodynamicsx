# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from .elasticmaterial import ElasticMaterial
from elastodynamicsx.utils import get_functionspace_tags_marker


class CubicMaterial(ElasticMaterial):
    """
    A linear elastic material with cubic symmetry
    -> 3 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, C12, C44: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['cubic']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C12, C44, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'CubicMaterial requires a vector function space'

        C11 = C22 = C33 = C11
        C12 = C13 = C23 = C12
        C44 = C55 = C66 = C44

        C_21 = [0] * 21
        C_21[0], C_21[6], C_21[11] = C11, C22, C33
        C_21[1], C_21[2], C_21[7] = C12, C13, C23
        C_21[15], C_21[18], C_21[20] = C44, C55, C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)


class HexagonalMaterial(ElasticMaterial):
    """
    A linear elastic material with hexagonal symmetry
    -> 5 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, ..., C66: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['hexagonal']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C13, C33, C44, C66, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'HexagonalMaterial requires a vector function space'

        C22 = C11
        C12 = C11 - 2 * C66
        C23 = C13
        C55 = C44

        C_21 = [0] * 21
        C_21[0], C_21[6], C_21[11] = C11, C22, C33
        C_21[1], C_21[2], C_21[7] = C12, C13, C23
        C_21[15], C_21[18], C_21[20] = C44, C55, C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)


class TrigonalMaterial(ElasticMaterial):
    """
    A linear elastic material with trigonal symmetry
    -> 7 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, ..., C44: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['trigonal']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C12, C13, C14, C25, C33, C44, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'TrigonalMaterial requires a vector function space'

        C_21 = [0] * 21
        C_21[0] = C11
        C_21[1] = C12
        C_21[2] = C13
        C_21[3] = C14
        C_21[4] = -C25
        C_21[6] = C11
        C_21[7] = C13
        C_21[8] = -C14
        C_21[9] = C25
        C_21[11] = C33
        C_21[15] = C44
        C_21[17] = C25
        C_21[18] = C44
        C_21[19] = C14
        C_21[20] = 0.5 * (C11 - C12)
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)


class TetragonalMaterial(ElasticMaterial):
    """
    A linear elastic material with tetragonal symmetry
    -> 7 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, ..., C66: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['tetragonal']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C12, C13, C16, C33, C44, C66, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'TetragonalMaterial requires a vector function space'

        C_21 = [0] * 21
        C_21[0] = C11
        C_21[1] = C12
        C_21[2] = C13
        C_21[5] = C16
        C_21[6] = C11
        C_21[7] = C13
        C_21[10] = -C16
        C_21[11] = C33
        C_21[15] = C44
        C_21[18] = C44
        C_21[20] = C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)


class OrthotropicMaterial(ElasticMaterial):
    """
    A linear elastic material with orthotropic symmetry
    -> 9 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, ..., C66: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['orthotropic']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C12, C13, C22, C23, C33, C44, C55, C66, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'OrthotropicMaterial requires a vector function space'

        C_21 = [0] * 21
        C_21[0] = C11
        C_21[1] = C12
        C_21[2] = C13
        C_21[6] = C22
        C_21[7] = C23
        C_21[11] = C33
        C_21[15] = C44
        C_21[18] = C55
        C_21[20] = C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)


class MonoclinicMaterial(ElasticMaterial):
    """
    A linear elastic material with monoclinic symmetry
    -> 13 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, ..., C66: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['monoclinic']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C12, C13, C15, C22, C23,
                 C25, C33, C35, C44, C46, C55, C66, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'MonoclinicMaterial requires a vector function space'

        C_21 = [0] * 21
        C_21[0] = C11
        C_21[1] = C12
        C_21[2] = C13
        C_21[4] = C15
        C_21[6] = C22
        C_21[7] = C23
        C_21[9] = C25
        C_21[11] = C33
        C_21[13] = C35
        C_21[15] = C44
        C_21[17] = C46
        C_21[18] = C55
        C_21[20] = C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)


class TriclinicMaterial(ElasticMaterial):
    """
    A linear elastic material with triclinic symmetry
    -> 21 independent constants

    Args:
        functionspace_tags_marker: See Material
        rho: Density
        C11, ..., C66: Stiffness constant

    Keyword Args:
        **kwargs: Passed to ElasticMaterial
    """

    labels = ['triclinic']

    def __init__(self, functionspace_tags_marker, rho,
                 C11, C12, C13, C14, C15, C16, C22,
                 C23, C24, C25, C26, C33, C34, C35,
                 C36, C44, C45, C46, C55, C56, C66, **kwargs):

        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements > 0, 'TriclinicMaterial requires a vector function space'

        all21Csts = (C11, C12, C13, C14, C15, C16, C22,
                     C23, C24, C25, C26, C33, C34, C35,
                     C36, C44, C45, C46, C55, C56, C66)
        C_21 = [0] * 21

        for i, Ci in enumerate(all21Csts):
            C_21[i] = Ci

        super().__init__(functionspace_tags_marker, rho, C_21, **kwargs)
