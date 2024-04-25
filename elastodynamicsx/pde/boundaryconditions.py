# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
.. role:: python(code)
   :language: python

Module for handling boundary conditions (BCs) of various types. The proper way to
instantiate a BC is using the :python:`boundarycondition` function.
"""

from __future__ import annotations

from typing import List, Union, Tuple, Callable, Iterable, TypeGuard

import numpy as np
from dolfinx import fem, mesh, default_scalar_type
import ufl  # type: ignore

from .common import PDECONFIG
from elastodynamicsx.utils import _get_functionspace_tags_marker


class BoundaryConditionBase:
    """
    .. role:: python(code)
       :language: python

    Base class for representating of a variety of boundary conditions (BCs).
    The proper way to instantiate a BC is using the :python:`boundarycondition` function.

    BCs split into three families:
        - **BCStrongBase** class: strong BCs -> map to fem.DirichletBC
        - **BCMPCBase** class: multi-point constraints BCs -> used in dolfinx_mpc
        - **BCWeakBase** class: weak BCs -> terms added to the weak forms
    """

    # ## ### ### ## #
    # ## static  ## #
    # ## ### ### ## #

    labels: List[str]
    """String identifier(s) of the class, recognized by the `boundarycondition` function"""

    # ## ### ### ### ## #
    # ## non-static  ## #
    # ## ### ### ### ## #

    def __init__(self, functionspace_tags_marker, **kwargs):
        #
        # self._values: Union[fem.Function, fem.Constant, np.ndarray, Tuple, None]
        self._bc: Union[fem.DirichletBC, Callable, Tuple[Union[mesh.MeshTags, None], Union[int, None], Callable]]

        function_space, facet_tags, marker = _get_functionspace_tags_marker(functionspace_tags_marker)

        self._function_space: fem.FunctionSpaceBase = function_space
        self._facet_tags: Union[mesh.MeshTags, None] = facet_tags
        self._marker: Union[Tuple[int], int, None] = marker

    @property
    def bc(self):
        """The boundary condition"""
        return self._bc

    @property
    def function_space(self):
        return self._function_space

    @property
    def facet_tags(self):
        return self._facet_tags

    @property
    def marker(self):
        return self._marker


class BCStrongBase(BoundaryConditionBase):
    """
    Base class for representating strong BCs.
    This class carries a `fem.DirichletBC` object.
    """
    pass


class BCMPCBase(BoundaryConditionBase):
    """
    Base class for representating multi-point constraints BCs.
    Requires `dolfinx_mpc` to be installed.
    """
    pass


class BCWeakBase(BoundaryConditionBase):
    """
    Base class for representating weak BCs.
    """

    def __init__(self, functionspace_tags_marker, **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)
        md: dict = kwargs.get('metadata', PDECONFIG.default_metadata)
        self._ds = ufl.Measure("ds",
                               domain=self.function_space.mesh,
                               subdomain_data=self.facet_tags,
                               metadata=md)(self.marker)  # also valid if facet_tags or marker are None

    @property
    def ds(self):
        return self._ds

    def C_fn(self, u, v):
        return None

    def K_fn(self, u, v):
        return None

    def b_fn(self, v):
        return None

# ### ### ### #
# strong BCs  #
# ### ### ### #


class BCDirichlet(BCStrongBase):
    """
    Representation of a Dirichlet BC. Set :math:`u = u_0`,
    with :math:`u_0` the prescribed displacement.
    """

    labels: List[str] = ['dirichlet']

    def __init__(self, functionspace_tags_marker,
                 value: Union[fem.Function, fem.Constant, np.ndarray], **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)
        self._value = value

        function_space, facet_tags, marker = self.function_space, self.facet_tags, self.marker
        assert not (facet_tags is None), "facet_tags must not be None"
        assert not (marker is None), "marker must not be None"
        fdim = function_space.mesh.topology.dim - 1
        facets = facet_tags.find(marker)
        dofs = fem.locate_dofs_topological(function_space, fdim, facets)
        self._bc = fem.dirichletbc(value, dofs, function_space)

    @property
    def value(self):
        """Value of the prescribed displacement :math:`u_0`"""
        return self._value


class BCClamp(BCDirichlet):
    """
    Same as a Dirichlet BC with value set to zero -> Set :math:`u = 0`.
    """

    labels: List[str] = ['clamp']

    def __init__(self, functionspace_tags_marker, **kwargs):

        function_space, _, _ = _get_functionspace_tags_marker(functionspace_tags_marker)
        # number of components if vector space, 0 if scalar space
        nbcomps = function_space.element.num_sub_elements

        z0s = np.array([0] * nbcomps, dtype=default_scalar_type) if nbcomps > 0 else default_scalar_type(0)
        value = fem.Constant(function_space.mesh, z0s)

        super().__init__(functionspace_tags_marker, value, **kwargs)

# ### ### ### #
#  weak BCs   #
# ### ### ### #


class BCNeumann(BCWeakBase):
    """
    Representation of a Neumann BC. Prescribes :math:`\sigma(u).n = t_n`,
    with :math:`t_n` a boundary traction.
    """

    labels: List[str] = ['neumann']

    def __init__(self, functionspace_tags_marker,
                 value: Union[fem.Function, fem.Constant, np.ndarray], **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)
        self._value = value

    def b_fn(self, v):
        return ufl.inner(self.value, v) * self.ds

    @property
    def value(self):
        """Value of the :math:`t_n` boundary traction"""
        return self._value


class BCFree(BCNeumann):
    """
    Same as a Neumann BC with value set to zero. This BC has actually no effect on the PDE.
    """

    labels: List[str] = ['free']

    def __init__(self, functionspace_tags_marker, **kwargs):

        function_space, _, _ = _get_functionspace_tags_marker(functionspace_tags_marker)
        # number of components if vector space, 0 if scalar space
        nbcomps = function_space.element.num_sub_elements

        z0s = np.array([0] * nbcomps, dtype=default_scalar_type) if nbcomps > 0 else default_scalar_type(0)
        value = fem.Constant(function_space.mesh, z0s)

        super().__init__(functionspace_tags_marker, value, **kwargs)

    def b_fn(self, v):
        return None


class BCRobin(BCWeakBase):
    """
    Representation of a Robin BC. Prescribes :math:`\sigma(u).n = z * (u - s)`,
    with :math:`z` an impedance matrix and :math:`s` a given vector.
    """

    labels: List[str] = ['robin']

    def __init__(self, functionspace_tags_marker,
                 value1: Union[fem.Function, fem.Constant, np.ndarray],
                 value2: Union[fem.Function, fem.Constant, np.ndarray], **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)

        # number of components if vector space, 0 if scalar space
        nbcomps = self.function_space.element.num_sub_elements

        if nbcomps > 0:
            value1, value2 = ufl.as_matrix(value1), ufl.as_vector(value2)

        self._value1 = value1
        self._value2 = value2

    def K_fn(self, u, v):
        return -ufl.inner(self.value1 * u, v) * self.ds

    def b_fn(self, v):
        return -ufl.inner(self.value1 * self.value2, v) * self.ds

    @property
    def value1(self):
        return self._value1

    @property
    def value2(self):
        return self._value2


class BCDashpot(BCWeakBase):
    """
    Representation of a Dashpot BC, i.e. an impedance condition involving the velocity.
    Prescribes :math:`\sigma(u).n = z_n * v_n + z_t * v_t`,
    with :math:`z_n` and :math:`z_t` the normal and transverse impedances, and
    with :math:`v_n` and :math:`v_t` the normal and transverse components of the velocity.
    """

    labels: List[str] = ['dashpot']

    def __init__(self, functionspace_tags_marker,
                 *values: Union[fem.Function, fem.Constant, np.ndarray],
                 **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)

        self._values = values
        ds = self._ds
        function_space = self.function_space
        # number of components if vector space, 0 if scalar space
        nbcomps = function_space.element.num_sub_elements

        self._C_fn: Callable

        # scalar function space
        if nbcomps == 0:
            assert len(values) == 1
            value = values[0]

            def _C_fn(u_t, v):
                return value * ufl.inner(u_t, v) * ds

            self._C_fn = _C_fn

        # vector function space
        else:
            assert len(values) == 2
            if function_space.mesh.topology.dim == 1:
                def _C_fn(u_t, v):
                    return ((values[0] - values[1]) * ufl.inner(u_t[0], v[0]) + values[1] * ufl.inner(u_t, v)) * ds

                self._C_fn = _C_fn

            else:
                n = ufl.FacetNormal(function_space.mesh)
                dim = len(n)
                if nbcomps > dim:
                    n = ufl.as_vector([n[i] for i in range(dim)] + [0 for i in range(nbcomps - dim)])

                def _C_fn(u_t, v):
                    return ((values[0] - values[1]) * ufl.dot(u_t, n) * ufl.inner(n, v)
                            + values[1] * ufl.inner(u_t, v)) * ds

                self._C_fn = _C_fn

    def C_fn(self, u, v):
        return self._C_fn(u, v)

    @property
    def values(self):
        return self._values

# ### ### ### ### ### ### ### #
# Multi-point constraints BCs #
# ### ### ### ### ### ### ### #


class BCPeriodic(BCMPCBase):
    """
    Representation of a Periodic BC (translation), such as :math:`u(x_2) = u(x_1)`,
    with :math:`x_2 = x_1 + P` and :math:`P` a translation vector from the slave
    to the master boundary
    """

    labels: List[str] = ['periodic']

    def __init__(self, functionspace_tags_marker,
                 *values: Union[fem.Function, fem.Constant, np.ndarray],
                 **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)

        facet_tags = self.facet_tags
        marker = self.marker

        assert isinstance(facet_tags, mesh.MeshTags), "Periodic BC requires a specified facet_tags"
        assert isinstance(marker, int), "Periodic BC requires a single facet tag"

        P = np.asarray(values[0])

        def slave_to_master_map(x: np.ndarray):
            return x + P[:, np.newaxis]

        self._bc = (facet_tags, marker, slave_to_master_map)

# ### ### #
# filters #
# ### ### #


def get_dirichlet_BCs(bcs: Union[Iterable[Union[BoundaryConditionBase, fem.DirichletBC]],
                      Tuple[()]]) -> List[fem.DirichletBC]:
    """Returns the BCs of Dirichlet type in bcs"""
    out: List[fem.DirichletBC] = []
    for bc in bcs:
        if isinstance(bc, fem.DirichletBC):
            # assert not (isinstance(bc, BoundaryConditionBase))
            out.append(bc)
        elif isinstance(bc, BCStrongBase):
            out.append(bc.bc)

    return out


def get_mpc_BCs(bcs: Union[Iterable[Union[BoundaryConditionBase, fem.DirichletBC]],
                Tuple[()]]) -> List[BCMPCBase]:
    """Returns the BCs to be built with dolfinx_mpc in bcs"""
    def f(bc: Union[BoundaryConditionBase, fem.DirichletBC]) -> TypeGuard[BCMPCBase]:
        return isinstance(bc, BCMPCBase)

    return list(filter(f, bcs))


def get_weak_BCs(bcs: Union[Iterable[Union[BoundaryConditionBase, fem.DirichletBC]],
                 Tuple[()]]) -> List[BCWeakBase]:
    """Returns the weak BCs in bcs"""
    def f(bc: Union[BoundaryConditionBase, fem.DirichletBC]) -> TypeGuard[BCWeakBase]:
        return isinstance(bc, BCWeakBase)

    return list(filter(f, bcs))


# ### ### #
# builder #
# ### ### #

all_bcs = [BCDirichlet, BCClamp,
           BCNeumann, BCFree, BCRobin, BCDashpot,
           BCPeriodic]

all_strong_bcs = list(filter(lambda bctype: issubclass(bctype, BCStrongBase), all_bcs))
all_weak_bcs = list(filter(lambda bctype: issubclass(bctype, BCWeakBase), all_bcs))
all_mpc_bcs = list(filter(lambda bctype: issubclass(bctype, BCMPCBase), all_bcs))


def boundarycondition(functionspace_tags_marker, type_: str, *args, **kwargs) -> BoundaryConditionBase:
    """
    Builder function that instanciates the desired boundary condition (BC) type.

    Possible BCs are:
        'Free', 'Clamp'
        'Dirichlet', 'Neumann', 'Robin'
        'Periodic'
        'Dashpot'

    Args:
        functionspace_tags_marker: The function space, facet tags and marker(s) where the BC is applied
        type_: String identifier of the BC
        *args: Passed to the required BC

    Keyword Args:
        **kwargs: Passed to the required BC

    Example:
        General:

        .. highlight:: python
        .. code-block:: python

          # Imposes u = u_D on boundary n°1
          u_D = fem.Constant(V.mesh, [0.2,1.3])
          bc = boundarycondition((V, facet_tags, 1), 'Dirichlet', u_D)

          # Imposes sigma(u).n = T_N on boundary n°1
          T_N = fem.Function(V)
          T_N.interpolate(fancy_function)
          bc = boundarycondition((V, facet_tags, 1), 'Neumann', T_N)

          # Imposes sigma(u).n = z * (u - s) on boundary n°1
          z = ufl.as_matrix([[1, 2], [3, 4]])
          s = ufl.as_vector([0, 1])
          bc = boundarycondition((V, facet_tags, 1), 'Robin', z, s)

        Free or Clamp, on one or several tags:

        .. highlight:: python
        .. code-block:: python

          # Imposes sigma(u).n=0 on boundary n°1
          bc = boundarycondition((V, facet_tags, 1), 'Free')

          # Imposes u=0 on boundaries n°1 and 2
          bc = boundarycondition((V, facet_tags, (1, 2)), 'Clamp')

          # Apply BC to all boundaries when facet_tags and marker are not specified
          # or set to None
          bc = boundarycondition(V, 'Clamp')

        Periodic:

        .. highlight:: python
        .. code-block:: python

          # Given:   x(2) = x(1) - P
          # Imposes: u(2) = u(1)
          # Where: x(i) are the coordinates on boundaries n°1, 2
          #        P is a constant (translation) vector from slave to master
          #        u(i) is the field on boundaries n°1,2
          # Note that boundary n°2 is slave

          Px, Py, Pz = length, 0, 0 #for x-periodic, and tag=left
          Px, Py, Pz = 0, height, 0 #for y-periodic, and tag=bottom
          Px, Py, Pz = 0, 0, depth  #for z-periodic, and tag=back
          P  = [Px,Py,Pz]
          bc = boundarycondition((V, facet_tags, 2), 'Periodic', P)

        BCs involving the velocity:

        .. highlight:: python
        .. code-block:: python

          # (for a scalar function_space)
          # Imposes sigma(u).n = z * v on boundary n°1, with v=du/dt. Usually z=rho*c
          bc = boundarycondition((V, facet_tags, 1), 'Dashpot', z)

          # (for a vector function_space)
          # Imposes sigma(u).n = z_N * v_N + z_T * v_T on boundary n°1,
          # with v = du/dt, v_N = (v.n) n, v_T = v - v_N
          # Usually z_N = rho * c_L and z_T = rho * c_S
          bc = boundarycondition((V, facet_tags, 1), 'Dashpot', z_N, z_T)

    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
    """
    for BC in all_bcs:
        assert issubclass(BC, BoundaryConditionBase)
        if type_.lower() in BC.labels:
            return BC(functionspace_tags_marker, *args, **kwargs)
    #
    raise TypeError('unknown boundary condition type: ' + type_)
