# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import List, Union, Tuple, Callable, Iterable, TypeGuard

import numpy as np
from dolfinx import fem, mesh, default_scalar_type
import ufl  # type: ignore

from .common import PDECONFIG
from elastodynamicsx.utils import _get_functionspace_tags_marker


class BoundaryCondition:
    """
    Representation of a variety of boundary conditions (BCs)

    Possible BCs are:
        'Free', 'Clamp'
        'Dirichlet', 'Neumann', 'Robin'
        'Periodic'
        'Dashpot'

    Args:
        functionspace_tags_marker: The function space, facet tags and marker(s) where the BC is applied
        type_: String identifier of the BC
        values: The value(s) defining the BC, as ufl-compatible object(s)

    Keyword Args:
        metadata: dict, compiler-specific parameters, passed to ufl.Measure(..., metadata=metadata)

    Example:
        General:

        .. highlight:: python
        .. code-block:: python

          # Imposes u = u_D on boundary n°1
          u_D = fem.Constant(V.mesh, [0.2,1.3])
          bc = BoundaryCondition((V, facet_tags, 1), 'Dirichlet', u_D)

          # Imposes sigma(u).n = T_N on boundary n°1
          T_N = fem.Function(V)
          T_N.interpolate(fancy_function)
          bc = BoundaryCondition((V, facet_tags, 1), 'Neumann', T_N)

          # Imposes sigma(u).n = r * (u - s) on boundary n°1
          r = ufl.as_matrix([[1,2],[3,4]])
          s = ufl.as_vector([0,1])
          bc = BoundaryCondition((V, facet_tags, 1), 'Robin', (r, s))

        Free or Clamp, on one or several tags:

        .. highlight:: python
        .. code-block:: python

          # Imposes sigma(u).n=0 on boundary n°1
          bc = BoundaryCondition((V, facet_tags, 1), 'Free')

          # Imposes u=0 on boundaries n°1 and 2
          bc = BoundaryCondition((V, facet_tags, (1,2)), 'Clamp')

          # Apply BC to all boundaries when facet_tags and marker are not specified
          # or set to None
          bc = BoundaryCondition(V, 'Clamp')

        Periodic:

        .. highlight:: python
        .. code-block:: python

          # Given:   x(2) = x(1) - P
          # Imposes: u(2) = u(1)
          # Where: x(i) are the coordinates on boundaries n°1,2
          #        P is a constant (translation) vector from slave to master
          #        u(i) is the field on boundaries n°1,2
          # Note that boundary n°2 is slave

          Px, Py, Pz = length, 0, 0 #for x-periodic, and tag=left
          Px, Py, Pz = 0, height, 0 #for y-periodic, and tag=bottom
          Px, Py, Pz = 0, 0, depth  #for z-periodic, and tag=back
          P  = [Px,Py,Pz]
          bc = BoundaryCondition((V, facet_tags, 2), 'Periodic', P)

        BCs involving the velocity:

        .. highlight:: python
        .. code-block:: python

          # (for a scalar function_space)
          # Imposes sigma(u).n = z * v on boundary n°1, with v=du/dt. Usually z=rho*c
          bc = BoundaryCondition((V, facet_tags, 1), 'Dashpot', z)

          # (for a vector function_space)
          # Imposes sigma(u).n = z_N * v_N + z_T * v_T on boundary n°1,
          # with v = du/dt, v_N = (v.n) n, v_T = v - v_N
          # Usually z_N = rho * c_L and z_T = rho * c_S
          bc = BoundaryCondition((V, facet_tags, 1), 'Dashpot', (z_N, z_T))

    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
    """

    # ## ### ### ## #
    # ## static  ## #
    # ## ### ### ## #

    @staticmethod
    def get_dirichlet_BCs(bcs: Union[Tuple[Union[BoundaryCondition, fem.DirichletBC]],
                          Tuple[()]]) -> List[fem.DirichletBC]:
        """Returns the BCs of Dirichlet type in bcs"""
        out = []
        for bc in bcs:
            if issubclass(type(bc), fem.DirichletBC):
                assert not (isinstance(bc, BoundaryCondition))
                out.append(bc)
            elif isinstance(bc, BoundaryCondition) and issubclass(type(bc.bc), fem.DirichletBC):
                out.append(bc.bc)
        return out

    @staticmethod
    def get_mpc_BCs(bcs: Union[Tuple[Union[BoundaryCondition, fem.DirichletBC]],
                    Tuple[()]]) -> List[BoundaryCondition]:
        """Returns the BCs to be built with dolfinx_mpc in bcs"""
        out = []
        for bc in bcs:
            if isinstance(bc, BoundaryCondition) and (bc.type == 'periodic'):
                out.append(bc)
        return out

    @staticmethod
    def get_weak_BCs(bcs: Union[Tuple[Union[BoundaryCondition, fem.DirichletBC]],
                     Tuple[()]]) -> List[BoundaryCondition]:
        """Returns the weak BCs in bcs"""
        out = []
        for bc in bcs:
            if isinstance(bc, BoundaryCondition):
                test_mpc = bc.type == 'periodic'
                test_d = issubclass(type(bc.bc), fem.DirichletBC)
                if not (test_mpc or test_d):
                    out.append(bc)
        return out

    # ## ### ### ### ## #
    # ## non-static  ## #
    # ## ### ### ### ## #

    def __init__(self, functionspace_tags_marker, type_: str,
                 values: Union[fem.Function, fem.Constant, np.ndarray, Tuple, None] = None, **kwargs):
        #
        function_space, facet_tags, marker = _get_functionspace_tags_marker(functionspace_tags_marker)

        type_ = type_.lower()
        nbcomps = function_space.element.num_sub_elements  # number of components if vector space, 0 if scalar space

        # shortcuts: Free->Neumann with value=0; Clamp->Dirichlet with value=0
        if type_ == "free":
            type_ = "neumann"
            z0s = np.array([0] * nbcomps, dtype=default_scalar_type) if nbcomps > 0 else default_scalar_type(0)
            values = fem.Constant(function_space.mesh, z0s)

        elif type_ == "clamp":
            type_ = "dirichlet"
            z0s = np.array([0] * nbcomps, dtype=default_scalar_type) if nbcomps > 0 else default_scalar_type(0)
            values = fem.Constant(function_space.mesh, z0s)

        assert not (values is None)

        self._type: str = type_
        self._values = values
        self._bc: Union[fem.DirichletBC, Callable, Tuple[Union[mesh.MeshTags, None], Union[int, None], Callable]]

        md = kwargs.get('metadata', PDECONFIG.default_metadata)
        ds = ufl.Measure("ds",
                         domain=function_space.mesh,
                         subdomain_data=facet_tags,
                         metadata=md)(marker)  # also valid if facet_tags or marker are None

        if type_ == "dirichlet":
            assert not (isinstance(values, tuple))
            assert not (facet_tags is None), "facet_tags must not be None"
            assert not (marker is None), "marker must not be None"
            fdim = function_space.mesh.topology.dim - 1
            facets = facet_tags.find(marker)
            dofs = fem.locate_dofs_topological(function_space, fdim, facets)
            self._bc = fem.dirichletbc(values, dofs, function_space)

        elif type_ == "neumann":
            self._bc = lambda v: ufl.inner(values, v) * ds  # Linear form

        elif type_ == "robin":
            r_, s_ = values
            if nbcomps > 0:
                r_, s_ = ufl.as_matrix(r_), ufl.as_vector(s_)
            self._bc = lambda u, v: ufl.inner(r_ * (u - s_), v) * ds  # Bilinear form

        elif type_ == 'dashpot':
            # scalar function space
            if nbcomps == 0:
                # Bilinear form, to be applied on du/dt
                self._bc = lambda u_t, v: values * ufl.inner(u_t, v) * ds

            # vector function space
            else:
                if function_space.mesh.topology.dim == 1:
                    # Bilinear form, to be applied on du/dt
                    self._bc = lambda u_t, v: ((values[0] - values[1]) * ufl.inner(u_t[0], v[0])
                                               + values[1] * ufl.inner(u_t, v)) * ds
                else:
                    n = ufl.FacetNormal(function_space.mesh)
                    dim = len(n)
                    if nbcomps > dim:
                        n = ufl.as_vector([n[i] for i in range(dim)] + [0 for i in range(nbcomps - dim)])
                    # Bilinear form, to be applied on du/dt
                    self._bc = lambda u_t, v: ((values[0] - values[1]) * ufl.dot(u_t, n) * ufl.inner(n, v)
                                               + values[1] * ufl.inner(u_t, v)) * ds

        elif type_ == 'periodic':
            assert isinstance(marker, int), "Periodic BC requires a single facet tag"
            P = np.asarray(values)

            def slave_to_master_map(x: np.ndarray):
                return x + P[:, np.newaxis]
            self._bc = (facet_tags, marker, slave_to_master_map)

        # -- # -- # -- #
        elif type_ == 'periodic-do-not-use':  # TODO: try to calculate P from two given boundary markers
            assert len(marker) == 2, "Periodic BC requires two facet tags"  # type: ignore
            fdim = function_space.mesh.topology.dim - 1
            marker_master, marker_slave = marker  # type: ignore
            # facets_master = facet_tags.find(marker_master)  # not available on dolfinx v0.4.1
            facets_master = facet_tags.indices[facet_tags.values == marker_master]  # type: ignore # <- temporary
            facets_slave = facet_tags.indices[facet_tags.values == marker_slave]  # type: ignore # <- temporary
            dofs_master = fem.locate_dofs_topological(function_space, fdim, facets_master)
            dofs_slave = fem.locate_dofs_topological(function_space, fdim, facets_slave)
            x = function_space.tabulate_dof_coordinates()
            xm, xs = x[dofs_master], x[dofs_slave]  # noqa
            slave_to_master_map = None  # type: ignore
            raise NotImplementedError

        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type_))

    @property
    def bc(self):
        """The boundary condition"""
        return self._bc

    @property
    def type(self) -> str:
        """The string identifier of the BC"""
        return self._type

    @property
    def values(self) -> Union[fem.Function, fem.Constant, np.ndarray, Tuple]:
        return self._values


class BoundaryConditionBase:
    """
    Docstring here.
    """

    # ## ### ### ## #
    # ## static  ## #
    # ## ### ### ## #

    labels: List[str]

    @staticmethod
    def get_dirichlet_BCs(bcs: Union[Tuple[Union[BoundaryCondition, fem.DirichletBC]],
                          Tuple[()]]) -> List[fem.DirichletBC]:
        """Returns the BCs of Dirichlet type in bcs"""
        out = []
        for bc in bcs:
            if issubclass(type(bc), fem.DirichletBC):
                assert not (isinstance(bc, BoundaryCondition))
                out.append(bc)
            elif isinstance(bc, BoundaryCondition) and issubclass(type(bc.bc), fem.DirichletBC):
                out.append(bc.bc)
        return out

    @staticmethod
    def get_mpc_BCs(bcs: Union[Tuple[Union[BoundaryCondition, fem.DirichletBC]],
                    Tuple[()]]) -> List[BoundaryCondition]:
        """Returns the BCs to be built with dolfinx_mpc in bcs"""
        out = []
        for bc in bcs:
            if isinstance(bc, BoundaryCondition) and (bc.type == 'periodic'):
                out.append(bc)
        return out

    @staticmethod
    def get_weak_BCs(bcs: Union[Tuple[Union[BoundaryCondition, fem.DirichletBC]],
                     Tuple[()]]) -> List[BoundaryCondition]:
        """Returns the weak BCs in bcs"""
        out = []
        for bc in bcs:
            if isinstance(bc, BoundaryCondition):
                test_mpc = bc.type == 'periodic'
                test_d = issubclass(type(bc.bc), fem.DirichletBC)
                if not (test_mpc or test_d):
                    out.append(bc)
        return out

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
        md: dict = kwargs.get('metadata', PDECONFIG.default_metadata)

        self._ds = ufl.Measure("ds",
                               domain=function_space.mesh,
                               subdomain_data=facet_tags,
                               metadata=md)(marker)  # also valid if facet_tags or marker are None

#    @property
#    def values(self):
#        return self._values

    @property
    def bc(self):
        """The boundary condition"""
        return self._bc

    @property
    def ds(self):
        return self._ds

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
    pass


class BCWeakBase(BoundaryConditionBase):
    pass


class BCMPCBase(BoundaryConditionBase):
    pass

# ### ### ### #
# strong BCs  #
# ### ### ### #


class BCDirichlet(BCStrongBase):

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
        return self._value


class BCClamp(BCDirichlet):
    # shortcut: Clamp->Dirichlet with value=0

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

    labels: List[str] = ['neumann']

    def __init__(self, functionspace_tags_marker,
                 value: Union[fem.Function, fem.Constant, np.ndarray], **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)

        ds = self._ds
        self._value = value
        self._bc = lambda v: ufl.inner(value, v) * ds  # Linear form

    @property
    def value(self):
        return self._value


class BCFree(BCNeumann):
    # shortcut: Free->Neumann with value=0

    labels: List[str] = ['free']

    def __init__(self, functionspace_tags_marker, **kwargs):

        function_space, _, _ = _get_functionspace_tags_marker(functionspace_tags_marker)
        # number of components if vector space, 0 if scalar space
        nbcomps = function_space.element.num_sub_elements

        z0s = np.array([0] * nbcomps, dtype=default_scalar_type) if nbcomps > 0 else default_scalar_type(0)
        value = fem.Constant(function_space.mesh, z0s)

        super().__init__(functionspace_tags_marker, value, **kwargs)


class BCRobin(BCWeakBase):

    labels: List[str] = ['robin']

    def __init__(self, functionspace_tags_marker,
                 value1: Union[fem.Function, fem.Constant, np.ndarray],
                 value2: Union[fem.Function, fem.Constant, np.ndarray], **kwargs):

        super().__init__(functionspace_tags_marker, **kwargs)

        ds = self._ds
        self._value1 = value1
        self._value2 = value2

        # number of components if vector space, 0 if scalar space
        nbcomps = self.function_space.element.num_sub_elements

        r_, s_ = value1, value2
        if nbcomps > 0:
            r_, s_ = ufl.as_matrix(r_), ufl.as_vector(s_)
        self._bc = lambda u, v: ufl.inner(r_ * (u - s_), v) * ds  # Bilinear form

    @property
    def value1(self):
        return self._value1

    @property
    def value2(self):
        return self._value2


class BCDashpot(BCWeakBase):

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

        # scalar function space
        if nbcomps == 0:
            assert len(values) == 1
            value = values[0]
            # Bilinear form, to be applied on du/dt
            self._bc = lambda u_t, v: value * ufl.inner(u_t, v) * ds

        # vector function space
        else:
            assert len(values) == 2
            if function_space.mesh.topology.dim == 1:
                # Bilinear form, to be applied on du/dt
                self._bc = lambda u_t, v: ((values[0] - values[1]) * ufl.inner(u_t[0], v[0])
                                           + values[1] * ufl.inner(u_t, v)) * ds
            else:
                n = ufl.FacetNormal(function_space.mesh)
                dim = len(n)
                if nbcomps > dim:
                    n = ufl.as_vector([n[i] for i in range(dim)] + [0 for i in range(nbcomps - dim)])
                # Bilinear form, to be applied on du/dt
                self._bc = lambda u_t, v: ((values[0] - values[1]) * ufl.dot(u_t, n) * ufl.inner(n, v)
                                           + values[1] * ufl.inner(u_t, v)) * ds

    @property
    def values(self):
        return self._values

# ### ### ### ### ### ### ### #
# Multi-point constraints BCs #
# ### ### ### ### ### ### ### #


class BCPeriodic(BCMPCBase):

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

all_strong_bcs = list(filter(lambda bctype: issubclass(bctype, BCStrongBase)))  # type: ignore
all_weak_bcs = list(filter(lambda bctype: issubclass(bctype, BCWeakBase)))  # type: ignore
all_mpc_bcs = list(filter(lambda bctype: issubclass(bctype, BCMPCBase)))  # type: ignore


def boundarycondition(functionspace_tags_marker, type_: str, *args, **kwargs) -> BoundaryConditionBase:
    for BC in all_bcs:
        assert issubclass(BC, BoundaryConditionBase)
        if type_.lower() in BC.labels:
            return BC(functionspace_tags_marker, *args, **kwargs)
    #
    raise TypeError('unknown boundary condition type: ' + type_)
