# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

from petsc4py import PETSc
import numpy as np

from dolfinx import fem, default_scalar_type
import ufl  # type: ignore

try:
    import dolfinx_mpc
    from dolfinx_mpc import MultiPointConstraint
except ImportError:
    import warnings
    warnings.warn("Can't import dolfinx_mpc. Periodic boundaries are not available", Warning)
    dolfinx_mpc = None  # type: ignore
    MultiPointConstraint = None  # type: ignore

from .buildmpc import _build_mpc
from .common import PDECONFIG
from .boundaryconditions import get_dirichlet_BCs, get_mpc_BCs, get_weak_BCs, BCWeakBase, BCMPCBase
from .materials import Material


class PDE:
    """
    Representation of a PDE of the kind:

        M*a + C*v + K(u) = b
        + Boundary conditions

    as an assembly of materials, forces and bcs defined over different subdomains.

    Args:
        functions_space
        materials: a list of pde.Material instances

    Keyword Args:
        bodyforces: (default=[]) a list of pde.BodyForce instances
        bcs: (default=[]) a list of fem.DirichletBCMetaClass and/or pde.BoundaryConditionBase instances
        jit_options: (default=PDECONFIG.default_jit_options) options for the just-in-time compiler
        finalize: (default=True) call self.finalize() on build
    """

    def __init__(self, function_space: fem.FunctionSpaceBase, materials: typing.Iterable[Material], **kwargs):
        self._function_space = function_space
        self.materials = materials
        self.bodyforces = kwargs.get('bodyforces', [])
        self.bcs = kwargs.get('bcs', [])
        self.jit_options = kwargs.get('jit_options', PDECONFIG.default_jit_options)
        self._u = ufl.TrialFunction(function_space)
        self._v = ufl.TestFunction(function_space)
        self.verbose: bool = kwargs.get('verbose', False)

        # Declare stuff without building
        self._mpc: typing.Union[MultiPointConstraint, None] = None
        self._M_form: typing.Union[fem.forms.Form, None] = None
        self._C_form: typing.Union[fem.forms.Form, None] = None
        self._K_form: typing.Union[fem.forms.Form, None] = None
        self._b_form: typing.Union[fem.forms.Form, None] = None
        self._K0_form: typing.Union[fem.forms.Form, None] = None
        self._K1_form: typing.Union[fem.forms.Form, None] = None
        self._K2_form: typing.Union[fem.forms.Form, None] = None

        # ## Sort boundary conditions
        self._bcs_weak: typing.List[BCWeakBase] = get_weak_BCs(self.bcs)
        self._bcs_strong: typing.List[fem.DirichletBC] = get_dirichlet_BCs(self.bcs)
        self._bcs_mpc: typing.List[BCMPCBase] = get_mpc_BCs(self.bcs)

        self._omega_ufl = fem.Constant(function_space.mesh, default_scalar_type(0))

        # Finalize the PDE (compile, ...). Optionnally this can be done
        # manually later on by passing kwargs['finalize']=False
        if kwargs.get('finalize', True):
            self.finalize()

    @property
    def is_linear(self) -> bool:
        return not (sum([mat.is_linear is False for mat in self.materials]))

    @property
    def mpc(self):
        return self._mpc

# ## ### ### ### ### ### ### ### ### ## #
# ## Finalize the PDE (compile, ...) ## #
# ## ### ### ### ### ### ### ### ### ## #

    def finalize(self) -> None:
        """
        Finalize dolfinx-mpc objects
        """
        self._build_mpc()
        if self._mpc is None:
            self.update_b_frequencydomain = self._update_b_frequencydomain_WO_MPC
        else:
            self.update_b_frequencydomain = self._update_b_frequencydomain_WITH_MPC

        if self.verbose:
            if self.is_linear:
                print('linear PDE')
            else:
                print('non-linear PDE')

    def _build_mpc(self) -> None:
        """Required for handling multi-point constraints (e.g. periodic BC)"""
        if dolfinx_mpc is None:  # if import error
            return
        #
        # if there is no need to use MPC. Note that MPC slows down the runs even if
        # there is no constraint -> better avoid using it if not needed
        if len(self._bcs_mpc) == 0:
            return
        #
        self._mpc = _build_mpc(self._bcs_strong + self._bcs_mpc)

    def _compile_M(self) -> None:
        u, v = self._u, self._v
        M_ufl = self.M_fn(u, v)
        self._M_form = fem.form(M_ufl, jit_options=self.jit_options)

    def _compile_C_K_b(self) -> None:
        """Required for frequency domain or eigenvalue problems"""
        u, v = self._u, self._v

        zero = fem.Constant(self._function_space.mesh, default_scalar_type(0.))
        vzero = zero
        if v.ufl_function_space().num_sub_spaces != 0:  # VectorFunctionSpace
            vzero = fem.Constant(self._function_space.mesh, default_scalar_type([0.] * len(v)))

        # Interior
        k = self.K_fn(u, v)

        # Retrieve the integral measures in k to build compatible default zero forms 'c' and 'L'
        measures = [ufl.Measure("dx",
                                domain=self._function_space.mesh,
                                subdomain_data=cint.subdomain_data(),
                                metadata=cint.metadata())(cint.subdomain_id()) for cint in k.integrals()]

        c_NULL = sum([zero * ufl.inner(u, v) * dx for dx in measures])
        L_NULL = sum([ufl.inner(vzero, v) * dx for dx in measures])
        c = [self.C_fn(u, v)]

        L = [self.b_fn(v)]

        # Boundaries
        k += sum(filter(None, [bc.K_fn(u, v) for bc in self._bcs_weak]))
        c += [bc.C_fn(u, v) for bc in self._bcs_weak]
        L += [bc.b_fn(v) for bc in self._bcs_weak]

        c = sum(filter(None, c)) if any(c) else c_NULL
        L = sum(filter(None, L)) if any(L) else L_NULL

        self._C_form = fem.form(c, jit_options=self.jit_options)
        self._K_form = fem.form(k, jit_options=self.jit_options)
        self._b_form = fem.form(L, jit_options=self.jit_options)

        # Executes the following only if using complex numbers
        if np.issubdtype(default_scalar_type, np.complexfloating):
            # #Mat_lhs = -w*w*_M_ + 1J*w*_C_ + _K_
            m = self.M_fn(u, v)
            w = self._omega_ufl
            self._a_form = fem.form(-w * w * m + 1J * w * c + k, jit_options=self.jit_options)  # type: ignore

    def _compile_K0_K1_K2(self) -> None:
        """Required for waveguide problems"""
        u, v = self._u, self._v

        assert self._function_space.element.basix_element.discontinuous is False, \
            'K0, K1, K2 are not implemented for a DG formulation'

        # Interior
        k0 = self.K0_fn(u, v)
        k1 = self.K1_fn(u, v)
        k2 = self.K2_fn(u, v)

        assert not any(filter(None, [bc.C_fn(u, v) for bc in self._bcs_weak])), \
            'C!=0 not yet implemented for waveguides'
        assert not any(filter(None, [bc.C_fn(u, v) for bc in self._bcs_weak])), \
            'K!=0 not yet implemented for waveguides'

        self._K0_form = fem.form(k0, jit_options=self.jit_options)
        self._K1_form = fem.form(k1, jit_options=self.jit_options)
        self._K2_form = fem.form(k2, jit_options=self.jit_options)

# ## ### ### ### ### ### ### ### ### ### ## #
# ## Linear and bilinear form functions  ## #
# ## ### ### ### ### ### ### ### ### ### ## #

    def M_fn(self, u, v):
        """(bilinear) Mass form function"""
        return sum([mat.M_fn(u, v) for mat in self.materials])

    def C_fn(self, u, v):
        """(bilinear) Damping form function"""
        c_ = [mat.C_fn(u, v) for mat in self.materials]
        return sum(filter(None, c_)) if any(c_) else None

    @property
    def K_fn(self) -> typing.Callable:
        """(bilinear) Stiffness form function"""
        return lambda u, v: sum([mat.K_fn(u, v) for mat in self.materials])

    @property
    def K0_fn(self) -> typing.Callable:
        """(bilinear) K0 stiffness form function (waveguide problems)"""
        return lambda u, v: sum([mat.K0_fn_CG(u, v) for mat in self.materials])

    @property
    def K1_fn(self) -> typing.Callable:
        """(bilinear) K1 stiffness form function (waveguide problems)"""
        return lambda u, v: sum([mat.K1_fn_CG(u, v) for mat in self.materials])

    @property
    def K2_fn(self) -> typing.Callable:
        """(bilinear) K2 stiffness form function (waveguide problems)"""
        return lambda u, v: sum([mat.K2_fn_CG(u, v) for mat in self.materials])

    @property
    def K_fn_CG(self) -> typing.Callable:
        """(bilinear) Stiffness form function (Continuous Galerkin)"""
        return lambda u, v: sum([mat.K_fn_CG(u, v) for mat in self.materials])

    @property
    def K_fn_DG(self) -> typing.Callable:
        """(bilinear) Stiffness form function (Discontinuous Galerkin)"""
        return lambda u, v: sum([mat.K_fn_DG(u, v) for mat in self.materials])

    @property
    def DG_numerical_flux(self) -> typing.Callable:
        """(bilinear) Numerical flux form function (Disontinuous Galerkin)"""
        return lambda u, v: sum([mat.DG_numerical_flux(u, v) for mat in self.materials])

    def b_fn(self, v):
        """Linear form function"""
        if len(self.bodyforces) == 0:
            return None
        else:
            return sum([f.b_fn(v) for f in self.bodyforces])

# ## ### ### ### ### ### ### ## #
# ## Compiled dolfinx forms  ## #
# ## ### ### ### ### ### ### ## #

    @property
    def M_form(self) -> typing.Union[fem.forms.Form, None]:
        """Compiled mass bilinear form"""
        return self._M_form

    @property
    def C_form(self) -> typing.Union[fem.forms.Form, None]:
        """Compiled damping bilinear form"""
        return self._C_form

    @property
    def K_form(self) -> typing.Union[fem.forms.Form, None]:
        """Compiled stiffness bilinear form"""
        return self._K_form

    @property
    def b_form(self) -> typing.Union[fem.forms.Form, None]:
        """Compiled linear form"""
        return self._b_form

# ## ### ### ### ### ### ### ### ## #
# ## PETSc matrices and vectors  ## #
# ## ### ### ### ### ### ### ### ## #

    def M(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Mass matrix"""
        if self._M_form is None:
            self._compile_M()

        assert not (self._M_form is None)

        if self._mpc is None:
            M = fem.petsc.assemble_matrix(self._M_form, bcs=self._bcs_strong)
        else:
            M = dolfinx_mpc.assemble_matrix(self._M_form, self._mpc, bcs=self._bcs_strong)
        M.assemble()
        return M

    def C(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Damping matrix"""
        if self._C_form is None:
            self._compile_C_K_b()

        assert not (self._C_form is None)

        if self._mpc is None:
            C = fem.petsc.assemble_matrix(self._C_form, bcs=self._bcs_strong)
        else:
            C = dolfinx_mpc.assemble_matrix(self._C_form, self._mpc, bcs=self._bcs_strong)
        C.assemble()
        return C

    def K(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """Stiffness matrix"""
        if self._K_form is None:
            self._compile_C_K_b()

        assert not (self._K_form is None)

        if self._mpc is None:
            K = fem.petsc.assemble_matrix(self._K_form, bcs=self._bcs_strong)
        else:
            K = dolfinx_mpc.assemble_matrix(self._K_form, self._mpc, bcs=self._bcs_strong)
        K.assemble()
        return K

    def b(self, omega=0) -> PETSc.Vec:  # type: ignore[name-defined]
        """Load vector"""
        b = self.init_b()
        self.update_b_frequencydomain(b, omega)
        return b

    def init_b(self) -> PETSc.Vec:  # type: ignore[name-defined]
        """Declares a zero vector compatible with the linear form"""
        if self._b_form is None:
            self._compile_C_K_b()

        assert not (self._b_form is None)

        if self._mpc is None:
            return fem.petsc.create_vector(self._b_form)
        else:
            return dolfinx_mpc.assemble_vector(self._b_form, self._mpc)

    def K0(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """K0 stiffness matrix (waveguide problems)"""
        if self._K0_form is None:
            self._compile_K0_K1_K2()

        assert not (self._K0_form is None)

        if self._mpc is None:
            K0 = fem.petsc.assemble_matrix(self._K0_form, bcs=self._bcs_strong)
        else:
            K0 = dolfinx_mpc.assemble_matrix(self._K0_form, self._mpc, bcs=self._bcs_strong)
        K0.assemble()
        return K0

    def K1(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """K1 stiffness matrix (waveguide problems)"""
        if self._function_space.mesh.geometry.dim == 3:  # special case: K1=K, K2=K3=0
            return None
        if self._K1_form is None:
            self._compile_K0_K1_K2()

        assert not (self._K1_form is None)

        if self._mpc is None:
            K1 = fem.petsc.assemble_matrix(self._K1_form, bcs=self._bcs_strong)
        else:
            K1 = dolfinx_mpc.assemble_matrix(self._K1_form, self._mpc, bcs=self._bcs_strong)
        K1.assemble()
        return K1

    def K2(self) -> PETSc.Mat:  # type: ignore[name-defined]
        """K2 stiffness matrix (waveguide problems)"""
        if self._function_space.mesh.geometry.dim == 3:  # special case: K0=K, K1=K2=0
            return None
        if self._K2_form is None:
            self._compile_K0_K1_K2()

        assert not (self._K2_form is None)

        if self._mpc is None:
            K2 = fem.petsc.assemble_matrix(self._K2_form, bcs=self._bcs_strong)
        else:
            K2 = dolfinx_mpc.assemble_matrix(self._K2_form, self._mpc, bcs=self._bcs_strong)
        K2.assemble()
        return K2

# ## ### ### ### ###  ## #
# ## Update functions ## #
# ## ### ### ### ###  ## #

    # def update_b_frequencydomain(self, b:PETSc.Vec, omega:float) -> None: #NOW SET TO EITHER METHOD BELOW IN __init__

    def _update_b_frequencydomain_WO_MPC(self, b: PETSc.Vec, omega: float) -> None:  # type: ignore[name-defined]
        """Updates the b vector (in-place) for a given angular frequency omega"""
        # set to 0
        with b.localForm() as loc_b:
            loc_b.set(0)

        # fill with values
        fem.petsc.assemble_vector(b, self.b_form)

        # BC modifyier
        self._omega_ufl.value = omega
        fem.petsc.apply_lifting(b, [self._a_form], [self._bcs_strong])

        # ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]

        # apply BC value
        fem.petsc.set_bc(b, self._bcs_strong)

    def _update_b_frequencydomain_WITH_MPC(self, b: PETSc.Vec, omega: float) -> None:  # type: ignore[name-defined]
        """Updates the b vector (in-place) for a given angular frequency omega"""
        assert not (self._mpc is None)

        # set to 0
        with b.localForm() as loc_b:
            loc_b.set(0)

        # fill with values
        # fem.petsc.assemble_vector(b, self.b_form)
        dolfinx_mpc.assemble_vector(self.b_form, self._mpc, b)

        # BC modifyier
        self._omega_ufl.value = omega
        # fem.petsc.apply_lifting(b, [self._a_form], [self._bcs_strong])
        dolfinx_mpc.apply_lifting(b, [self._a_form], [self._bcs_strong], self._mpc)

        # ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]

        # apply BC value
        fem.petsc.set_bc(b, self._bcs_strong)  # not modified by dolfinx_mpc
