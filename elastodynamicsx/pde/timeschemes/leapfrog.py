# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from typing import Union, Callable, List, Tuple

from petsc4py import PETSc

from dolfinx import fem, default_scalar_type
import ufl  # type: ignore

from .timeschemebase import TimeScheme, FEniCSxTimeScheme
from elastodynamicsx.pde import PDECONFIG, _build_mpc
from elastodynamicsx.pde.boundaryconditions import BoundaryConditionBase, get_dirichlet_BCs, get_weak_BCs


class LeapFrog(FEniCSxTimeScheme):
    """
    Implementation of the *Leapfrog* time-stepping scheme, or *Explicit central difference scheme*.
    *Leapfrog* is a special case of *Newmark-beta* methods with :math:`\\beta=0` and :math:`\gamma=0.5`

    Scheme:
        | :math:`u_n`, :math:`v_n`, :math:`a_n` are the (known) displacement,
            velocity and acceleration at current time step
        | :math:`u_{n+1}` is the unknown: displacement at next time step
        | :math:`a_n = (u_{n+1} - 2 u_n + u_{n-1}) / dt^2`
        | :math:`v_n = (u_{n+1} - u_n-1) / (2 dt)`

    Implicit / explicit?
        Explicit

    Accuracy:
        Second-order

    Args:
        function_space: The Finite Element functionnal space
        M_fn: The mass form. Usually:

            :python:`M_fn = lambda u,v: rho* ufl.dot(u, v) * ufl.dx`

        C_fn (optional, ignored if None): The damping form. E.g. for Rayleigh damping:

            :python:`C_fn = lambda u,v: eta_m * M_fn(u,v) + eta_k * K_fn(u,v)`

        K_fn: The stiffness form. Usually:

            :python:`K_fn = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx`

        b_fn (optional, ignored if None): Right hand term
        dt: Time step
        bcs: The set of boundary conditions

    Keyword Args:
        jit_options: (default=PDECONFIG.default_jit_options) options for the just-in-time compiler

    See:
        https://en.wikipedia.org/wiki/Leapfrog_integration
    """

    labels = ['leapfrog', 'central-difference']

    def __init__(self, function_space: fem.FunctionSpace,
                 M_fn: Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form],
                 C_fn: Union[None, Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form]],
                 K_fn: Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form],
                 b_fn: Union[None, Callable[['ufl.TestFunction'], ufl.form.Form]],
                 dt,
                 bcs: Union[Tuple[BoundaryConditionBase], Tuple[()]] = (),
                 **kwargs):

        self.jit_options = kwargs.get('jit_options', PDECONFIG.default_jit_options)
        dt_ = fem.Constant(function_space.mesh, default_scalar_type(dt))

        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)

        self._u_n = fem.Function(function_space, name="u")  # u(t)
        self._u_nm1 = fem.Function(function_space)          # u(t-dt)
        self._u_nm2 = fem.Function(function_space)          # u(t-2*dt)
        #
        self._u0 = self._u_nm1
        self._v0 = self._u_n
        self._a0 = self._u_nm2

        # linear and bilinear forms for mass and stiffness matrices
        self._a = M_fn(u, v)
        self._L = -dt_ * dt_ * K_fn(self._u_nm1, v) + 2 * M_fn(self._u_nm1, v) - M_fn(self._u_nm2, v)

        self._m0_form = M_fn(u, v)
        self._L0_form = -K_fn(self._u0, v)

        _L_terms = []
        if not (b_fn is None):
            _L_terms.append(b_fn(v))

        # linear and bilinear forms for damping matrix if given
        if not (C_fn is None):
            C_uv_ufl = C_fn(u, v)
            if not (C_uv_ufl is None):
                self._a += 0.5 * dt_ * C_uv_ufl
                self._L += 0.5 * dt_ * C_fn(self._u_nm2, v)
                self._L0_form -= C_fn(self._v0, v)

        # boundary conditions
        mpc = _build_mpc(bcs)
        dirichletbcs = get_dirichlet_BCs(bcs)
        weak_BCs = get_weak_BCs(bcs)

        # damping term, BC
        self._a += 0.5 * dt_ * sum(filter(None, [bc.C_fn(u, v) for bc in weak_BCs]))
        self._L += 0.5 * dt_ * sum(filter(None, [bc.C_fn(self._u_nm2, v) for bc in weak_BCs]))

        # stiffness term, BC
        self._a += dt_ * dt_ * sum(filter(None, [bc.K_fn(u, v) for bc in weak_BCs]))

        # load term, BC
        _L_terms += [bc.b_fn(v) for bc in weak_BCs]

        # initial value rhs, BC
        self._L0_form += sum(filter(None, [bc.C_fn(self._v0, v) for bc in weak_BCs]))
        self._L0_form += sum(filter(None, [bc.K_fn(self._u0, v) for bc in weak_BCs]))

        # sum and report into forms
        _L_terms_sum = sum(filter(None, _L_terms))
        self._L += dt_ * dt_ * _L_terms_sum
        self._L0_form += _L_terms_sum

        # compile forms
        bilinear_form = fem.form(self._a, jit_options=self.jit_options)
        linear_form = fem.form(self._L, jit_options=self.jit_options)
        #
        super().__init__(dt, self._u_n, bilinear_form, linear_form, mpc, dirichletbcs,
                         explicit=True, nbsteps=1, linear_ODE=True, **kwargs)

    @property
    def u(self) -> fem.Function:
        """The displacement field at current time step"""
        return self._u_n

    @property
    def u_nm1(self) -> fem.Function:
        """The displacement field at previous time step"""
        return self._u_nm1

    def prepareNextIteration(self) -> None:
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        self._u_nm2.x.array[:] = self._u_nm1.x.array
        self._u_nm1.x.array[:] = self._u_n.x.array

    def initialStep(self,
                    t0,
                    callfirsts: List[Callable] = [],
                    callbacks: List[Callable] = [],
                    verbose: int = 0) -> None:

        # ## -------------------------------------------------
        #    --- first step: given u0 and v0, solve for a0 ---
        # ## -------------------------------------------------
        #
        if verbose >= 10:
            PETSc.Sys.Print('Solving the initial value step')  # type: ignore[attr-defined]
            PETSc.Sys.Print('Callfirsts...')  # type: ignore[attr-defined]

        for callfirst in callfirsts:
            callfirst(t0)  # <- update stuff

        # Known: u0, v0.
        # Solve for a0.
        # u1 is directly obtained from u0, v0, a0 (explicit scheme)
        # u2 requires to solve a new system (enter the time loop)
        problem = fem.petsc.LinearProblem(self._m0_form, self._L0_form, bcs=self._bcs, u=self._a0,
                                          petsc_options=TimeScheme.petsc_options_t0,
                                          jit_options=self.jit_options)
        problem.solve()

        u0, v0, a0 = self._u0.x.array, self._v0.x.array, self._a0.x.array
        u1 = self._u_n.x.array

        # remember that self._u_nm1 = self._u0 -> self._u_nm1 is already at the correct value
        dt = float(self.dt)  # keep good speed if self.dt is e.g. a fem.Constant
        u1[:] = u0 + dt * v0 + 0.5 * dt**2 * a0
        # unm1 is not needed
        # at this point: self._u_n = u1, self._u_nm1 = u0

        self.prepareNextIteration()

        if verbose >= 10:
            PETSc.Sys.Print('Initial value problem solved, entering loop')  # type: ignore[attr-defined]
        for callback in callbacks:
            callback(0, self._u_n.x.petsc_vec)  # <- store solution, plot, print, ...
        #
        # ## -------------------------------------------------
