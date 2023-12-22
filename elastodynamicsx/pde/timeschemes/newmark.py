# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from typing import Union, Callable, List, Tuple

from petsc4py import PETSc

from dolfinx import fem
import ufl  # type: ignore

from .timeschemebase import TimeScheme, FEniCSxTimeScheme
from elastodynamicsx.pde import PDECONFIG, _build_mpc
from elastodynamicsx.pde.boundaryconditions import BoundaryConditionBase, get_dirichlet_BCs, get_weak_BCs


class GalphaNewmarkBeta(FEniCSxTimeScheme):
    """
    .. role:: python(code)
      :language: python

    Implementation of the 'g-a-newmark' (*Generalized-alpha Newmark*) time-stepping scheme,
    for :math:`\\beta>0`. The special case :math:`\\beta=0` is implemented in the :python:`LeapFrog` class.

    Implicit / explicit?
        Implicit

    Accuracy:
        First or second order, depending on parameters

    Args:
        function_space: The Finite Element functionnal space
        m: The mass form. Usually:

            :python:`m = lambda u,v: rho* ufl.dot(u, v) * ufl.dx`

        c (optional, ignored if None): The damping form. E.g. for Rayleigh damping:

            :python:`c = lambda u,v: eta_m * m(u,v) + eta_k * k(u,v)`

        k: The stiffness form. Usually:

            :python:`k = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx`

        L (optional, ignored if None): Linear form
        dt: Time step
        bcs: The set of boundary conditions

    Keyword Args:
        rho_inf (default = 0.75): Spectral radius in the high frequency limit.
            Bounds: (1/2,1). Setting **rho_inf** is the preferred way of defining the scheme.

        alpha_m (default = (2*rho_inf-1)/(rho_inf+1)): Unconditionnal stability
            if :math:`-1 \leq \\alpha_m \leq \\alpha_f \leq 0.5`

        alpha_f (default = rho_inf/(rho_inf+1)): Unconditionnal stability
            if :math:`-1 \leq \\alpha_m \leq \\alpha_f \leq 0.5`

        gamma (default = 1/2 - alpha_m + alpha_f): Default value ensures second-order accuracy.
            Other values give first-order accuracy

        beta (default = 1/4*(gamma+1/2)**2): Unconditionnal stability
            if :math:`\\beta \geq 0.25 + 0.5 (\\alpha_f - \\alpha_m)`

        jit_options (default=PDECONFIG.default_jit_options): Options for the just-in-time compiler

    Reference:
        J. Chung and G. M. Hulbert, "A time integration algorithm for structural
        dynamics with improved numerical dissipation: The generalized-α method,"
        J. Appl. Mech. 60(2), 371–375 (1993).

    Adapted from:
        https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
    """
    labels = ['g-a-newmark', 'generalized-alpha']

    def __init__(self, function_space: fem.FunctionSpaceBase,
                 m_: Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form],
                 c_: Union[None, Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form]],
                 k_: Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form],
                 L: Union[None, Callable[['ufl.TestFunction'], ufl.form.Form]],
                 dt,
                 bcs: Union[Tuple[BoundaryConditionBase], Tuple[()]] = (), **kwargs):

        linear_ODE = kwargs.pop('linear', True)
        if linear_ODE is False:
            raise NotImplementedError

        self.jit_options = kwargs.get('jit_options', PDECONFIG.default_jit_options)
        rho_inf = kwargs.get('rho_inf', 0.75)
        alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
        alpha_f = rho_inf / (rho_inf + 1)
        alpha_m = kwargs.get('alpha_m', alpha_m)
        alpha_f = kwargs.get('alpha_f', alpha_f)
        gamma = kwargs.get('gamma', 0.5 - alpha_m + alpha_f)
        beta = kwargs.get('beta', 0.25 * (gamma + 1 / 2)**2)

        self.alpha_m, self.alpha_f, self.gamma, self.beta = alpha_m, alpha_f, gamma, beta

        def const(c):
            return fem.Constant(function_space.mesh, PETSc.ScalarType(c))
        c1 = const(dt * gamma * (1 - alpha_f) / beta)
        c2 = const(dt**2 * (1 - (gamma - alpha_f) / beta))
        c3 = const(dt**3 * (1 - alpha_f) * (1 - gamma / 2 / beta))

        m1 = const((1 - alpha_m) / beta)
        m2 = const(dt * (1 - alpha_m) / beta)
        m3 = const(dt**2 * (1 - (1 - alpha_m) / 2 / beta))

        dt_ = const(dt)
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self._u_n = fem.Function(function_space, name="u")  # u(t)
        self._v_n = fem.Function(function_space, name="v")  # u'(t)  = du/dt
        self._a_n = fem.Function(function_space, name="a")  # u''(t) = d(du/dt)/dt
        #
        # At the time of solving, a_n relates to step n-1.
        # But at the time of callbacks or end of loop, the update has been done
        # and relates to step n (same step as u_n)
        self._a_nm1 = self._a_n
        self._v_nm1 = self._v_n  # same convention as for a_n
        self._u_nm1 = fem.Function(function_space, name="u_nm1")  # u(t-dt)
        self._a_garb = fem.Function(function_space, name="garbage")  # Garbage
        #
        self._u0 = self._u_nm1
        self._v0 = self._v_nm1
        self._a0 = self._a_nm1

        # linear and bilinear forms for mass and stiffness matrices
        self._a = m1 * m_(u, v) + dt_ * dt_ * const(1 - alpha_f) * k_(u, v)
        self._L = -const(dt * dt * alpha_f) * k_(self._u_nm1, v) \
            + m1 * m_(self._u_nm1, v) + m2 * m_(self._v_nm1, v) - m3 * m_(self._a_nm1, v)

        self._m0_form = m_(u, v)
        self._L0_form = -k_(self._u0, v)

        if not (L is None):
            self._L += dt_ * dt_ * L(v)
            self._L0_form += L(v)

        # linear and bilinear forms for damping matrix if given
        if not (c_ is None):
            self._a += c1 * c_(u, v)
            self._L += c1 * c_(self._u_nm1, v) + c2 * c_(self._v_nm1, v) - c3 * c_(self._a_nm1, v)
            self._L0_form -= c_(self._v0, v)

        # boundary conditions
        mpc = _build_mpc(bcs)
        dirichletbcs = get_dirichlet_BCs(bcs)
        weak_BCs = get_weak_BCs(bcs)

        # damping term, BC
        d1 = dt * gamma / beta
        d2 = dt**2 * (1 - gamma / beta)
        d3 = dt**3 * (1 - gamma / beta / 2)
        u_c = d1 * self._u_nm1 - d2 * self._v_nm1 - d3 * self._a_nm1
        self._a += d1 * sum(filter(None, [bc.c(u, v) for bc in weak_BCs]))
        self._L += sum(filter(None, [bc.c(u_c, v) for bc in weak_BCs]))

        # stiffness term, BC
        self._a += dt_ * dt_ * sum(filter(None, [bc.k(u, v) for bc in weak_BCs]))

        # load term, BC
        self._L += dt_ * dt_ * sum(filter(None, [bc.L(v) for bc in weak_BCs]))

        # initial value rhs, BC
        self._L0_form += sum(filter(None, [bc.c(self._v0, v) for bc in weak_BCs]))
        self._L0_form += sum(filter(None, [bc.k(self._u0, v) for bc in weak_BCs]))
        self._L0_form += sum(filter(None, [bc.L(v) for bc in weak_BCs]))

        # compile forms
        bilinear_form = fem.form(self._a, jit_options=self.jit_options)
        linear_form = fem.form(self._L, jit_options=self.jit_options)
        #
        super().__init__(dt, self._u_n, bilinear_form, linear_form, mpc, dirichletbcs, intermediate_dt=self.alpha_f,
                         explicit=False, nbsteps=1, linear_ODE=linear_ODE, **kwargs)

    @property
    def u(self) -> fem.Function:
        return self._u_n

    @property
    def v(self) -> fem.Function:
        return self._v_n

    @property
    def a(self) -> fem.Function:
        return self._a_n

    def prepareNextIteration(self) -> None:
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        dt = self.dt
        beta, gamma = self.beta, self.gamma
        un, unm1 = self._u_n.x.array, self._u_nm1.x.array
        vnm1, anm1 = self._v_nm1.x.array, self._a_nm1.x.array

        self._a_garb.x.array[:] = (un - unm1 - dt * vnm1) / beta / dt**2 - (1 - 2 * beta) / 2 / beta * anm1
        self._v_n.x.array[:] = vnm1 + dt * ((1 - gamma) * anm1 + gamma * self._a_garb.x.array)
        self._a_n.x.array[:] = self._a_garb.x.array
        self._u_nm1.x.array[:] = un

    def initialStep(self,
                    t0,
                    callfirsts: List[Callable] = [],
                    callbacks: List[Callable] = [],
                    verbose: int = 0) -> None:

        # ## -------------------------------------------------
        #   --- first step: given u0 and v0, solve for a0 ---
        # ## -------------------------------------------------
        #
        if verbose >= 10:
            PETSc.Sys.Print('Solving the initial value step')  # type: ignore[attr-defined]
            PETSc.Sys.Print('Callfirsts...')  # type: ignore[attr-defined]

        for callfirst in callfirsts:
            callfirst(t0)  # <- update stuff

        # known: u0, v0. Solve for a0. u1 requires to solve a new system (loop)
        problem = fem.petsc.LinearProblem(self._m0_form, self._L0_form, bcs=self._bcs, u=self._a0,
                                          petsc_options=TimeScheme.petsc_options_t0,
                                          jit_options=self.jit_options)
        problem.solve()

        if verbose >= 10:
            PETSc.Sys.Print('Initial value problem solved, entering loop')  # type: ignore[attr-defined]
        # no callback because u1 is not solved yet
        #
        # ## -------------------------------------------------


class HilberHughesTaylor(GalphaNewmarkBeta):
    """
    Implementation of the *Hilber-Hughes-Taylor* or *alpha-Newmark* time-stepping scheme.
    *HHT* is a special case of the *Generalized-alpha* scheme (:math:`\\alpha_m=0` and :math:`\\alpha_f=\\alpha`).

    **(!)** The definition of the :math:`\\alpha` parameter is different
    from that in the original article. Here: :math:`\\alpha = -\\alpha_{HHT}`

    Implicit / explicit?
        Implicit

    Accuracy:
        First or second order, depending on parameters

    Args:
        *args: See GalphaNewmarkBeta

    Keyword Args:
        rho_inf (default = 0.9): Spectral radius in the high frequency limit. Bounds: (1/2,1).
            Setting **rho_inf** is the preferred way of defining the scheme.

        alpha (default = (1-rho_inf)/(1+rho_inf)): Set to e.g. 0.05 or sqrt(2) for low or moderate dissipation

        gamma (default = 1/2 + alpha): Default value ensures second-order accuracy.
            Other values give first-order accuracy

        beta (default = 1/4*(gamma+1/2)**2): Unconditionnal stability if :math:`\\beta \geq 0.25 + 0.5 \\alpha`

    Reference:
        H. M. Hilber, T. J. R. Hughes, and R. L. Taylor,
        "Improved Numerical Dissipation for Time Integration Algorithms in Structural Dynamics",
        Earthquake Engineering and Structural Dynamics, vol. 5, pp. 283–292, 1977.
    """
    labels = ['hilber-hughes-taylor', 'hht', 'hht-alpha']

    def __init__(self, *args, **kwargs):

        rho_inf = kwargs.pop('rho_inf', 0.9)
        alpha_f = (1 - rho_inf) / (1 + rho_inf)

        alpha = kwargs.get('alpha', alpha_f)
        kwargs['alpha_m'] = 0
        kwargs['alpha_f'] = alpha
        super().__init__(*args, **kwargs)


class NewmarkBeta(GalphaNewmarkBeta):
    """
    .. role:: python(code)
      :language: python

    Implementation of the *Newmark* or *Newmark-beta* time-stepping scheme, for :math:`\\beta>0`.
    The special case :math:`\\beta=0` is implemented in the :python:`LeapFrog` class.
    *Newmark-beta* is a special case of the *Generalized-alpha* scheme (:math:`\\alpha_m=\\alpha_f=0`).

    Implicit / explicit?
        Implicit

    Accuracy:
        First-order unless :math:`gamma=1/2` (second-order)

    Args:
        *args: See GalphaNewmarkBeta

    Keyword Args:
        gamma (default = 1/2): Default value ensures second-order accuracy.
            Other values give first-order accuracy

        beta (default = 1/4*(gamma+1/2)**2): Unconditionnal stability if :math:`\\beta \geq 0.25`

    See:
        https://en.wikipedia.org/wiki/Newmark-beta_method
    """
    labels = ['newmark', 'newmark-beta']

    def __init__(self, *args, **kwargs):
        kwargs['alpha_m'] = 0
        kwargs['alpha_f'] = 0
        super().__init__(*args, **kwargs)


class MidPoint(NewmarkBeta):
    """
    Implementation of the *Midpoint* time-stepping scheme, or *Average Acceleration Method*.
    *Midpoint* is a special case of *Newmark-beta* methods with :math:`\\beta=1/4` and :math:`\gamma=1/2`
    and is unconditionally stable.

    Implicit / explicit?
        Implicit

    Accuracy:
        Second-order

    Args:
        *args: See GalphaNewmarkBeta

    See:
        https://en.wikipedia.org/wiki/Midpoint_method
    """
    labels = ['midpoint', 'average-acceleration-method', 'aam']

    def __init__(self, *args, **kwargs):
        kwargs['gamma'] = 1 / 2
        kwargs['beta'] = 1 / 4
        super().__init__(*args, **kwargs)


class LinearAccelerationMethod(NewmarkBeta):
    """
    Implementation *Linear Acceleration Method* time-stepping scheme.
    It is a special case of *Newmark-beta* methods with :math:`\\beta=1/6` and :math:`\gamma=1/2`
    and is unconditionally stable.

    Implicit / explicit?
        Implicit

    Accuracy:
        Second-order

    Args:
        *args: See GalphaNewmarkBeta
    """
    labels = ['linear-acceleration-method', 'lam']

    def __init__(self, *args, **kwargs):
        kwargs['gamma'] = 1 / 2
        kwargs['beta'] = 1 / 6
        super().__init__(*args, **kwargs)
