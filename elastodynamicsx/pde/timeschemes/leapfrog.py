from dolfinx import fem
from petsc4py import PETSc
import ufl

from . import FEniCSxTimeScheme
from elastodynamicsx.solvers import TimeStepper, OneStepTimeStepper
from elastodynamicsx.pde import PDE, BoundaryCondition

from typing import Union, Callable


class LeapFrog(FEniCSxTimeScheme):
    """
    Implementation of the 'leapfrog' time-stepping scheme, or Explicit central difference scheme.
    Leapfrog is a special case of Newmark-beta methods with beta=0 and gamma=0.5
    see: https://en.wikipedia.org/wiki/Leapfrog_integration

    implicit/explicit? explicit
    accuracy: second-order

    a_n = (u_n+1 - 2u_n + u_n-1)/dt**2
    v_n = (u_n+1 - u_n-1)/(2*dt)
    """
    labels = ['leapfrog', 'central-difference']


    def build_timestepper(*args, **kwargs) -> 'TimeStepper':
        tscheme = LeapFrog(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm
        return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)


    def __init__(self, function_space:fem.FunctionSpace,
                 m_:Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form],
                 c_:Union[None, Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form]],
                 k_:Callable[['ufl.TrialFunction', 'ufl.TestFunction'], ufl.form.Form],
                 L:Union[None, Callable[['ufl.TestFunction'], ufl.form.Form]],
                 dt, bcs:list=[], **kwargs):
        """
        Args:
            function_space: The Finite Element functionnal space
            m_: The mass form
                -> usually: m_ = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
            c_: (optional, ignored if None) The damping form
                -> e.g. for Rayleigh damping: c_ = lambda u,v: eta_m * m_(u,v) + eta_k * k_(u,v)
            k_: The stiffness form
                -> usually: k_ = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
            L:  Linear form
            dt: Time step
            bcs: The set of boundary conditions
        kwargs:
            jit_options: (default=PDE.default_jit_options) options for the just-in-time compiler
        """
        self.jit_options = kwargs.get('jit_options', PDE.default_jit_options)
        dt_  = fem.Constant(function_space.mesh, PETSc.ScalarType(dt))

        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)

        self._u_n   = fem.Function(function_space, name="u")  # u(t)
        self._u_nm1 = fem.Function(function_space)            # u(t-dt)
        self._u_nm2 = fem.Function(function_space)            # u(t-2*dt)
        #
        self._u0 = self._u_nm1
        self._v0 = self._u_n
        self._a0 = self._u_nm2

        # linear and bilinear forms for mass and stiffness matrices
        self._a = m_(u,v)
        self._L =-dt_*dt_*k_(self._u_nm1, v) + 2*m_(self._u_nm1,v) - m_(self._u_nm2,v)

        self._m0_form = m_(u,v)
        self._L0_form =-k_(self._u0, v)

        if not(L is None):
            self._L += dt_*dt_*L(v)
            self._L0_form += L(v)

        # linear and bilinear forms for damping matrix if given
        if not(c_ is None):
            self._a += 0.5*dt_*c_(u, v)
            self._L += 0.5*dt_*c_(self._u_nm2, v)
            self._L0_form -= c_(self._v0, v)

        # boundary conditions
        mpc          = PDE.build_mpc(function_space, bcs)
        dirichletbcs = BoundaryCondition.get_dirichlet_BCs(bcs)
        supportedbcs = BoundaryCondition.get_weak_BCs(bcs)
        for bc in supportedbcs:
            if   bc.type == 'neumann':
                self._L += dt_*dt_*bc.bc(v)
                self._L0_form += bc.bc(v)
            elif bc.type == 'robin':
                F_bc = dt_*dt_*bc.bc(u,v)
                self._a += ufl.lhs(F_bc)
                self._L += ufl.rhs(F_bc)
                self._L0_form += bc.bc(self._u0, v)
            elif bc.type == 'dashpot':
                F_bc = 0.5*dt_*bc.bc(u-self._u_nm2, v)
                self._a += ufl.lhs(F_bc)
                self._L += ufl.rhs(F_bc)
                self._L0_form += bc.bc(self._v0, v)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))

        # compile forms
        bilinear_form = fem.form(self._a, jit_options=self.jit_options)
        linear_form   = fem.form(self._L, jit_options=self.jit_options)
        #
        super().__init__(dt, self._u_n, bilinear_form, linear_form, mpc, dirichletbcs, explicit=True, **kwargs)


    @property
    def u(self) -> fem.Function:
        return self._u_n


    @property
    def u_nm1(self) -> fem.Function:
        return self._u_nm1


    def prepareNextIteration(self) -> None:
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        self._u_nm2.x.array[:] = self._u_nm1.x.array
        self._u_nm1.x.array[:] = self._u_n.x.array


    def initialStep(self, t0, callfirsts:list=[], callbacks:list=[], verbose=0) -> None:
        """Specific to the initial value step"""
        # ## -------------------------------------------------
        #    --- first step: given u0 and v0, solve for a0 ---
        # ## -------------------------------------------------
        #
        if verbose >= 10:
            PETSc.Sys.Print('Solving the initial value step')
            PETSc.Sys.Print('Callfirsts...')

        for callfirst in callfirsts:
            callfirst(t0)  # <- update stuff

        # Known: u0, v0.
        # Solve for a0.
        # u1 is directly obtained from u0, v0, a0 (explicit scheme)
        # u2 requires to solve a new system (enter the time loop)
        problem = fem.petsc.LinearProblem(self._m0_form, self._L0_form, bcs=self._bcs, u=self._a0,
                                          petsc_options=TimeStepper.petsc_options_t0,
                                          jit_options=self.jit_options)
        problem.solve()

        u0, v0, a0 = self._u0.x.array, self._v0.x.array, self._a0.x.array
        u1 = self._u_n.x.array

        # remember that self._u_nm1 = self._u0 -> self._u_nm1 is already at the correct value
        u1[:] = u0 + self.dt*v0 + 1/2*self.dt**2*a0
        # unm1 is not needed
        # at this point: self._u_n = u1, self._u_nm1 = u0

        self.prepareNextIteration()

        if verbose >= 10:
            PETSc.Sys.Print('Initial value problem solved, entering loop')
        for callback in callbacks:
            callback(0, self._u_n.vector)  # <- store solution, plot, print, ...
        #
        # ## -------------------------------------------------
