from dolfinx import fem
from petsc4py import PETSc
import ufl

from . import FEniCSxTimeScheme
from elastodynamicsx.solvers import TimeStepper, NonlinearTimeStepper, OneStepTimeStepper
from elastodynamicsx.pde import PDE, BoundaryCondition

from typing import Union, Callable


class GalphaNewmarkBeta(FEniCSxTimeScheme):
    """
    Implementation of the 'g-a-newmark' (Generalized-alpha Newmark) time-stepping scheme,
    for beta>0. The special case beta=0 is implemented in the LeapFrog class.

    Reference:
        J. Chung and G. M. Hulbert, "A time integration algorithm for structural
        dynamics with improved numerical dissipation: The generalized-α method,"
        J. Appl. Mech. 60(2), 371–375 (1993).

    Adapted from:
        https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html

    implicit/explicit? implicit
    accuracy: first or second order, depending on parameters
    """
    labels = ['g-a-newmark', 'generalized-alpha']


    def build_timestepper(*args, **kwargs) -> 'TimeStepper':
        tscheme = GalphaNewmarkBeta(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm
        if kwargs.pop('linear', True):
            return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)
        else:
            return NonlinearTimeStepper(comm, tscheme, **kwargs)


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

        kwargs: The four parameters ('alpha_m', 'alpha_m', 'gamma', 'beta')
                can be set manually, although the preferred way is by setting
                the desired spectral radius 'rho_inf'.
            rho_inf: spectral radius in the high frequency limit. bounds: (1/2,1)
                default = 0.75
            alpha_m: unconditionnal stability if -1 <= alpha_m <= alpha_f <= 0.5
                default = (2*rho_inf-1)/(rho_inf+1)
            alpha_f: unconditionnal stability if -1 <= alpha_m <= alpha_f <= 0.5
                default = rho_inf/(rho_inf+1)
            gamma: default value ensures second-order accuracy
                other values give first-order accuracy
                default = 1/2 - alpha_m + alpha_f
            beta: unconditionnal stability if beta >= 0.25 + 0.5*(alpha_f-alpha_m)
                default = 1/4*(gamma+1/2)**2

            jit_options: (default=PDE.default_jit_options) options for the just-in-time compiler
        """
        self.jit_options = kwargs.get('jit_options', PDE.default_jit_options)
        rho_inf = kwargs.get('rho_inf', 0.75)
        alpha_m = (2*rho_inf-1)/(rho_inf+1)
        alpha_f = rho_inf/(rho_inf+1)
        alpha_m = kwargs.get('alpha_m', alpha_m)
        alpha_f = kwargs.get('alpha_f', alpha_f)
        gamma   = kwargs.get('gamma', 1/2 - alpha_m + alpha_f)
        beta    = kwargs.get('beta', 1/4*(gamma+1/2)**2)

        self.alpha_m, self.alpha_f, self.gamma, self.beta = alpha_m, alpha_f, gamma, beta

        const = lambda c: fem.Constant(function_space.mesh, PETSc.ScalarType(c))
        c1 = const(dt*gamma*(1-alpha_f)/beta)
        c2 = const(dt**2*(1 - (gamma-alpha_f)/beta))
        c3 = const(dt**3*(1-alpha_f)*(1-gamma/2/beta))

        m1 = const((1-alpha_m)/beta)
        m2 = const(dt*(1-alpha_m)/beta)
        m3 = const(dt**2*(1 - (1-alpha_m)/2/beta))

        dt_        = const(dt)
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self._u_n     = fem.Function(function_space, name="u")  # u(t)
        self._v_n     = fem.Function(function_space, name="v")  # u'(t)  = du/dt
        self._a_n     = fem.Function(function_space, name="a")  # u''(t) = d(du/dt)/dt
        #
        # At the time of solving, a_n relates to step n-1.
        # But at the time of callbacks or end of loop, the update has been done
        # and relates to step n (same step as u_n)
        self._a_nm1 = self._a_n
        self._v_nm1 = self._v_n  # same convention as for a_n
        self._u_nm1 = fem.Function(function_space, name="u_nm1")  # u(t-dt)
        self._a_garb= fem.Function(function_space, name="garbage")  # Garbage
        #
        self._u0 = self._u_nm1
        self._v0 = self._v_nm1
        self._a0 = self._a_nm1

        # linear and bilinear forms for mass and stiffness matrices
        self._a = m1*m_(u,v) + dt_*dt_*const(1-alpha_f)*k_(u,v)
        self._L = - const(dt*dt*alpha_f)*k_(self._u_nm1, v) \
                  + m1*m_(self._u_nm1, v) + m2*m_(self._v_nm1, v) - m3*m_(self._a_nm1, v)

        self._m0_form = m_(u,v)
        self._L0_form =-k_(self._u0, v)

        if not(L is None):
            self._L += dt_*dt_*L(v)
            self._L0_form += L(v)

        # linear and bilinear forms for damping matrix if given
        if not(c_ is None):
            self._a += c1*c_(u,v)
            self._L += c1*c_(self._u_nm1, v) + c2*c_(self._v_nm1, v) - c3*c_(self._a_nm1, v)
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
                d1, d2, d3 = dt * gamma/beta, dt**2 * (1 - gamma/beta), dt**3 * (1 - gamma/beta/2)
                #
                F_bc = bc.bc(d1*u - d1*self._u_nm1 + d2*self._v_nm1 + d3*self._a_nm1, v)
                self._a += ufl.lhs(F_bc)
                self._L += ufl.rhs(F_bc)
                self._L0_form += bc.bc(self._v0, v)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))

        # compile forms
        bilinear_form = fem.form(self._a, jit_options=self.jit_options)
        linear_form   = fem.form(self._L, jit_options=self.jit_options)
        #
        super().__init__(dt, self._u_n, bilinear_form, linear_form, mpc, dirichletbcs,
                         explicit=False, intermediate_dt=self.alpha_f, **kwargs)


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
        un, unm1    = self._u_n.x.array, self._u_nm1.x.array
        vnm1, anm1  = self._v_nm1.x.array, self._a_nm1.x.array

        self._a_garb.x.array[:]= (un - unm1 - dt*vnm1)/beta/dt**2 - (1-2*beta)/2/beta*anm1
        self._v_n.x.array[:]   = vnm1 + dt*((1-gamma)*anm1 + gamma*self._a_garb.x.array)
        self._a_n.x.array[:]   = self._a_garb.x.array
        self._u_nm1.x.array[:] = un


    def initialStep(self, t0, callfirsts:list=[], callbacks:list=[], verbose=0) -> None:
        """Specific to the initial value step"""
        ### -------------------------------------------------
        #   --- first step: given u0 and v0, solve for a0 ---
        ### -------------------------------------------------
        #
        if verbose >= 10:
            PETSc.Sys.Print('Solving the initial value step')
            PETSc.Sys.Print('Callfirsts...')

        for callfirst in callfirsts:
            callfirst(t0)  # <- update stuff

        # known: u0, v0. Solve for a0. u1 requires to solve a new system (loop)
        problem = fem.petsc.LinearProblem(self._m0_form, self._L0_form, bcs=self._bcs, u=self._a0,
                                          petsc_options=TimeStepper.petsc_options_t0,
                                          jit_options=self.jit_options)
        problem.solve()

        if verbose >= 10:
            PETSc.Sys.Print('Initial value problem solved, entering loop')
        # no callback because u1 is not solved yet
        #
        # ## -------------------------------------------------


class HilberHughesTaylor(GalphaNewmarkBeta):
    """
    Implementation of the 'hilber-hughes-taylor' or 'alpha-newmark' time-stepping scheme.

    Reference:
        H. M. Hilber, T. J. R. Hughes, and R. L. Taylor,
        "Improved Numerical Dissipation for Time Integration Algorithms in Structural Dynamics",
        Earthquake Engineering and Structural Dynamics, vol. 5, pp. 283–292, 1977.

    implicit/explicit? implicit
    accuracy: first or second order, depending on parameters

    /!\ the definition of the alpha parameter is different from that in the original article. Here: alpha = -alpha_HHT
    """
    labels = ['hilber-hughes-taylor', 'hht', 'hht-alpha']


    def build_timestepper(*args, **kwargs) -> 'TimeStepper':
        tscheme = HilberHughesTaylor(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm
        if kwargs.pop('linear', True):
            return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)
        else:
            return NonlinearTimeStepper(comm, tscheme, **kwargs)


    def __init__(self, *args, **kwargs):
        """
        *args: See GalphaNewmarkBeta
        **kwargs: The three parameters ('alpha', 'gamma', 'beta') can be set manually,
            although the preferred way is by setting the desired spectral radius 'rho_inf'.
            rho_inf: spectral radius in the high frequency limit. bounds: (1/2,1)
                default = 0.9
            alpha: set to e.g. 0.05 // sqrt(2) for low // moderate dissipation
                default = (1-rho_inf)/(1+rho_inf)
            gamma: default value ensures second-order accuracy; other values give first-order accuracy
                default = 1/2 + alpha
            beta: unconditionnal stability if beta >= 0.25 + 0.5*alpha
                default = 1/4*(gamma+1/2)**2
        """
        rho_inf = kwargs.pop('rho_inf', 0.9)
        alpha_m = 0
        alpha_f = (1-rho_inf)/(1+rho_inf)

        alpha             = kwargs.get('alpha', alpha_f)
        kwargs['alpha_m'] = 0
        kwargs['alpha_f'] = alpha
        super().__init__(*args, **kwargs)


class NewmarkBeta(GalphaNewmarkBeta):
    """
    Implementation of the 'newmark' or 'newmark-beta' time-stepping scheme, for beta>0.
    The special case beta=0 is implemented in the LeapFrog class.

    See:
        https://en.wikipedia.org/wiki/Newmark-beta_method

    implicit/explicit? implicit
    accuracy: first-order unless gamma=1/2 (second-order)
    """
    labels = ['newmark', 'newmark-beta']


    def build_timestepper(*args, **kwargs) -> 'TimeStepper':
        tscheme = NewmarkBeta(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm
        if kwargs.pop('linear', True):
            return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)
        else:
            return NonlinearTimeStepper(comm, tscheme, **kwargs)


    def __init__(self, *args, **kwargs):
        kwargs['alpha_m'] = 0
        kwargs['alpha_f'] = 0
        super().__init__(*args, **kwargs)


class MidPoint(NewmarkBeta):
    """
    Implementation of the 'midpoint' time-stepping scheme, or Average Acceleration Method.
    Midpoint is a special case of Newmark-beta methods with beta=1/4 and gamma=1/2
    and is unconditionally stable.

    See:
        https://en.wikipedia.org/wiki/Midpoint_method

    implicit/explicit? implicit
    accuracy: second-order
    """
    labels = ['midpoint', 'average-acceleration-method', 'aam']


    def build_timestepper(*args, **kwargs) -> 'TimeStepper':
        tscheme = MidPoint(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm
        if kwargs.pop('linear', True):
            return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)
        else:
            return NonlinearTimeStepper(comm, tscheme, **kwargs)


    def __init__(self, *args, **kwargs):
        kwargs['gamma'] = 1/2
        kwargs['beta']  = 1/4
        super().__init__(*args, **kwargs)


class LinearAccelerationMethod(NewmarkBeta):
    """
    Implementation 'linear-acceleration-method' time-stepping scheme.
    It is a special case of Newmark-beta methods with beta=1/6 and gamma=1/2
    and is unconditionally stable.

    implicit/explicit? implicit
    accuracy: second-order
    """
    labels = ['linear-acceleration-method', 'lam']


    def build_timestepper(*args, **kwargs) -> 'TimeStepper':
        tscheme = LinearAccelerationMethod(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm
        if kwargs.pop('linear', True):
            return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)
        else:
            return NonlinearTimeStepper(comm, tscheme, **kwargs)


    def __init__(self, *args, **kwargs):
        kwargs['gamma'] = 1/2
        kwargs['beta']  = 1/6
        super().__init__(*args, **kwargs)
