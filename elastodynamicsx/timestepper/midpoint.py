from dolfinx import fem
from petsc4py import PETSc
import ufl

from .timestepper import TimeStepper


class MidPoint_old(TimeStepper):
    """
    Implementation of the 'midpoint' time-stepping scheme, or Average constant acceleration.
    Midpoint is a special case of Newmark-beta methods with beta=0.25 and gamma=0.5 and is unconditionally stable.
    see: https://en.wikipedia.org/wiki/Midpoint_method

    implicit/explicit? implicit
    accuracy: second-order
    """
    def __init__(self, a_tt, a_xx, L, dt, function_space, bcs=[], **kwargs):
        """
        a_tt: function(u,v) that returns the ufl expression of the bilinear form with second derivative on time
                 -> usually: a_tt = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
        a_xx: function(u,v) that returns the ufl expression of the bilinear form with no derivative on time
                 -> used to build the stiffness matrix
                 -> usually: a_xx = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L:    linear form
        dt: time step
        function_space: the Finite Element functionnal space
        bcs: the set of boundary conditions
        """
        dt_  = fem.Constant(function_space.mesh, PETSc.ScalarType(dt))
        four = fem.Constant(function_space.mesh, PETSc.ScalarType(4))
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self.u_n     = fem.Function(function_space, name="u") #u(t)
        self.du_n    = fem.Function(function_space, name="v") #u'(t)  = du/dt
        self.ddu_n   = fem.Function(function_space, name="a") #u''(t) = d(du/dt)/dt   = 4/dt**2 * (u(t) - u_bar(t-dt))
        self.ddu_nm1 = fem.Function(function_space) #u''(t-dt)
        self.u_n_bar = fem.Function(function_space) #u_bar(t) = u + dt*u' + dt**2/4 * u''

        #
        self.__a = four*a_tt(u,v) + dt_*dt_*a_xx(u, v)
        self.__L = dt_*dt_*L(v) + four*a_tt(self.u_n_bar,v)
        self.bilinear_form = fem.form(self.__a)
        self.linear_form   = fem.form(self.__L)
        #
        super().__init__(dt, bcs, **kwargs)

    def initial_condition(self, u, du, t0=0):
        ###
        """
        Apply initial conditions
        
        u: u at t0
        du: du/dt at t0
        t0: start time (default: 0)
        """
        self.t_n = t0
        self.u_n_bar.x.array[:] = u.x.array #faux
        self.ddu_nm1.x.array[:] = u.x.array #faux
        self.ddu_n.x.array[:] = u.x.array #faux
        self.du_n.x.array[:] = u.x.array #faux
        self.u_n.x.array[:] = u.x.array #faux

    def prepareNextIteration(self):
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        dt = self.dt
        self.ddu_nm1.x.array[:] = self.ddu_n.x.array
        self.ddu_n.x.array[:]   = 4/(dt*dt)*(self.u_n.x.array - self.u_n_bar.x.array)
        self.du_n.x.array[:]    = self.du_n.x.array + 0.5*dt*(self.ddu_n.x.array + self.ddu_nm1.x.array)
        self.u_n_bar.x.array[:] = self.u_n.x.array  + dt*self.du_n.x.array + 0.25*dt*dt*self.ddu_n.x.array

