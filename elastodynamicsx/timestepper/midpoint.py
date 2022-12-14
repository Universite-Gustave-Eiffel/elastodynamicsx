from dolfinx import fem
import ufl

from .timestepper import TimeStepper


class MidPoint(TimeStepper):
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
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self.u_n     = fem.Function(function_space, name="u") #u(t)
        self.du_n    = fem.Function(function_space, name="v") #u'(t)  = du/dt
        self.ddu_n   = fem.Function(function_space, name="a") #u''(t) = d(du/dt)/dt   = 4/dt**2 * (u(t) - u_bar(t-dt))
        self.ddu_nm1 = fem.Function(function_space) #u''(t-dt)
        self.u_n_bar = fem.Function(function_space) #u_bar(t) = u + dt*u' + dt**2/4 * u''

        #
        self.__a = 4*a_tt(u,v) + dt*dt*a_xx(u, v)
        self.__L = dt*dt*L(v) + 4*a_tt(self.u_n_bar,v)
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
        X_ = self.u_n.function_space.element.interpolation_points #version 0.5 -> X_ = V.element.interpolation_points()
        self.ddu_nm1.x.array[:] = self.ddu_n.x.array
        self.ddu_n.interpolate( fem.Expression(4/(self.dt*self.dt)*(self.u_n - self.u_n_bar), X_) )
        self.du_n.interpolate( fem.Expression(self.du_n + 0.5*self.dt*(self.ddu_n + self.ddu_nm1), X_) )
        self.u_n_bar.interpolate( fem.Expression(self.u_n + self.dt*self.du_n + 0.25*self.dt*self.dt*self.ddu_n, X_) )

