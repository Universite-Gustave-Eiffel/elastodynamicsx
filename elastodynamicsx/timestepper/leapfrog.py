from dolfinx import fem
from petsc4py import PETSc
import ufl

from .timestepper import TimeStepper


class LeapFrog(TimeStepper):
    """
    Implementation of the 'leapfrog' time-stepping scheme, or Explicit central difference scheme.
    Leapfrog is a special case of Newmark-beta methods with beta=0 and gamma=0.5
    see: https://en.wikipedia.org/wiki/Leapfrog_integration

    implicit/explicit? explicit
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
        two  = fem.Constant(function_space.mesh, PETSc.ScalarType(2))
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self.u_n     = fem.Function(function_space, name="u") #u(t)
        self.__u_nm1 = fem.Function(function_space)           #u(t-dt)
        self.__u_nm2 = fem.Function(function_space)           #u(t-2*dt)
        #
        self.__a = a_tt(u,v)
        self.__L = dt_*dt_*L(v) - dt_*dt_*a_xx(self.__u_nm1, v) + two*a_tt(self.__u_nm1,v) - a_tt(self.__u_nm2,v)
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
        self.__u_nm2.x.array[:] = u.x.array #u_nm2 = u(0)
        self.__u_nm1.x.array[:] = u.x.array #faux, u_nm1 = u(0) + dt*u'(0)
        self.u_n.x.array[:] = u.x.array #faux
    
    def prepareNextIteration(self):
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        self.__u_nm2.x.array[:] = self.__u_nm1.x.array
        self.__u_nm1.x.array[:] = self.u_n.x.array

