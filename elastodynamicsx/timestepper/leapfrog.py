from dolfinx import fem
from petsc4py import PETSc
import ufl

from .timestepper import OneStepTimeStepper


class LeapFrog(OneStepTimeStepper):
    """
    Implementation of the 'leapfrog' time-stepping scheme, or Explicit central difference scheme.
    Leapfrog is a special case of Newmark-beta methods with beta=0 and gamma=0.5
    see: https://en.wikipedia.org/wiki/Leapfrog_integration

    implicit/explicit? explicit
    accuracy: second-order
    """
    
    labels = ['leapfrog', 'central-difference']
    
    def __init__(self, m_, c_, k_, L, dt, function_space, bcs=[], **kwargs):
        """
        m_: function(u,v) that returns the ufl expression of the bilinear form with second derivative on time
               -> usually: m_ = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
        c_: (optional, ignored if None) function(u,v) that returns the ufl expression of the bilinear form with a first derivative on time (damping form)
               -> for Rayleigh damping: c_ = lambda u,v: eta_m * m_(u,v) + eta_k * k_(u,v)
        k_: function(u,v) that returns the ufl expression of the bilinear form with no derivative on time
               -> used to build the stiffness matrix
               -> usually: k_ = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L:  linear form
        dt: time step
        function_space: the Finite Element functionnal space
        bcs: the set of boundary conditions
        """
        dt_  = fem.Constant(function_space.mesh, PETSc.ScalarType(dt))
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self._u_n   = fem.Function(function_space, name="u") #u(t)
        self._u_nm1 = fem.Function(function_space)           #u(t-dt)
        self._u_nm2 = fem.Function(function_space)           #u(t-2*dt)
        #
        self._a = m_(u,v)
        self._L = dt_*dt_*L(v) - dt_*dt_*k_(self._u_nm1, v) + 2*m_(self._u_nm1,v) - m_(self._u_nm2,v)
        
        if not(c_ is None):
            self._a += 0.5*dt_*c_(u,v)
            self._L += 0.5*dt_*c_(self._u_nm2,v)
        
        self.bilinear_form = fem.form(self._a)
        self.linear_form   = fem.form(self._L)
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
        self._t = t0
        self._u_nm2.x.array[:] = u.x.array #u_nm2 = u(0)
        self._u_nm1.x.array[:] = u.x.array #faux, u_nm1 = u(0) + dt*u'(0)
        self._u_n.x.array[:] = u.x.array #faux
    
    def _prepareNextIteration(self):
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        self._u_nm2.x.array[:] = self._u_nm1.x.array
        self._u_nm1.x.array[:] = self._u_n.x.array

    @property
    def v(self): #warning: this is v(t-dt)
        v_ = fem.Function(self.u.function_space)
        v_.x.array[:] = 0.5/self.dt*(self._u_n.x.array - self._u_nm2.x.array)
        return v_

    @property
    def a(self): #warning: this is a(t-dt)
        a_ = fem.Function(self.u.function_space)
        a_.x.array[:] = 1/self.dt**2 *(self._u_n.x.array - 2*self._u_nm1.x.array + self._u_nm2.x.array)
        return a_

