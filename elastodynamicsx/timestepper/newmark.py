from dolfinx import fem
from petsc4py import PETSc
import ufl

from .timestepper import TimeStepper


class GalphaNewmarkBeta(TimeStepper):
    """
    Implementation of the 'g-a-newmark' (Generalized-alpha Newmark) time-stepping scheme, for beta>0. The special case beta=0 is implemented in the LeapFrog class.
    implementation adapted from: https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
    
    /!\ DOES NOT WORK YET IN THE GENERAL CASE, FOR alpha_m, alpha_f > 0 /!\

    implicit/explicit? implicit
    accuracy: second-order
    """
    def __init__(self, m_, k_, L, dt, function_space, bcs=[], **kwargs):
        """
        m_: function(u,v) that returns the ufl expression of the bilinear form with second derivative on time
               -> usually: m_ = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
        k_: function(u,v) that returns the ufl expression of the bilinear form with no derivative on time
               -> used to build the stiffness matrix
               -> usually: k_ = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L:    linear form
        dt: time step
        function_space: the Finite Element functionnal space
        bcs: the set of boundary conditions
        
        **kwargs: important optional parameters are 'beta' and 'gamma'. Default values give the MipPoint scheme.
            gamma: (default = 1/2)
            beta:  (default = 1/4*(gamma+1/2)**2)
        """
        c_ = lambda u,v: 0 #TODO
        
        alpha_m = kwargs.get('alpha_m', 0)
        alpha_f = kwargs.get('alpha_f', 0)
        gamma   = kwargs.get('gamma', 1/2 - alpha_m + alpha_f)
        beta    = kwargs.get('beta', 1/4*(gamma+1/2)**2)
        
        self.alpha_m, self.alpha_f, self.gamma, self.beta = alpha_m, alpha_f, gamma, beta
        
        const = lambda c: fem.Constant(function_space.mesh, PETSc.ScalarType(c))
        c1, c2, c3 = const(dt*gamma*(1-alpha_f)/beta), const(dt**2*(1 - (gamma-alpha_f)/beta)), const(dt**3*(1-alpha_f)*(1-gamma/2/beta))
        m1, m2, m3 = const((1-alpha_m)/beta),          const(dt*(1-alpha_m)/beta),              const(dt**2*(1 - (1-alpha_m)/2/beta))
        dt_        = const(dt)
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self.u_n     = fem.Function(function_space, name="u") #u(t)
        self.v_n     = fem.Function(function_space, name="v") #u'(t)  = du/dt
        self.a_n     = fem.Function(function_space, name="a") #u''(t) = d(du/dt)/dt
        #
        self.__a_nm1 = self.a_n #at the time of solving, a_n relates to step n-1. But at the time of callbacks or end of loop, the update has been done and relates to step n (same step as u_n)
        self.__v_nm1 = self.v_n #same convention as for a_n
        self.__u_nm1 = fem.Function(function_space, name="u_nm1") #u(t-dt)
        self.__a_garb= fem.Function(function_space, name="garbage") #garbage

        #
        self.__a = m1*m_(u,v) + c1*c_(u,v) + dt_*dt_*const(1-alpha_f)*k_(u,v)
        self.__L = dt_*dt_*L(v) - const(alpha_f)*k_(self.__u_nm1, v) \
                   + c1*c_(self.__u_nm1, v) + c2*c_(self.__v_nm1, v) - c3*c_(self.__a_nm1, v) \
                   + m1*m_(self.__u_nm1, v) + m2*m_(self.__v_nm1, v) - m3*m_(self.__a_nm1, v)
        
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
        self.__u_nm1.x.array[:] = u.x.array
        #self.u_n.x.array[:]     = u.x.array #not necessary
        self.v_n.x.array[:]     = du.x.array
        self.a_n.x.array[:]     = du.x.array #faux, TODO

    def prepareNextIteration(self):
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        dt = self.dt
        #
        self.__a_garb.x.array[:]= (self.u_n.x.array - self.__u_nm1.x.array - dt*self.__v_nm1.x.array)/self.beta/dt**2 - (1-2*self.beta)/2/self.beta*self.__a_nm1.x.array
        self.v_n.x.array[:]     = self.__v_nm1.x.array + dt*((1-self.gamma)*self.__a_nm1.x.array + self.gamma*self.__a_garb.x.array)
        self.a_n.x.array[:]     = self.__a_garb.x.array
        self.__u_nm1.x.array[:] = self.u_n.x.array

class NewmarkBeta(GalphaNewmarkBeta):
    """
    Implementation of the 'newmark' or 'newmark-beta' time-stepping scheme, for beta>0. The special case beta=0 is implemented in the LeapFrog class.
    see: https://en.wikipedia.org/wiki/Newmark-beta_method

    implicit/explicit? implicit
    accuracy: second-order
    """
    def __init__(self, *args, **kwargs):
        kwargs['alpha_m'] = 0
        kwargs['alpha_f'] = 0
        super().__init__(*args, **kwargs)

class MidPoint(NewmarkBeta):
    """
    Implementation of the 'midpoint' time-stepping scheme, or Average constant acceleration.
    Midpoint is a special case of Newmark-beta methods with beta=0.25 and gamma=0.5 and is unconditionally stable.
    see: https://en.wikipedia.org/wiki/Midpoint_method

    implicit/explicit? implicit
    accuracy: second-order
    """
    def __init__(self, *args, **kwargs):
        kwargs['gamma'] = 1/2
        kwargs['beta']  = 1/4
        super().__init__(*args, **kwargs)

