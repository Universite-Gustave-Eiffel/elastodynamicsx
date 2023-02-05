from dolfinx import fem
from petsc4py import PETSc
import ufl

from .timestepper import TimeStepper, OneStepTimeStepper
from elastodynamicsx.pde import BoundaryCondition


class LeapFrog(OneStepTimeStepper):
    """
    Implementation of the 'leapfrog' time-stepping scheme, or Explicit central difference scheme.
    Leapfrog is a special case of Newmark-beta methods with beta=0 and gamma=0.5
    see: https://en.wikipedia.org/wiki/Leapfrog_integration

    implicit/explicit? explicit
    accuracy: second-order
    """
    
    labels = ['leapfrog', 'central-difference']
    
    def __init__(self, function_space, m_, c_, k_, L, dt, bcs=[], **kwargs):
        """
        Args:
            function_space: The Finite Element functionnal space
            m_: Function(u,v) that returns the ufl expression of the bilinear form
                with second derivative on time
                -> usually: m_ = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
            c_: (optional, ignored if None) Function(u,v) that returns the ufl expression
                of the bilinear form with a first derivative on time (damping form)
                -> for Rayleigh damping: c_ = lambda u,v: eta_m * m_(u,v) + eta_k * k_(u,v)
            k_: Function(u,v) that returns the ufl expression of the bilinear form
                with no derivative on time
                -> used to build the stiffness matrix
                -> usually: k_ = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
            L:  Linear form
            dt: Time step
            bcs: The set of boundary conditions
        """
        dt_  = fem.Constant(function_space.mesh, PETSc.ScalarType(dt))
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        #
        self._u_n   = fem.Function(function_space, name="u") #u(t)
        self._u_nm1 = fem.Function(function_space)           #u(t-dt)
        self._u_nm2 = fem.Function(function_space)           #u(t-2*dt)
        #
        self._u0 = self._u_nm1
        self._v0 = self._u_n
        self._a0 = self._u_nm2

        #linear and bilinear forms for mass and stiffness matrices
        self._a = m_(u,v)
        self._L =-dt_*dt_*k_(self._u_nm1, v) + 2*m_(self._u_nm1,v) - m_(self._u_nm2,v)
        
        self._m0_form = m_(u,v)
        self._L0_form =-k_(self._u0, v)
        
        if not(L is None):
            self._L += dt_*dt_*L(v)
            self._L0_form += L(v)
        
        #linear and bilinear forms for damping matrix if given
        if not(c_ is None):
            self._a += 0.5*dt_*c_(u, v)
            self._L += 0.5*dt_*c_(self._u_nm2, v)
            self._L0_form -= c_(self._v0, v)
        
        #boundary conditions
        dirichletbcs = [bc for bc in bcs if issubclass(type(bc), fem.DirichletBCMetaClass)]
        supportedbcs = [bc for bc in bcs if type(bc) == BoundaryCondition]
        for bc in supportedbcs:
            if   bc.type == 'dirichlet':
                dirichletbcs.append(bc.bc)
            elif bc.type == 'neumann':
                self._L += dt_*dt_*bc.bc(v)
                self._L0_form += bc.bc(v)
            elif bc.type == 'robin':
                F_bc = dt_*dt_*bc.bc(u,v) #TODO: verifier
                self._a += ufl.lhs(F_bc)
                self._L += ufl.rhs(F_bc)
                self._L0_form += bc.bc(self._u0, v)
            elif bc.type == 'dashpot':
                F_bc = 0.5*dt_*bc.bc(u-self._u_nm2,v)
                self._a += ufl.lhs(F_bc)
                self._L += ufl.rhs(F_bc)
                self._L0_form += bc.bc(self._v0, v)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))
        
        #compile forms
        self.bilinear_form = fem.form(self._a)
        self.linear_form   = fem.form(self._L)
        #
        super().__init__(function_space, m_, c_, k_, L, dt, dirichletbcs, explicit=True, **kwargs)
        self._i0 = 1 #because solving the initial value problem a0=M_inv.(F(t0)-K(u0)-C.v0) also yields u1=u(t0+dt)
    
    def _prepareNextIteration(self):
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        self._u_nm2.x.array[:] = self._u_nm1.x.array
        self._u_nm1.x.array[:] = self._u_n.x.array
        
    def _initialStep(self, callfirsts, callbacks, verbose=0): #TODO: faux. corriger.
        """Specific to the initial value step"""
        
        ### -------------------------------------------------
        #   --- first step: given u0 and v0, solve for a0 ---
        ### -------------------------------------------------
        #
        if verbose >= 10: PETSc.Sys.Print('Solving the initial value step')
        if verbose >= 10: PETSc.Sys.Print('Callfirsts...')
        for callfirst in callfirsts: callfirst(self.t - 0*self._intermediate_dt, self) #<- update stuff #F_body.interpolate(F_body_function(t))
        
        problem = fem.petsc.LinearProblem(self._m0_form, self._L0_form, bcs=self._bcs, u=self._a0, petsc_options=TimeStepper.petsc_options_t0)
        problem.solve() #known: u0, v0. Solve for a0. u1 requires to solve a new system (loop)

        u0, v0, a0 = self._u0.x.array, self._v0.x.array, self._a0.x.array
        self._u_n.x.array[:], self._u_nm2.x.array[:] = u0 + self.dt*v0 + 1/2*self.dt**2*a0, u0 - self.dt*v0 + 1/2*self.dt**2*a0 #remind that self._u_nm1 = self._u0 -> already at the correct value
        #at this point: self._u_n = u1, self._u_nm1 = u0, self._u_nm2 = u-1
        
        if verbose >= 10: PETSc.Sys.Print('Initial value problem solved, entering loop')
        for callback in callbacks: callback(0, self) #<- store solution, plot, print, ...
        #
        ### -------------------------------------------------

    @property
    def u_nm1(self):
        return self._u_nm1

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

