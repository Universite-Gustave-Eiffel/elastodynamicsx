#TODO: optimize PETSc default options

from dolfinx import fem
from petsc4py import PETSc
import ufl
import numpy as np

from elastodynamicsx.pde import BoundaryCondition

class FrequencyDomainSolver:
    """
    Class for solving frequency domain problems.
    
    Example of use:
        #imports
        from dolfinx import mesh, fem
        import ufl
        from mpi4py import MPI
        from elastodynamicsx.solvers import FrequencyDomainSolver
        from elastodynamicsx.pde import material, BoundaryCondition

        #domain
        length, height = 10, 10
        Nx, Ny = 10, 10
        domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [length,height]], [Nx,Ny])
        V      = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))

        #material
        rho, lambda_, mu = 1, 2, 1
        mat = material(V, rho, lambda_, mu)

        #absorbing boundary condition
        Z_N, Z_T = mat.Z_N, mat.Z_T #P and S mechanical impedances
        bcs = [ BoundaryCondition(V, 'Dashpot', (Z_N, Z_T)) ]
        
        #gaussian source term
        F0     = fem.Constant(domain, PETSc.ScalarType([1,0])) #polarization
        R0     = 0.1 #radius
        x0, y0 = length/2, height/2 #center
        x      = ufl.SpatialCoordinate(domain)
        gaussianBF = F0 * ufl.exp(-((x[0]-x0)**2+(x[1]-y0)**2)/2/R0**2) / (2*3.141596*R0**2)
        bf         = BodyForce(V, gaussianBF)
        
        #solve
        omega    = 1.0
        fdsolver = FrequencyDomainSolver(V, mat.m, mat.c, mat.k, bf.L, bcs=bcs, omega=omega)
        u        = fdsolver.solve()
    """
    
    def __init__(self, function_space, m_, c_, k_, L, bcs=[], **kwargs):
        """
        Args:
            function_space: The function space
            m_:  Function of u,v that returns the mass form
            c_:  Function of u,v that returns the damping form
            k_:  Function of u,v that returns the stiffness form
            L :  Function of v   that returns the linear form
            bcs: List of instances of the class BoundaryCondition
            kwargs:
                omega: (optional) a delfinx.fem.Constant to be pointed to.
                    By default creates its own instance. To get it: self.omega_dolfinx
        """
        #
        w = kwargs.get('omega', 1)
        if not isinstance(w, fem.Constant):
            w = fem.Constant(function_space.mesh, PETSc.ScalarType(w))
        self._w   = w
        
        #self._m_function = m_
        #self._c_function = c_
        #self._k_function = k_
        
        #
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        
        #linear and bilinear forms
        self._m = m_(u,v)
        self._c = c_(u, v) if not(c_ is None) else 0
        self._k = k_(u,v)
        self._L = L(v)     if not(L is None) else 0*ufl.conj(v)
        
        #boundary conditions
        dirichletbcs = [bc for bc in bcs if issubclass(type(bc), fem.DirichletBCMetaClass)]
        supportedbcs = [bc for bc in bcs if type(bc) == BoundaryCondition]
        for bc in supportedbcs:
            if   bc.type == 'dirichlet':
                dirichletbcs.append(bc.bc)
            elif bc.type == 'neumann':
                self._L += bc(v)
            elif bc.type == 'robin':
                F_bc = bc.bc(u,v)
                self._k += ufl.lhs(F_bc)
                self._L += ufl.rhs(F_bc)
            elif bc.type == 'dashpot':
                self._c += bc.bc(u,v)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))

        self._bcs = dirichletbcs
        self._problem = None
        
    
    
    def solve(self, omega=None): #TODO: solve loop for omega=[w1, w2, ...] -> optimize by pre-building M,C,K matrices -> A=-w*w*M + i*w*C + K
        """
        Assemble and solve the linear problem
        
        Args:
            omega: If given, update the value of the angular frequency
        
        Returns:
            u: The solution (displacement field)
        """
        return self._solve_single_omega(omega)

    def _solve_single_omega(self, omega=None):
        if not omega is None:
            self._w.value = omega
        
        w = self._w
        a = -w*w*self._m + 1J*w*self._c + self._k
        self._problem = fem.petsc.LinearProblem(a, self._L, bcs=self._bcs)
        return self._problem.solve()
    
    def _solve_multiple_omega(self, omega, callbacks=[]):
        raise NotImplementedError
    
    @property
    def omega(self):
        """The angular frequency"""
        return self._w.value
    
    @property
    def omega_dolfinx(self):
        return self._w
    
    @property
    def problem(self):
        """The fem.petsc.LinearProblem instance that has been previously solved"""
        return self._problem

