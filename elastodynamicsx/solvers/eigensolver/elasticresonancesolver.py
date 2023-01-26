#TODO: documentation

from dolfinx import fem
from petsc4py import PETSc
from slepc4py import SLEPc
import ufl
import numpy as np

from elastodynamicsx.pde import BoundaryCondition
from elastodynamicsx.solutions import ModalBasis

class ElasticResonanceSolver(SLEPc.EPS):
    """
    Convenience class inhereted from SLEPc.EPS, with default parameters and convenience methods that are relevant for computing the resonances of an elastic component.
    
    Example of use (free resonances of an elastic cube):
        
        from dolfinx import mesh, fem
        from mpi4py import MPI
        from elastodynamicsx.solvers import ElasticResonanceSolver
        from elastodynamicsx.pde import Material
        #
        domain = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [1., 1., 1.]], [10, 10, 10])
        V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        #
        rho, lambda_, mu = 1, 2, 1
        material = Material.build(V, rho, lambda_, mu)
        eps = ElasticResonanceSolver(material.m, material.k, V, bcs=[], nev=6+6) #the first 6 resonances are rigid body motion
        eps.solve()
        eps.plot()
        freqs = eps.getEigenfrequencies()
        print('First resonance frequencies:', freqs)
    """

    def __init__(self, m_, k_, function_space, bcs=[], **kwargs):
        """
        m_: function(u,v) that returns the ufl expression of the bilinear form with second derivative on time
               -> usually: m_ = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
        k_: function(u,v) that returns the ufl expression of the bilinear form with no derivative on time
               -> used to build the stiffness matrix
               -> usually: k_ = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        function_space: the Finite Element functionnal space
        bcs: the set of boundary conditions
        """
        #
        super().__init__()
        
        self.function_space = function_space
        
        ### Variational problem
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        m_form = m_(u,v)
        k_form = k_(u,v)
        #
        
        #boundary conditions
        dirichletbcs = [bc for bc in bcs if issubclass(type(bc), fem.DirichletBCMetaClass)]
        supportedbcs = [bc for bc in bcs if type(bc) == BoundaryCondition]
        for bc in supportedbcs:
            assert(bc.type in ('dirichlet', 'neumann', 'robin'), "unsupported boundary condition {0:s}".format(bc.type))
            if   bc.type == 'dirichlet':
                dirichletbcs.append(bc.bc)
            elif bc.type == 'neumann':
                pass
            elif bc.type == 'robin':
                F_bc    = bc.bc(u,v)
                k_form += ufl.lhs(F_bc)
        
        ### assemble mass M and stifness K matrices
        M = fem.petsc.assemble_matrix(fem.form(m_form), bcs=dirichletbcs)
        M.assemble()
        K = fem.petsc.assemble_matrix(fem.form(k_form), bcs=dirichletbcs)
        K.assemble()
        
        #
        self.create(function_space.mesh.comm)
        self.setOperators(K, M)
        self.setProblemType(SLEPc.EPS.ProblemType.GHEP) #GHEP = Generalized Hermitian Eigenvalue Problem
        #self.setTolerances(tol=1e-9)
        self.setType(SLEPc.EPS.Type.KRYLOVSCHUR) #note that Krylov-Schur is the default solver

        ### Spectral transform
        st = self.getST()
        st.setType(SLEPc.ST.Type.SINVERT) #SINVERT = Shift and invert. By default, Slepc computes the largest eigenvalue, while we are interested in the smallest ones
        st.setShift( 1e-8 ) #can be set to a different value if the focus is set on another part of the spectrum

        ### Number of eigenvalues to be computed
        nev = kwargs.get('nev', 10)
        self.setDimensions(nev=nev)

    def getWn(self):
        """The eigen angular frequencies from the computed eigenvalues"""
        return np.array([np.sqrt(abs(self.getEigenvalue(i).real)) for i in range(self._getNout())]) #abs because rigid body motions may lead to minus zero: -0.00000
        
    def getEigenfrequencies(self):
        """The eigenfrequencies from the computed eigenvalues"""
        return self.getWn()/(2*np.pi)

    def getEigenmodes(self, which='all'):
        """
        Returns the desired modeshapes
        
        Args:
            which: 'all', or an integer, or a list of integers, or a slice object
        
        Examples of use:
            getEigenmodes()  #returns all computed eigenmodes
            getEigenmodes(3) #returns mode number 4
            getEigenmodes([3,5]) #returns modes number 4 and 6
            getEigenmodes(slice(0,None,2)) #returns even modes
        """
        indexes = _slice_array(np.arange(self._getNout()), which)
        eigenmodes = [ fem.Function(self.function_space) for i in range(np.size(indexes)) ]
        for i, eigM in zip(indexes, eigenmodes):
            self.getEigenpair(i, eigM.vector) # Save eigenvector in eigM
        return eigenmodes
    
    def getModalBasis(self):
        return ModalBasis(self.getWn(), self.getEigenmodes())

    def getErrors(self):
        """Returns the error estimate on the computed eigenvalues"""
        return np.array([self.computeError(i, SLEPc.EPS.ErrorType.RELATIVE) for i in range(self._getNout())]) # Compute error for i-th eigenvalue
    
    def plot(self, which='all', **kwargs):
        """
        Plots the desired modeshapes
        
        Args:
            which: 'all', or an integer, or a list of integers, or a slice object
                -> the same as for getEigenmodes
        """
        self.getModalBasis().plot(which, **kwargs)
    
    def printEigenvalues(self):
        """Prints the computed eigenvalues and error estimates"""
        v = [self.getEigenvalue(i) for i in range(self._getNout())]
        e = self.getErrors()
        PETSc.Sys.Print("       eigenvalue \t\t\t error ")
        for cv, ce in zip(v, e): PETSc.Sys.Print(cv, '\t', ce)

    def _getNout(self):
        """Returns the number of eigenpairs that can be returned. Usually equal to 'nev'."""
        nconv = self.getConverged()
        nev, _, _ = self.getDimensions()
        nout = min(nev, nconv)
        return nout

def _slice_array(a, which):
    """Not intended to be called by user"""
    if which == 'all'    : which = slice(0,None,None)
    if type(which) is int: which = slice(which, which+1, None)
    return a[which]

