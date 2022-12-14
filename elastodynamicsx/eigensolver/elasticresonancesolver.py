#TODO: documentation

from dolfinx import fem, plot
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import ufl
import numpy as np
import pyvista

class ElasticResonanceSolver(SLEPc.EPS):
    """
    Convenience class inhereted from SLEPc.EPS, with default parameters and convenience methods that are relevant for computing the resonances of an elastic component.
    
    Minimalistic example (free resonances of an elastic cube):
        
        from dolfinx import mesh, fem
        from mpi4py import MPI
        from elastodynamicsx.eigensolver import ElasticResonanceSolver
        #
        domain = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0., 0., 0.], [1., 1., 1.]], [10, 10, 10])
        V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        #
        rho, lambda_, mu = 1, 2, 1
        eps = ElasticResonanceSolver.build_isotropicMaterial(rho, lambda_, mu, V, bcs=[], nev=6+6) #the first 6 resonances are rigid body motion
        eps.solve()
        eps.plot()
        freqs = eps.getEigenfrequencies()
        print('First resonance frequencies:', freqs)
    """

    def build_isotropicMaterial(rho, lambda_, mu, function_space, bcs=[], **kwargs):
        """
        Convenience static method that instanciates a ElasticResonanceSolver for given isotropic mechanical properties
        """
        # -----------------------------------------------------
        #                        PDE
        # -----------------------------------------------------
        def epsilon(u): return ufl.sym(ufl.grad(u))
        def sigma(u): return lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)
        
        a_tt = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
        a_xx = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        ###
        return ElasticResonanceSolver(a_tt, a_xx, function_space, bcs=bcs, **kwargs)

    def __init__(self, a_tt, a_xx, function_space, bcs=[], **kwargs):
        """
        a_tt: function(u,v) that returns the ufl expression of the bilinear form with second derivative on time
                 -> used to build the mass matrix
                 -> usually: a_tt = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
        a_xx: function(u,v) that returns the ufl expression of the bilinear form with no derivative on time
                 -> used to build the stiffness matrix
                 -> usually: a_xx = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        function_space: the Finite Element functionnal space
        bcs: the set of boundary conditions
        """
        #
        super().__init__()
        
        self.function_space = function_space
        
        ### Variational problem
        u, v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        m_form = fem.form(a_tt(u,v))
        k_form = fem.form(a_xx(u,v))
        #
        ### assemble mass M and stifness K matrices
        M = fem.petsc.assemble_matrix(m_form, bcs=bcs)
        M.assemble()
        K = fem.petsc.assemble_matrix(k_form, bcs=bcs)
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

    def getEigenfrequencies(self):
        """Returns the eigenfrequencies from the computed eigenvalues"""
        return np.array([np.sqrt(abs(self.getEigenvalue(i).real))/(2*np.pi) for i in range(self.__getNout())]) #abs because rigid body motions may lead to minus zero: -0.00000

    def getEigenmodes(self, which='all'):
        """
        Returns the desired modeshapes
        which: 'all', or an integer, or a list of integers, or a slice object
        
        examples:
            getEigenmodes()  #returns all computed eigenmodes
            getEigenmodes(3) #returns mode number 4
            getEigenmodes([3,5]) #returns modes number 4 and 6
            getEigenmodes(slice(0,None,2)) #returns even modes
        """
        indexes = _slice_array(np.arange(self.__getNout()), which)
        eigenmodes = [ fem.Function(self.function_space) for i in range(np.size(indexes)) ]
        for i, eigM in zip(indexes, eigenmodes):
            self.getEigenpair(i, eigM.vector) # Save eigenvector in eigM
        return eigenmodes

    def getErrors(self):
        """Returns the error estimate on the computed eigenvalues"""
        return np.array([self.computeError(i, SLEPc.EPS.ErrorType.RELATIVE) for i in range(self.__getNout())]) # Compute error for i-th eigenvalue
    
    def plot(self, which='all', **kwargs):
        """
        Plots the desired modeshapes
        which: 'all', or an integer, or a list of integers, or a slice object
            -> the same as for getEigenmodes
        """
        #inspired from https://docs.pyvista.org/examples/99-advanced/warp-by-vector-eigenmodes.html
        indexes = _slice_array(np.arange(self.__getNout()), which)
        eigenmodes = self.getEigenmodes(which)
        eigenfreqs = self.getEigenfrequencies()
        #
        topology, cell_types, geom = plot.create_vtk_mesh(self.function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        nbcomps = eigenmodes[0].x.array.size // geom.shape[0] #number of components
        if nbcomps < 3:
            z0s = np.zeros((geom.shape[0], 3-nbcomps), dtype=eigenmodes[0].x.array.dtype)
        for i, eigM in zip(indexes, eigenmodes):
            if nbcomps == 3: grid['eigenmode_'+str(i)] = eigM.x.array.reshape((geom.shape[0], 3)) #ok if 3D. not ok if 2D.
            else:            grid['eigenmode_'+str(i)] = np.append(eigM.x.array.reshape((geom.shape[0], nbcomps)), z0s, axis=1)
        #
        nbcols = int(np.ceil(np.sqrt(indexes.size)))
        nbrows = int(np.ceil(indexes.size/nbcols))
        shape  = kwargs.get('shape', (nbrows, nbcols))
        factor = kwargs.get('factor', 1.)
        plotter = pyvista.Plotter(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                plotter.subplot(i,j)
                current_index = i*shape[1] + j
                if current_index >= indexes.size: break
                vector = 'eigenmode_'+str(indexes[current_index])
                plotter.add_text("mode "+str(indexes[current_index])+", freq. "+str(round(eigenfreqs[indexes[current_index]],2)), font_size=10)
                if kwargs.get('wireframe', False): plotter.add_mesh(grid, style='wireframe', color='black')
                plotter.add_mesh(grid.warp_by_vector(vector, factor=factor), scalars=vector)
        plotter.show()
    
    def printEigenvalues(self):
        """Prints the computed eigenvalues and error estimates"""
        v = [self.getEigenvalue(i) for i in range(self.__getNout())]
        e = self.getErrors()
        PETSc.Sys.Print("       eigenvalue \t\t\t error ")
        for cv, ce in zip(v, e): PETSc.Sys.Print(cv, '\t', ce)

    def __getNout(self):
        """Returns the number of eigenpairs that can be returned. Usually equal to 'nev'."""
        nconv = self.getConverged()
        nev, _, _ = self.getDimensions()
        nout = min(nev, nconv)
        return nout

def _slice_array(a, which):
    if which is 'all': which = slice(0,None,None)
    if type(which) is int: which = slice(which, which+1, None)
    return a[which]

