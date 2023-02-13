from petsc4py import PETSc
try: from tqdm.auto import tqdm
except ModuleNotFoundError: tqdm = lambda x: x



class FrequencyDomainSolver:
    """
    Class for solving frequency domain problems.
    
    Example of use:
        #imports
        from dolfinx import mesh, fem
        import ufl
        from mpi4py import MPI
        from elastodynamicsx.solvers import FrequencyDomainSolver
        from elastodynamicsx.pde import material, BoundaryCondition, PDE

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
        
        #PDE
        pde = PDE(V, materials=[mat], bodyforces=[bf], bcs=bcs)
        
        #solve
        fdsolver = FrequencyDomainSolver(V.mesh.comm, pde.M(), pde.C(), pde.K(), pde.b())
        omega    = 1.0
        u        = fem.Function(V, name='solution')
        fdsolver.solve(omega=omega, out=u.vector)
    """
    
    default_petsc_options = {"ksp_type": "preonly", "pc_type": "lu"} #"pc_factor_mat_solver_type": "mumps"

    def __init__(self, comm:'_MPI.Comm', M:PETSc.Mat, C:PETSc.Mat, K:PETSc.Mat, b:PETSc.Vec, b_update_function:'function'=None, **kwargs):
        """
        Args:
            comm: The MPI communicator
            M: The mass matrix
            C: The damping matrix
            K: The stiffness matrix
            b: The load vector
            b_update_function: A function that updates the load vector (in-place)
                The function must take b,omega as parameters.
                e.g.: b_update_function = lambda b,omega: b[:]=omega
                If set to None, the call is ignored.
            kwargs:
                petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
        """
        self._M = M
        self._C = C
        self._K = K
        self._b = b
        self._b_update_function = b_update_function
        
        #### ####
        # Initialize the PETSc solver
        petsc_options = kwargs.get('petsc_options', FrequencyDomainSolver.default_petsc_options)
        self.solver = PETSc.KSP().create(comm)
        #self.solver.setOperators(self._A) #do it at solve
        
        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self.solver.setOptionsPrefix(problem_prefix)
        
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self.solver.setFromOptions()
        #### ####


    def solve(self, omega, out:PETSc.Vec=None, callbacks:list=[], **kwargs) -> PETSc.Vec:
        """
        Solve the linear problem
        
        Args:
            omega: The angular frequency (scalar or array)
            out: The solution (displacement field) to the last solve. If
                None a new PETSc.Vec is created
            callbacks: If omega is an array, list of callback functions
                to be called after each solve (e.g. plot, store solution, ...).
            kwargs:
                live_plotter: a plotter object that can refresh through
                a live_plotter.live_plotter_update_function(i, out) function
        
        Returns:
            out
        """
        if out is None:
            out = self._b.copy()
        if hasattr(omega, '__iter__'):
            return self._solve_multiple_omegas(omega, out, callbacks, **kwargs)
        else:
            return self._solve_single_omega(omega, out)


    def _solve_single_omega(self, omega, out:PETSc.Vec) -> PETSc.Vec:
        #update load vector at angular frequency 'omega'
        if not(self._b_update_function is None):
            self._b_update_function(self._b, omega)
            #self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE) #assume this has already been done
        
        #update PDE matrix
        w = omega
        A = PETSc.ScalarType(-w*w)*self._M + PETSc.ScalarType(1J*w)*self._C + self._K
        self.solver.setOperators(A)

        #solve
        self.solver.solve(self._b, out)
        
        #update the ghosts in the solution
        out.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return out
    
    
    def _solve_multiple_omegas(self, omegas, out:PETSc.Vec, callbacks:list=[], **kwargs) -> PETSc.Vec:
        #loop on values in omegas -> _solve_single_omega
        
        live_plt = kwargs.get('live_plotter', None)
        if not(live_plt is None):
            if type(live_plt)==dict:
                from elastodynamicsx.plot import live_plotter
                live_plt = live_plotter(self.u, live_plt.pop('refresh_step', 1), **live_plt)
            callbacks.append(live_plt.live_plotter_update_function)
            live_plt.show(interactive_update=True)
            
        for i in tqdm(range(len(omegas))):
            self._solve_single_omega(omegas[i], out)
            for callback in callbacks:
                callback(i, out) #<- store solution, plot, print, ...
        return out

    
    @property
    def M(self) -> PETSc.Mat:
        """The mass matrix"""
        return self._M
    
    @property
    def C(self) -> PETSc.Mat:
        """The damping matrix"""
        return self._C
    
    @property
    def K(self) -> PETSc.Mat:
        """The stiffness matrix"""
        return self._K

    @property
    def b(self) -> PETSc.Vec:
        """The load vector"""





    def __OLD__solve_single_omega(self, omega): #TODO: remove
        w = omega
        a = -w*w*self._m + 1J*w*self._c + self._k
        self._problem = fem.petsc.LinearProblem(a, self._L, bcs=self._bcs, petsc_options=self._petsc_options)
        return self._problem.solve()
    

