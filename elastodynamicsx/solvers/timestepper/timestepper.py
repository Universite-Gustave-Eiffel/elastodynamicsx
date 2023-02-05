#TODO : Euler 1, RK4

import time
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
try: from tqdm.auto import tqdm
except ModuleNotFoundError: tqdm = lambda x: x

from elastodynamicsx.plot import CustomScalarPlotter, CustomVectorPlotter

class TimeStepper:
    """
    Base class for solving time-dependent problems.
    """
    
    ### --------------------------
    ### --------- static ---------
    ### --------------------------

    labels = ['supercharge me']
    petsc_options_t0 = {"ksp_type": "preonly", "pc_type": "lu"} #PETSc options to solve a0 = M_inv.(F(t0) - C.v0 - K(u0))
    petsc_options_explicit_scheme = petsc_options_t0
    petsc_options_implicit_scheme_linear = {"ksp_type": "preonly", "pc_type": "lu"}
    #petsc_options_implicit_scheme_nonlinear = #TODO
    
    def build(*args, **kwargs):
        """
        Convenience static method that instanciates the required time-stepping scheme
        
        Args:
            args: (passed to the required scheme)
            kwargs: The required parameter is 'scheme'. The other **kwargs are passed to the scheme
                scheme: Available options are:
                    'leapfrog'
                    'midpoint'
                    'linear-acceleration-method'
                    'newmark'
                    'hht-alpha'
                    'generalized-alpha'
        """
        scheme = kwargs.pop('scheme', 'unknown')
        allSchemes = (LeapFrog, MidPoint, LinearAccelerationMethod, NewmarkBeta, HilberHughesTaylor, GalphaNewmarkBeta)
        for s_ in allSchemes:
          if scheme.lower() in s_.labels: return s_(*args, **kwargs)
        #
        raise TypeError('unknown scheme: '+scheme)
        
    def Courant_number(domain, c_max, dt):
        """
        The Courant number: C = c_max*dt/h, with h the cell diameter
        
        Related to the Courant-Friedrichs-Lewy (CFL) condition.

        see: https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
        """
        V = fem.FunctionSpace(domain, ("DG", 0))
        c_number = fem.Function(V)
        
        if type(V.element.interpolation_points) == np.ndarray:
            pts = V.element.interpolation_points   #DOLFINx.__version__ < 0.5
        else:
            pts = V.element.interpolation_points() #DOLFINx.__version__ >=0.5
        
        h = ufl.MinCellEdgeLength(V.mesh) #or rather ufl.CellDiameter?
        c_number.interpolate(fem.Expression(dt*c_max/h, pts))
        return V.mesh.comm.allreduce( np.amax(c_number.x.array) , op=MPI.MAX)


    ### --------------------------
    ### ------- non-static -------
    ### --------------------------
    
    def __init__(self, function_space, m_, c_, k_, L, dt, bcs=[], explicit=False, **kwargs):
        """
        Args:
            function_space: The function space
            m_: Function of u,v that returns the mass form
            c_: Function of u,v that returns the damping form
            k_: Function of u,v that returns the stiffness form
            L : Function of v   that returns the linear form
            dt: Time step
            bcs: List of instances of the class BoundaryCondition
            kwargs:
                petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
        """
        #
        self._t   = 0
        self._dt  = dt
        self._bcs = bcs
        self._m_function = m_
        self._c_function = c_
        self._k_function = k_
        #
        self._callfirsts = []
        self._callbacks  = []
        self.live_plotter = None
        self.live_plotter_step = 0
        #
        self._explicit = explicit
        ###
        #note: any inherited class must define the following attributes:
        #        self._u_n, self._v_n, self._a_n,
        #        self.linear_form, self.bilinear_form,
        #        self._u0, self._v0, self._a0
        #        self._m0_form, self._L0_form
        ###
        #
        if self.explicit:
            default_petsc_options = TimeStepper.petsc_options_explicit_scheme
        else:
            default_petsc_options = TimeStepper.petsc_options_implicit_scheme_linear
        petsc_options = kwargs.get('petsc_options', default_petsc_options)
        self._init_solver(petsc_options)
        
    @property
    def A(self): return self._A
    
    @property
    def b(self): return self._b
    
    @property
    def explicit(self): return self._explicit
    
    @property
    def t(self): return self._t
    
    @property
    def dt(self): return self._dt
    
    @property
    def u(self): return self._u_n

    @property
    def v(self): return self._v_n
    
    def Energy_elastic(self):
        domain = self.u.function_space.mesh
        return domain.comm.allreduce( fem.assemble_scalar(fem.form( 1/2* self._k_function(self.u, self.u) )) , op=MPI.SUM)

    def Energy_damping(self):
        domain = self.u.function_space.mesh
        if self._c_function is None:
            return 0
        else:
            return self.dt*domain.comm.allreduce( fem.assemble_scalar(fem.form( self._c_function(self.v, self.v) )) , op=MPI.SUM)

    def Energy_kinetic(self):
        domain = self.u.function_space.mesh
        return domain.comm.allreduce( fem.assemble_scalar(fem.form( 1/2* self._m_function(self.v, self.v) )) , op=MPI.SUM)
    
    def initial_condition(self, u0, v0, t0=0):
        ###
        """
        Apply initial conditions
        
        Args:
            u0: u at t0
            v0: du/dt at t0
            t0: start time (default: 0)
        
            u0 and v0 can be:
                - function -> interpolated at nodes
                    -> e.g. u0 = lambda x: np.zeros((domain.topology.dim, x.shape[1]), dtype=PETSc.ScalarType)
                - scalar (int, float, complex, PETSc.ScalarType)
                    -> e.g. u0 = 0
                - array (list, tuple, np.ndarray) or fem.function.Constant
                    -> e.g. u0 = [0,0,0]
                - fem.function.Function
                    -> e.g. u0 = fem.Function(V)
        """
        self._t = t0
        for selfVal, val in ((self._u0, u0), (self._v0, v0)):
            if   type(val) == type(lambda x:x):
                selfVal.interpolate(val)
            elif issubclass(type(val), fem.function.Constant):
                selfVal.x.array[:] = np.tile(val.value, np.size(selfVal.x.array)//np.size(val.value))
            elif type(val) in (list, tuple, np.ndarray):
                selfVal.x.array[:] = np.tile(val, np.size(selfVal.x.array)//np.size(val))
            elif type(val) in (int, float, complex, PETSc.ScalarType):
                selfVal.x.array[:] = val
            elif issubclass(type(val), fem.function.Function):
                selfVal.x.array[:] = val.x.array
            else:
                raise TypeError("Unknown type of initial value "+str(type(val)))

    def _cbck_livePlot_scalar(self, i, tStepper):
        # Viewing while calculating: Update plotter
        if self.live_plotter_step > 0 and i % self.live_plotter_step == 0:
            self.live_plotter.update_scalars(tStepper.u.x.array)
            time.sleep(0.01)

    def _cbck_livePlot_vector(self, i, tStepper):
        # Viewing while calculating: Update plotter
        if self.live_plotter_step > 0 and i % self.live_plotter_step == 0:
            self.live_plotter.update_vectors(tStepper.u.x.array)
            time.sleep(0.01)
    
    def _init_solver(self, petsc_options={}):
        # see https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/fem/petsc.py
        # see also https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html
        ###   ###   ###   ###
        # Declare left-hand-side matrix 'A' and right-hand-side vector 'b'
        #    build 'A' once for all
        #    declare 'b' (to be updated in the time loop)
        self._A = fem.petsc.assemble_matrix(self.bilinear_form, bcs=self._bcs)
        self._A.assemble()
        self._b = fem.petsc.create_vector(self.linear_form)
        
        # Solver
        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self._A)
        #self.solver.setType(PETSc.KSP.Type.PREONLY)
        #self.solver.getPC().setType(PETSc.PC.Type.LU)
        
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
        
        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()
        
    
    def run(self, num_steps, **kwargs): print('Supercharge me')

    def set_live_plotter(self, live_plotter_step=1, **kwargs):
        """
        Enable and configure the plotter to display the current result within the run() loop
        
        Args:
            live_plotter_step: Step for refreshing the plot.
                >=0 means no live plot, 1 means refresh at each step, 10 means refresh each 10 steps, ...
            kwargs: Passed to CustomScalarPlotter / CustomVectorPlotter
        """
        ###
        self.live_plotter_step = live_plotter_step
        #        
        if self.live_plotter_step > 0:
            dim = self.u.function_space.element.num_sub_elements #0 for scalar FunctionSpace, 2 for 2D VectorFunctionSpace
            if dim == 0: #scalar
                self.live_plotter = CustomScalarPlotter(self.u, **kwargs)
                self._callbacks.append(self._cbck_livePlot_scalar)
            else: #vector
                self.live_plotter = CustomVectorPlotter(self.u, **kwargs)
                self._callbacks.append(self._cbck_livePlot_vector)




class OneStepTimeStepper(TimeStepper):
    """
    Base class for solving time-dependent problems with one-step algorithms (e.g. Newmark-beta methods).
    """
    
    def __init__(self, function_space, m_, c_, k_, L, dt, bcs=[], explicit=False, **kwargs):
        #
        self._i0 = 0
        self._intermediate_dt = 0
        super().__init__(function_space, m_, c_, k_, L, dt, bcs, explicit, **kwargs)

    def _prepareNextIteration(self): print('Supercharge me')
    def _initialStep(self, callfirsts, callbacks, verbose=0):
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
        
        if verbose >= 10: PETSc.Sys.Print('Initial value problem solved, entering loop')
        #no callback because u1 is not solved yet
        #
        ### -------------------------------------------------

    def run(self, num_steps, **kwargs):
        """
        Run the loop on time steps
        
        Args:
            num_steps: number of time steps to integrate
            kwargs: important optional parameters are 'callfirsts' and 'callbacks'
                callfirsts: (default=[]) list of functions to be called at the beginning
                    of each iteration (before solving). For instance: update a source term.
                    Each callfirst if of the form: cf = lambda t, timestepper: do_something
                    where t is the time at which to evaluate the sources and
                    timestepper is the timestepper being run
                callbacks:  (detault=[]) similar to callfirsts, but the callbacks are called
                    at the end of each iteration (after solving). For instance: store/save, plot, print, ...
                    Each callback if of the form: cb = lambda i, timestepper: do_something
                    where i is the iteration index and timestepper is the timestepper being run
               -- other optional parameters --
               live_plotter: (default=None) Setting live_plotter={...} will forward
                   these parameters to 'set_live_plotter' (see documentation)
               verbose     : (default=0) Verbosity level. >9 means an info msg before each step
        """
        ###
        verbose = kwargs.get('verbose', 0)
        
        if not kwargs.get('live_plotter', None) is None:
            if verbose >= 10: PETSc.Sys.Print('Initializing live plotter...')
            self.set_live_plotter(**kwargs.get('live_plotter'))
        
        callfirsts = kwargs.get('callfirsts', [lambda t, tStepper: 1]) + self._callfirsts
        callbacks  = kwargs.get('callbacks',  [lambda i, tStepper: 1]) + self._callbacks
        
        if self.live_plotter_step > 0:
            self.live_plotter.show(interactive_update=True)
        
        self._initialStep(callfirsts, callbacks, verbose=verbose)
        
        for i in tqdm(range(self._i0, num_steps)):
            self._t += self.dt
            
            if verbose >= 10: PETSc.Sys.Print('Callfirsts...')
            for callfirst in callfirsts: callfirst(self.t - self.dt*self._intermediate_dt, self) #<- update stuff #F_body.interpolate(F_body_function(t))

            # Update the right hand side reusing the initial vector
            if verbose >= 10: PETSc.Sys.Print('Update the right hand side reusing the initial vector...')
            with self._b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self._b, self.linear_form)
            
            # Apply Dirichlet boundary condition to the vector // even without Dirichlet BC this is important for parallel computing
            if verbose >= 10: PETSc.Sys.Print('Applying BCs and ghostUpdate...')
            fem.petsc.apply_lifting(self._b, [self.bilinear_form], [self._bcs])
            self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self._b, self._bcs)

            # Solve linear problem
            if verbose >= 10: PETSc.Sys.Print('Solving...')
            self.solver.solve(self._b, self._u_n.vector)
            self._u_n.x.scatter_forward()
            
            #
            if verbose >= 10: PETSc.Sys.Print('Time-stepping for next iteration...')
            self._prepareNextIteration()
            
            if verbose >= 10: PETSc.Sys.Print('Callbacks...')
            for callback in callbacks: callback(i, self) #<- store solution, plot, print, ...

# -----------------------------------------------------
# Import subclasses -- must be done at the end to avoid loop imports
# -----------------------------------------------------
from .leapfrog import LeapFrog
from .newmark  import GalphaNewmarkBeta, HilberHughesTaylor, NewmarkBeta, MidPoint, LinearAccelerationMethod

