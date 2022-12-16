#TODO : Euler 1, RK4

import time
from dolfinx import fem
from petsc4py import PETSc
try: from tqdm import tqdm
except ModuleNotFoundError: tqdm = lambda x: x

from elastodynamicsx.plotting import CustomScalarPlotter, CustomVectorPlotter

class TimeStepper:
    """
    Base class for solving time-dependent problems.
    
    **kwargs:
        live_plotter = (i, **live_plotter_kwargs):
            if i>0: initializes a plotter that will display the result every 'i' step. Ex: i=0 means no display, i=10 means a refresh every 10 step.
            **kwargs: to be passed to pyvista.add_mesh
    """

    def build(*args, **kwargs):
        """
        Convenience static method that instanciates the required time-stepping scheme
        
        -- Input --
        *args: (passed to the required scheme)
        **kwargs: the important parameter is 'scheme'
           scheme: available options are: 'leapfrog', 'midpoint'
        """
        scheme = kwargs.pop('scheme', 'unknown')
        if   scheme.lower() == 'leapfrog': return LeapFrog(*args, **kwargs) #args = (a_tt, a_xx, L, dt, V, bcs=[])
        elif scheme.lower() == 'midpoint': return MidPoint(*args, **kwargs) #args = (a_tt, a_xx, L, dt, V, bcs=[])
        elif scheme.lower() == 'newmark' : return NewmarkBeta(*args, **kwargs) #args = (a_tt, a_xx, L, dt, V, bcs=[], gamma=0.5, beta=0.25)
        elif scheme.lower() == 'newmark-beta': return NewmarkBeta(*args, **kwargs) #args = (a_tt, a_xx, L, dt, V, bcs=[], gamma=0.5, beta=0.25)
        elif scheme.lower() == 'g-a-newmark' : return GalphaNewmarkBeta(*args, **kwargs) #args = (a_tt, a_xx, L, dt, V, bcs=[], alpha_m=0, alpha_f=0, gamma=0.5, beta=0.25)
        elif scheme.lower() == 'midpoint-old': return MidPoint_old(*args, **kwargs) #args = (a_tt, a_xx, L, dt, V, bcs=[])
        else:                              print('TODO')

    def __init__(self, dt, bcs=[], **kwargs):
        #
        self.t_n = 0
        self.dt  = dt
        self.bcs = bcs
        #
        self.callfirsts = []
        self.callbacks  = []
        self.live_plotter = None
        self.live_plotter_step = 0
        ###
        #note: any inherited class must define the following attributes:
        #        self.u_n, self.linear_form, self.bilinear_form
        ###
        #
        self.__compile()
        self.__init_solver()
    
    def initial_condition(self, u, du, t0=0): print('Supercharge me')
    def prepareNextIteration(self): print('Supercharge me')

    def __cbck_livePlot_scalar(self, i, tStepper):
        # Viewing while calculating: Update plotter
        if self.live_plotter_step > 0 and i % self.live_plotter_step == 0:
            self.live_plotter.update_scalars(tStepper.u_n.x.array)
            time.sleep(0.01)

    def __cbck_livePlot_vector(self, i, tStepper):
        # Viewing while calculating: Update plotter
        if self.live_plotter_step > 0 and i % self.live_plotter_step == 0:
            self.live_plotter.update_vectors(tStepper.u_n.x.array)
            time.sleep(0.01)

    def __compile(self):
        ###
        # Declare left-hand-side matrix 'A' and right-hand-side vector 'b'
        #    build 'A' once for all
        #    declare 'b' (to be updated in the time loop)
        self.A = fem.petsc.assemble_matrix(self.bilinear_form, bcs=self.bcs)
        self.A.assemble()
        self.b = fem.petsc.create_vector(self.linear_form)
    
    def __init_solver(self):
        ###
        # Solver
        self.solver = PETSc.KSP().create(self.u_n.function_space.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
    
    def run(self, num_steps, **kwargs):
        """
        Run the loop on time steps
        
        -- Input --
        num_steps: number of time steps to integrate
        **kwargs: important optional parameters are 'callfirsts' and 'callbacks'
           callfirsts: (default=[]) list of functions to be called at the beginning of each iteration (before solving). For instance: update a source term.
                       Each callfirst if of the form: cf = lambda i, timestepper: do_something; where i is the iteration index and timestepper is the timestepper being run
           callbacks:  (detault=[]) similar to callfirsts, but the callbacks are called at the end of each iteration (after solving). For instance: store/save, plot, print, ...
                       Each callback if of the form: cb = lambda i, timestepper: do_something; where i is the iteration index and timestepper is the timestepper being run
           -- other optional parameters --
           live_plotter: (default=None) setting live_plotter={...} will forward these parameters to 'set_live_plotter' (see documentation)
           verbose     : (default=0) verbosity level. >9 means an info msg before each step
        """
        ###
        verbose = kwargs.get('verbose', 0)
        
        if not kwargs.get('live_plotter', None) is None:
            if verbose >= 10: PETSc.Sys.Print('Initializing live plotter...')
            self.set_live_plotter(**kwargs.get('live_plotter'))
        
        callfirsts = kwargs.get('callfirsts', [lambda i, tStepper: 1]) + self.callfirsts
        callbacks  = kwargs.get('callbacks',  [lambda i, tStepper: 1]) + self.callbacks
        
        if self.live_plotter_step > 0:
            self.live_plotter.show(interactive_update=True)
        
        for i in tqdm(range(num_steps)):
            self.t_n += self.dt
            
            if verbose >= 10: PETSc.Sys.Print('Callfirsts...')
            for callfirst in callfirsts: callfirst(i, self) #<- update stuff #F_body.interpolate(F_body_function(t_n))

            # Update the right hand side reusing the initial vector
            if verbose >= 10: PETSc.Sys.Print('Update the right hand side reusing the initial vector...')
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.linear_form)
            
            # Apply Dirichlet boundary condition to the vector // even without Dirichlet BC this is important for parallel computing
            if verbose >= 10: PETSc.Sys.Print('Applying BCs and ghostUpdate...')
            fem.petsc.apply_lifting(self.b, [self.bilinear_form], [self.bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, self.bcs)

            # Solve linear problem
            if verbose >= 10: PETSc.Sys.Print('Solving...')
            self.solver.solve(self.b, self.u_n.vector)
            self.u_n.x.scatter_forward()
            
            #
            if verbose >= 10: PETSc.Sys.Print('Time-stepping for next iteration...')
            self.prepareNextIteration()
            
            if verbose >= 10: PETSc.Sys.Print('Callbacks...')
            for callback in callbacks: callback(i, self) #<- store solution, plot, print, ...

    def set_live_plotter(self, live_plotter_step=1, **kwargs):
        """
        Enable and configure the plotter for displaying the current result within the run() loop
        
        live_plotter_step: step for refreshing the plot.
            >=0 means no live plot, 1 means refresh at each step, 10 means refresh each 10 steps, ...
        
        **kwargs: optional parameters to be passed to CustomScalarPlotter / CustomVectorPlotter
        """
        ###
        self.live_plotter_step = live_plotter_step
        #        
        if self.live_plotter_step > 0:
            dim = self.u_n.function_space.element.num_sub_elements #0 for scalar FunctionSpace, 2 for 2D VectorFunctionSpace
            if dim == 0: #scalar
                self.live_plotter = CustomScalarPlotter(self.u_n, **kwargs)
                self.callbacks.append(self.__cbck_livePlot_scalar)
            else: #vector
                self.live_plotter = CustomVectorPlotter(self.u_n, **kwargs)
                self.callbacks.append(self.__cbck_livePlot_vector)

# -----------------------------------------------------
# Import subclasses -- must be done at the end to avoid loop imports
# -----------------------------------------------------
from .leapfrog import LeapFrog
from .midpoint import MidPoint_old
from .newmark  import GalphaNewmarkBeta, NewmarkBeta, MidPoint

