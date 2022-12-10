#TODO : Euler 1, RK4

from dolfinx import fem
from petsc4py import PETSc
import ufl

class TimeStepper:

    def build(*args, **kwargs):
        tscheme = kwargs.get('timescheme', 'leapfrog')
        if tscheme.lower() == 'leapfrog': return LeapFrog(*args) #args = (dt, V, a_tt, a_xx, L)
        else: print('TODO')

    def __init__(self, dt):
        #
        self.t_n = 0
        self.dt  = dt
        ###
        #note: any inherited class must define the following attributes:
        #        self.u_n, self.linear_form, self.bilinear_form
        ###
        #
        self.__compile()
        self.__init_solver()
    
    def initial_condition(self, u, du, t0=0): print('Supercharge me')
    def prepareNextIteration(self): print('Supercharge me')
    
    def __compile(self):
        ###
        # Declare left-hand-side matrix 'A' and right-hand-side vector 'b'
        #    build 'A' once for all
        #    declare 'b' (to be updated in the time loop)
        self.A = fem.petsc.assemble_matrix(self.bilinear_form, bcs=[])
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
        callfirsts = kwargs.get('callfirsts', [lambda i, tStepper: 1])
        callbacks  = kwargs.get('callbacks',  [lambda i, tStepper: 1])
        for i in range(num_steps):
            self.t_n += self.dt
            
            for callfirst in callfirsts: callfirst(i, self) #<- update stuff #F_body.interpolate(F_body_function(t_n))

            # Update the right hand side reusing the initial vector
            with self.b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.linear_form)
            
            # Apply Dirichlet boundary condition to the vector // even without Dirichlet BC this is important for parallel computing
            fem.petsc.apply_lifting(self.b, [self.bilinear_form], [[]])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, [])

            # Solve linear problem
            self.solver.solve(self.b, self.u_n.vector)
            self.u_n.x.scatter_forward()
            
            #
            self.prepareNextIteration()
            
            for callback in callbacks: callback(i, self) #<- store solution, plot, print, ...


class LeapFrog(TimeStepper):
# --------------------------scheme-dependent part----------------------------
#  Variational problem
#      -> Explicit scheme: Leapfrog method
# ---------------------------------------------------------------------------
    def __init__(self, dt, V, a_tt, a_xx, L):
        #
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        #
        self.u_n     = fem.Function(V, name="u") #u(t)
        self.__u_nm1 = fem.Function(V)           #u(t-dt)
        self.__u_nm2 = fem.Function(V)           #u(t-2*dt)
        #
        self.__a = a_tt(u,v)
        self.__L = dt*dt*L(v) - dt*dt*a_xx(self.__u_nm1, v) + 2*a_tt(self.__u_nm1,v) - a_tt(self.__u_nm2,v)
        self.bilinear_form = fem.form(self.__a)
        self.linear_form   = fem.form(self.__L)
        #
        super().__init__(dt)

    def initial_condition(self, u, du, t0=0):
        ###
        """
        Apply initial conditions
        
        u: u at t0
        du: du/dt at t0
        t0: start time (default: 0)
        """
        self.t_n = t0
        self.__u_nm2.interpolate(u) #u_nm2 = u(0)
        self.__u_nm1.interpolate(u) #faux, u_nm1 = u(0) + dt*u'(0)
        self.u_n.interpolate(u) #faux
    
    def prepareNextIteration(self):
        """Next-time-step function, to prepare next iteration -> Call it after solving"""
        self.__u_nm2.x.array[:] = self.__u_nm1.x.array
        self.__u_nm1.x.array[:] = self.u_n.x.array

