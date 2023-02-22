from dolfinx import fem
from petsc4py import PETSc
import numpy as np



class TimeScheme():
    """Abstract base class for time schemes as needed by the TimeStepper solvers"""

    labels = ['supercharge me']
    
    def build_timestepper(*args, **kwargs) -> 'TimeStepper': # supercharge me
        raise NotImplementedError
    
    def __init__(self, dt, out:PETSc.Vec, **kwargs):
        self._dt = dt
        self._out = out
        self._explicit = kwargs.get('explicit', False)
        self._intermediate_dt = kwargs.get('intermediate_dt', 0.)

    @property
    def explicit(self) -> bool:
        return self._explicit
        
    @property
    def dt(self):
        """The time step"""
        return self._dt
    
    @property
    def intermediate_dt(self):
        return self._intermediate_dt
        
    @property
    def out(self) -> PETSc.Vec:
        """The solution vector"""
        return self._out
        
    def b_update_function(self, b:PETSc.Vec, t) -> None: # supercharge me
        raise NotImplementedError
    
    def prepareNextIteration(self) -> None: # supercharge me
        raise NotImplementedError

    def initial_condition(self, u0, v0) -> None: # supercharge me
        raise NotImplementedError
    
    def initialStep(self, t0, callfirsts:list=[], callbacks:list=[], verbose=0) -> None: # supercharge me
        raise NotImplementedError



class FEniCSxTimeScheme(TimeScheme):
    """Abstract base class based on FEniCSx's form language"""
    def __init__(self, dt, out:fem.Function, bilinear_form:'dolfinx.fem.forms.Form', linear_form:'dolfinx.fem.forms.Form', bcs=[], **kwargs):
        super().__init__(dt, out.vector, **kwargs)
        self._bilinear_form = bilinear_form
        self._linear_form   = linear_form
        self._bcs           = bcs # dirichlet BCs only
        self._out_fenicsx   = out

    @property
    def out_fenicsx(self) -> fem.Function:
        """The solution vector"""
        return self._out_fenicsx
        
    def A(self) -> PETSc.Mat:
        A = fem.petsc.assemble_matrix(self._bilinear_form, bcs=self._bcs)
        A.assemble()
        return A
        
    def init_b(self) -> PETSc.Vec:
        return fem.petsc.create_vector(self._linear_form)

    def b_update_function(self, b:PETSc.Vec, t) -> None: #TODO: use t?
        """Updates the b vector (in-place) for a given time t"""
        with b.localForm() as loc_b:
            loc_b.set(0)
        
        #fill with values
        fem.petsc.assemble_vector(b, self._linear_form)
            
        #BC modifyier 
        fem.petsc.apply_lifting(b, [self._bilinear_form], [self._bcs])
        
        #ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        #apply BC value
        fem.petsc.set_bc(b, self._bcs)

    def initial_condition(self, u0, v0) -> None:
        ###
        """
        Apply initial conditions
        
        Args:
            u0: u at t0
            v0: du/dt at t0
        
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



