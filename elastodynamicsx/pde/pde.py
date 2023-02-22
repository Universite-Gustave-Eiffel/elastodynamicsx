from petsc4py import PETSc
import numpy as np
from dolfinx import fem
try:
    import dolfinx_mpc
except ImportError:
    import warnings
    warnings.warn("Can't import dolfinx_mpc. Periodic boundaries are not available", Warning)
    dolfinx_mpc = None
import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker
from . import BoundaryCondition



class PDE():
    """
    Representation of a PDE of the kind:
    
        M*a + C*v + K(u) = b
    
    as an assembly of materials and forces defined over different subdomains
    """
    
    
    def build_mpc(function_space, bcs):
        bcs_strong = BoundaryCondition.get_dirichlet_BCs(bcs)
        bcs_mpc    = BoundaryCondition.get_mpc_BCs(bcs)
        
        if len(bcs_mpc)==0:
            return None
        
        mpc = dolfinx_mpc.MultiPointConstraint(function_space)
        
        for bc in bcs_mpc:
            if bc.type == 'periodic':
                facet_tags, marker, slave_to_master_map = bc.bc
                mpc.create_periodic_constraint_topological(function_space, facet_tags, marker, slave_to_master_map, bcs_strong)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))
        
        mpc.finalize()
        return mpc
    
    
    def __init__(self, function_space, materials=[], bodyforces=[], bcs=[], **kwargs):

        self._function_space = function_space
        self.materials = materials
        self.bodyforces= bodyforces
        self.bcs = bcs
        self._mpc = None
        self._u, self._v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        
        # Sort boundary conditions
        self._bcs_weak   = BoundaryCondition.get_weak_BCs(bcs)      # custom weak BSs, instances of BoundaryCondition
        self._bcs_strong = BoundaryCondition.get_dirichlet_BCs(bcs) # dolfinx.fem.DirichletBCMetaClass
        self._bcs_mpc    = BoundaryCondition.get_mpc_BCs(bcs)       # instances of BoundaryCondition used to add multi-point constraints
        #new_list = filter(lambda v: v not in b, a)
        
        self._omega_ufl = fem.Constant(function_space, PETSc.ScalarType(0))
        
        # Finalize the PDE (compile, ...). Optionnally this can be done
        # manually later on by passing kwargs['finalize']=False
        if kwargs.get('finalize', True):
            self.finalize()
    


    @property
    def is_linear(self):
        return not(sum([not(mat.is_linear) for mat in self.materials]))

    @property
    def mpc(self):
        return self._mpc


### ### ### ### ### ### ### ### ### ###
### Finalize the PDE (compile, ...) ###
### ### ### ### ### ### ### ### ### ###

    def finalize(self) -> None:
        self._build_mpc()
        if self._mpc is None:
            self.update_b_frequencydomain = self._update_b_frequencydomain_WO_MPC
        else:
            self.update_b_frequencydomain = self._update_b_frequencydomain_WITH_MPC
        
        if self.is_linear:
            print('linear PDE')
            self._compile_M_C_K_b()
        else:
            print('non-linear PDE')
    
    
    def _build_mpc(self) -> None:
        """Required for handling multi-point constraints (e.g. periodic BC)"""
        if dolfinx_mpc is None: # if import error
            return
        if len(self._bcs_mpc)==0: # if there is no need to use MPC. Note that MPC slows down the runs even if there is no constraint -> better avoid using it if not needed
            return
        self._mpc = PDE.build_mpc(self._function_space, self._bcs_strong + self._bcs_mpc)
        
        
    def _compile_M_C_K_b(self) -> None:
        """Required for frequency domain problems"""
        u, v = self._u, self._v
        zero = fem.Constant(self._function_space, PETSc.ScalarType(0.))
        vzero = zero if v.ufl_function_space().num_sub_spaces==0 else fem.Constant(self._function_space, PETSc.ScalarType([0.]*len(v)))
        
        #interior
        m = self.m(u,v)
        c = self.c(u,v) if not(self.c is None) else zero*ufl.inner(u,v)*ufl.dx
        k = self.k(u,v)
        L = self.L(v)   if not(self.L is None) else ufl.inner(vzero,v)*ufl.dx
        
        #boundaries
        for bc in self._bcs_weak:
            if bc.type == 'neumann':
                L += bc.bc(v)
            elif bc.type == 'robin':
                F_bc = bc.bc(u,v)
                k += ufl.lhs(F_bc)
                L += ufl.rhs(F_bc)
            elif bc.type == 'dashpot':
                c += bc.bc(u,v)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))
        
        self._m_form = fem.form(m)
        self._c_form = fem.form(c)
        self._k_form = fem.form(k)
        self._b_form = fem.form(L)

        #Executes the following only if using complex numbers
        if np.issubdtype(PETSc.ScalarType, np.complexfloating):
            ##Mat_lhs = -w*w*_M_ + 1J*w*_C_ + _K_
            w = self._omega_ufl
            self._a_form = fem.form(-w*w*m + 1J*w*c + k)
    


### ### ### ### ### ### ### ### ### ### ###
### Linear and bilinear form functions  ###
### ### ### ### ### ### ### ### ### ### ###

    @property
    def m(self) -> 'function':
        """(bilinear) Mass form function"""
        return lambda u,v: sum([mat.m(u,v) for mat in self.materials])
    
    @property
    def c(self) -> 'function':
        """(bilinear) Damping form function"""
        non0dampings = [mat.c for mat in self.materials if not(mat.c) is None]
        if len(non0dampings)==0:
            return None
        else:
            return lambda u,v: sum([c(u,v) for c in non0dampings])
    
    @property
    def k(self) -> 'function':
        """(bilinear) Stiffness form function"""
        return lambda u,v: sum([mat.k(u,v) for mat in self.materials])

    @property
    def k_CG(self) -> 'function':
        """(bilinear) Stiffness form function (Continuous Galerkin)"""
        return lambda u,v: sum([mat.k_CG(u,v) for mat in self.materials])
        
    @property
    def k_DG(self) -> 'function':
        """(bilinear) Stiffness form function (Discontinuous Galerkin)"""
        return lambda u,v: sum([mat.k_DG(u,v) for mat in self.materials])
    
    @property
    def DG_numerical_flux(self) -> 'function':
        """(bilinear) Numerical flux form function (Disontinuous Galerkin)"""
        return lambda u,v: sum([mat.DG_numerical_flux(u,v) for mat in self.materials])
    
    @property
    def L(self) -> 'function':
        """Linear form function"""
        if len(self.bodyforces)==0:
            return None
        else:
            return lambda v: sum([f.L(v) for f in self.bodyforces])



### ### ### ### ### ### ### ###
### Compiled dolfinx forms  ###
### ### ### ### ### ### ### ###

    @property
    def m_form(self) -> 'dolfinx.fem.forms.Form':
        """Compiled mass bilinear form"""
        return self._m_form
        
    @property
    def c_form(self) -> 'dolfinx.fem.forms.Form':
        """Compiled damping bilinear form"""
        return self._c_form
        
    @property
    def k_form(self) -> 'dolfinx.fem.forms.Form':
        """Compiled stiffness bilinear form"""
        return self._k_form
    
    @property
    def b_form(self) -> 'dolfinx.fem.forms.Form':
        """Compiled linear form"""
        return self._b_form



### ### ### ### ### ### ### ### ###
### PETSc matrices and vectors  ###
### ### ### ### ### ### ### ### ###
    
    def M(self) -> PETSc.Mat:
        """Mass matrix"""
        if self._mpc is None:
            M = fem.petsc.assemble_matrix(self._m_form, bcs=self._bcs_strong)
        else:
            M = dolfinx_mpc.assemble_matrix(self._m_form, self._mpc, bcs=self._bcs_strong)
        M.assemble()
        return M

    def C(self) -> PETSc.Mat:
        """Damping matrix"""
        if self._mpc is None:
            C = fem.petsc.assemble_matrix(self._c_form, bcs=self._bcs_strong)
        else:
            C = dolfinx_mpc.assemble_matrix(self._c_form, self._mpc, bcs=self._bcs_strong)
        C.assemble()
        return C
        
    def K(self) -> PETSc.Mat:
        """Stiffness matrix"""
        if self._mpc is None:
            K = fem.petsc.assemble_matrix(self._k_form, bcs=self._bcs_strong)
        else:
            K = dolfinx_mpc.assemble_matrix(self._k_form, self._mpc, bcs=self._bcs_strong)
        K.assemble()
        return K

    def b(self, omega=0) -> PETSc.Vec:
        """Load vector"""
        b = self.init_b()
        self.update_b_frequencydomain(b, omega)
        return b

    def init_b(self) -> PETSc.Vec:
        """Declares a zero vector compatible with the linear form"""
        if self._mpc is None:
            return fem.petsc.create_vector(self.b_form)
        else:
            return dolfinx_mpc.assemble_vector(self.b_form, self._mpc)
        


### ### ### ### ###  ###
### Update functions ###
### ### ### ### ###  ###

    #def update_b_frequencydomain(self, b:PETSc.Vec, omega:float) -> None: #NOW SET TO EITHER METHOD BELOW IN __init__
    
    def _update_b_frequencydomain_WO_MPC(self, b:PETSc.Vec, omega:float) -> None:
        """Updates the b vector (in-place) for a given angular frequency omega"""
        #set to 0
        with b.localForm() as loc_b:
            loc_b.set(0)
        
        #fill with values
        fem.petsc.assemble_vector(b, self.b_form)
        
        #BC modifyier 
        self._omega_ufl.value=omega
        fem.petsc.apply_lifting(b, [self._a_form], [self._bcs_strong])
        
        #ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        #apply BC value
        fem.petsc.set_bc(b, self._bcs_strong)

    def _update_b_frequencydomain_WITH_MPC(self, b:PETSc.Vec, omega:float) -> None:
        """Updates the b vector (in-place) for a given angular frequency omega"""
        #set to 0
        with b.localForm() as loc_b:
            loc_b.set(0)
        
        #fill with values
        #fem.petsc.assemble_vector(b, self.b_form)
        dolfinx_mpc.assemble_vector(self.b_form, self._mpc, b)
        
        #BC modifyier 
        self._omega_ufl.value=omega
        #fem.petsc.apply_lifting(b, [self._a_form], [self._bcs_strong])
        dolfinx_mpc.apply_lifting(b, [self._a_form], [self._bcs_strong], self._mpc)
        
        #ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        #apply BC value
        fem.petsc.set_bc(b, self._bcs_strong) #not modified by dolfinx_mpc
        
