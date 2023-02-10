from petsc4py import PETSc
from dolfinx import fem
import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker
from . import BoundaryCondition

class PDE():
    """
    Representation of a PDE of the kind:
    
        M*a + C*v + K(u) = b
    
    as an assembly of materials and forces defined over different subdomains
    """
    
    def __init__(self, materials=[], bodyforces=[]):
        self.materials = materials
        self.bodyforces= bodyforces
    
    def add_material(self, material):
        self.materials.append(material)
    
    def add_bodyforce(self, bodyforce):
        self.bodyforces.append(bodyforce)

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
    def L(self) -> 'function':
        """Linear form function"""
        if len(self.bodyforces)==0:
            return None
        else:
            return lambda v: sum([f.L(v) for f in self.bodyforces])



class PDE2(PDE):
    def __init__(self, function_space, materials=[], bodyforces=[], bcs=[]):
        super().__init__(materials, bodyforces)
        
        self._function_space = function_space
        self._u, self._v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)
        
        self._bcs_weak   = []
        self._bcs_strong = []
        for bc in bcs:
            if issubclass(type(bc), fem.DirichletBCMetaClass):
                self._bcs_strong.append(bc)
            elif type(bc) == BoundaryCondition:
                if issubclass(type(bc.bc), fem.DirichletBCMetaClass):
                    self._bcs_strong.append(bc.bc)
                else:
                    self._bcs_weak.append(bc)
            else:
                raise TypeError("Unsupported boundary condition"+str(type(bc)))
        
        self._omega_ufl = fem.Constant(function_space, PETSc.ScalarType(0))
        self._compile_M_C_K_L()
    
    
    def _compile_M_C_K_L(self):
        u, v = self._u, self._v
        zero = fem.Constant(self._function_space, PETSc.ScalarType(0.))
        
        #interior
        m = self.m(u,v)
        c = self.c(u,v) if not(self.c is None) else zero*ufl.inner(u,v)*ufl.dx
        k = self.k(u,v)
        L = self.L(v)   if not(self.L is None) else zero*ufl.conj(v)*ufl.dx
        
        #boundaries
        for bc in self._bcs_weak:
            if bc.type == 'neumann':
                L += bc(v)
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

        ##Mat_lhs = -w*w*_M_ + 1J*w*_C_ + _K_
        w = self._omega_ufl
        self._a_form = fem.form(-w*w*m + 1J*w*c + k)
    
    
    @property
    def m_form(self):
        """Compiled mass bilinear form"""
        return self._m_form
        
    @property
    def c_form(self):
        """Compiled damping bilinear form"""
        return self._c_form
        
    @property
    def k_form(self):
        """Compiled stiffness bilinear form"""
        return self._k_form
    
    @property
    def b_form(self):
        """Compiled linear form"""
        return self._b_form
    
    def M(self) -> PETSc.Mat:
        """Mass matrix"""
        M = fem.petsc.assemble_matrix(self._m_form, bcs=self._bcs_strong)
        M.assemble()
        return M

    def C(self) -> PETSc.Mat:
        """Damping matrix"""
        #return 0*self.M()
        C = fem.petsc.assemble_matrix(self._c_form, bcs=self._bcs_strong)
        C.assemble()
        return C
        
    def K(self) -> PETSc.Mat:
        """Stiffness matrix"""
        K = fem.petsc.assemble_matrix(self._k_form, bcs=self._bcs_strong)
        K.assemble()
        return K

    def b(self, omega=0) -> PETSc.Vec:
        """Load vector"""
        b = self.init_b()
        self.update_b_frequencydomain(b, omega)
        return b

    def init_b(self) -> PETSc.Vec:
        """Declares a zero vector compatible with the linear form"""
        return fem.petsc.create_vector(self.b_form)
        
    def update_b_frequencydomain(self, b:PETSc.Vec, omega:float) -> None:
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

        
