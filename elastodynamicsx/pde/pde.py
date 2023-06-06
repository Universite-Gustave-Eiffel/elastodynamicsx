# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

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

    ### ### ### ###
    ### static  ###
    ### ### ### ###

    # The default metadata used by all measures (dx, ds, dS, ...) in the classes
    # of the pde package: Material, BoundaryCondition, BodyForce
    # Example: Spectral Element Method with GLL elements of degree 6
    # >>> from elastodynamicsx.pde import PDE
    # >>> from elastodynamicsx.utils import spectral_quadrature
    # >>> specmd = spectral_quadrature("GLL", 6)
    # >>> PDE.default_metadata = specmd
    #
    default_metadata = None

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



    ### ### ### ### ###
    ### non-static  ###
    ### ### ### ### ###

    def __init__(self, function_space:'dolfinx.fem.FunctionSpace', materials:list, **kwargs):
        """
        Args:
            functions_space
            materials: a list of pde.Material instances
        kwargs:
            bodyforces: (default=[]) a list of pde.BodyForce instances
            bcs: (default=[]) a list of fem.DirichletBCMetaClass and/or pde.BoundaryCondition instances
        """
        self._function_space = function_space
        self.materials = materials
        self.bodyforces= kwargs.get('bodyforces', [])
        self.bcs = kwargs.get('bcs', [])
        self._u, self._v = ufl.TrialFunction(function_space), ufl.TestFunction(function_space)

        # Declare stuff without building
        self._mpc = None
        self._m_form = None
        self._c_form = None
        self._k_form = None
        self._b_form = None
        self._k1_form = None
        self._k2_form = None
        self._k3_form = None

        # Sort boundary conditions
        self._bcs_weak   = BoundaryCondition.get_weak_BCs(self.bcs)      # custom weak BSs, instances of BoundaryCondition
        self._bcs_strong = BoundaryCondition.get_dirichlet_BCs(self.bcs) # dolfinx.fem.DirichletBCMetaClass
        self._bcs_mpc    = BoundaryCondition.get_mpc_BCs(self.bcs)       # instances of BoundaryCondition used to add multi-point constraints
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
        else:
            print('non-linear PDE')


    def _build_mpc(self) -> None:
        """Required for handling multi-point constraints (e.g. periodic BC)"""
        if dolfinx_mpc is None: # if import error
            return
        if len(self._bcs_mpc)==0: # if there is no need to use MPC. Note that MPC slows down the runs even if there is no constraint -> better avoid using it if not needed
            return
        self._mpc = PDE.build_mpc(self._function_space, self._bcs_strong + self._bcs_mpc)

    def _compile_M(self) -> None:
        u, v = self._u, self._v
        m = self.m(u,v)
        self._m_form = fem.form(m)


    def _compile_C_K_b(self) -> None:
        """Required for frequency domain or eigenvalue problems"""
        u, v = self._u, self._v

        zero = fem.Constant(self._function_space, PETSc.ScalarType(0.))
        vzero = zero
        if v.ufl_function_space().num_sub_spaces != 0:  # VectorFunctionSpace
            vzero = fem.Constant(self._function_space, PETSc.ScalarType([0.] * len(v)))

        # Interior
        k = self.k(u,v)

        # Retrieve the integral measures in k to build compatible default zero forms 'c' and 'L'
        measures = [ufl.Measure("dx",
                                domain=self._function_space.mesh,
                                subdomain_data=cint.subdomain_data(),
                                metadata=cint.metadata())(cint.subdomain_id()) for cint in k.integrals()]

        c = self.c(u,v) if not(self.c is None) else sum([zero*ufl.inner(u,v) * dx for dx in measures])
        L = self.L(v)   if not(self.L is None) else sum([ufl.inner(vzero,v) * dx for dx in measures])

        # Boundaries
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

        self._c_form = fem.form(c)
        self._k_form = fem.form(k)
        self._b_form = fem.form(L)

        # Executes the following only if using complex numbers
        if np.issubdtype(PETSc.ScalarType, np.complexfloating):
            ##Mat_lhs = -w*w*_M_ + 1J*w*_C_ + _K_
            m = self.m(u,v)
            w = self._omega_ufl
            self._a_form = fem.form(-w*w*m + 1J*w*c + k)


    def _compile_K1_K2_K3(self) -> None:
        """Required for waveguide problems"""
        u, v = self._u, self._v

        assert self._function_space.element.basix_element.discontinuous == False, 'K1, K2, K3 are not implemented for a DG formulation'

        # Interior
        k1 = self.k1(u,v)
        k2 = self.k2(u,v)
        k3 = self.k3(u,v)

        # Boundaries
        for bc in self._bcs_weak:
            if bc.type == 'neumann':
                pass #ignores
            elif bc.type == 'robin':
                F_bc = bc.bc(u,v)
                print('Robin BC: TODO')
                raise NotImplementedError
                #k += ufl.lhs(F_bc) #TODO
                #L += ufl.rhs(F_bc) #ignores right hand side
            elif bc.type == 'dashpot':
                print('Dashpot BC: TODO')
                raise NotImplementedError
                #c += bc.bc(u,v)
            else:
                raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))

        self._k1_form = fem.form(k1)
        self._k2_form = fem.form(k2)
        self._k3_form = fem.form(k3)



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
    def k1(self) -> 'function':
        """(bilinear) k1 stiffness form function (waveguide problems)"""
        return lambda u,v: sum([mat.k1_CG(u,v) for mat in self.materials])


    @property
    def k2(self) -> 'function':
        """(bilinear) k2 stiffness form function (waveguide problems)"""
        return lambda u,v: sum([mat.k2_CG(u,v) for mat in self.materials])


    @property
    def k3(self) -> 'function':
        """(bilinear) k3 stiffness form function (waveguide problems)"""
        return lambda u,v: sum([mat.k3_CG(u,v) for mat in self.materials])


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
        if self._m_form is None:
            self._compile_M()
        if self._mpc is None:
            M = fem.petsc.assemble_matrix(self._m_form, bcs=self._bcs_strong)
        else:
            M = dolfinx_mpc.assemble_matrix(self._m_form, self._mpc, bcs=self._bcs_strong)
        M.assemble()
        return M


    def C(self) -> PETSc.Mat:
        """Damping matrix"""
        if self._c_form is None:
            self._compile_C_K_b()
        if self._mpc is None:
            C = fem.petsc.assemble_matrix(self._c_form, bcs=self._bcs_strong)
        else:
            C = dolfinx_mpc.assemble_matrix(self._c_form, self._mpc, bcs=self._bcs_strong)
        C.assemble()
        return C


    def K(self) -> PETSc.Mat:
        """Stiffness matrix"""
        if self._k_form is None:
            self._compile_C_K_b()
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
        if self._b_form is None:
            self._compile_C_K_b()
        if self._mpc is None:
            return fem.petsc.create_vector(self._b_form)
        else:
            return dolfinx_mpc.assemble_vector(self._b_form, self._mpc)


    def K1(self) -> PETSc.Mat:
        """K1 stiffness matrix (waveguide problems)"""
        if self._k1_form is None:
            self._compile_K1_K2_K3()
        if self._mpc is None:
            K1 = fem.petsc.assemble_matrix(self._k1_form, bcs=self._bcs_strong)
        else:
            K1 = dolfinx_mpc.assemble_matrix(self._k1_form, self._mpc, bcs=self._bcs_strong)
        K1.assemble()
        return K1


    def K2(self) -> PETSc.Mat:
        """K2 stiffness matrix (waveguide problems)"""
        if self._function_space.mesh.geometry.dim == 3: #special case: K1=K, K2=K3=0
            return None
        if self._k2_form is None:
            self._compile_K1_K2_K3()
        if self._mpc is None:
            K2 = fem.petsc.assemble_matrix(self._k2_form, bcs=self._bcs_strong)
        else:
            K2 = dolfinx_mpc.assemble_matrix(self._k2_form, self._mpc, bcs=self._bcs_strong)
        K2.assemble()
        return K2


    def K3(self) -> PETSc.Mat:
        """K3 stiffness matrix (waveguide problems)"""
        if self._function_space.mesh.geometry.dim == 3: #special case: K1=K, K2=K3=0
            return None
        if self._k3_form is None:
            self._compile_K1_K2_K3()
        if self._mpc is None:
            K3 = fem.petsc.assemble_matrix(self._k3_form, bcs=self._bcs_strong)
        else:
            K3 = dolfinx_mpc.assemble_matrix(self._k3_form, self._mpc, bcs=self._bcs_strong)
        K3.assemble()
        return K3


### ### ### ### ###  ###
### Update functions ###
### ### ### ### ###  ###

    #def update_b_frequencydomain(self, b:PETSc.Vec, omega:float) -> None: #NOW SET TO EITHER METHOD BELOW IN __init__

    def _update_b_frequencydomain_WO_MPC(self, b:PETSc.Vec, omega:float) -> None:
        """Updates the b vector (in-place) for a given angular frequency omega"""
        # set to 0
        with b.localForm() as loc_b:
            loc_b.set(0)

        # fill with values
        fem.petsc.assemble_vector(b, self.b_form)

        # BC modifyier 
        self._omega_ufl.value=omega
        fem.petsc.apply_lifting(b, [self._a_form], [self._bcs_strong])

        # ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # apply BC value
        fem.petsc.set_bc(b, self._bcs_strong)


    def _update_b_frequencydomain_WITH_MPC(self, b:PETSc.Vec, omega:float) -> None:
        """Updates the b vector (in-place) for a given angular frequency omega"""
        # set to 0
        with b.localForm() as loc_b:
            loc_b.set(0)

        # fill with values
        #fem.petsc.assemble_vector(b, self.b_form)
        dolfinx_mpc.assemble_vector(self.b_form, self._mpc, b)

        # BC modifyier 
        self._omega_ufl.value=omega
        #fem.petsc.apply_lifting(b, [self._a_form], [self._bcs_strong])
        dolfinx_mpc.apply_lifting(b, [self._a_form], [self._bcs_strong], self._mpc)

        # ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        #apply BC value
        fem.petsc.set_bc(b, self._bcs_strong) #not modified by dolfinx_mpc
