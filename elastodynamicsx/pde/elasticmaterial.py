from dolfinx import fem
from petsc4py import PETSc
import ufl
import numpy as np

from .material   import Material
from .kinematics import epsilon_scalar, epsilon_vector
from elastodynamicsx.utils import get_functionspace_tags_marker

class ElasticMaterial(Material):
    """
    Base class for linear elastic materials
    """
    
    ### ### ### ###
    ### static  ###
    ### ### ### ###
    
    eq_IJ=np.zeros((6,6),dtype='int')
    cpt=0
    for i in range(6):
        for j in range(6-i):
            eq_IJ[i,i+j]=cpt
            cpt+=1
    for i in range(6):
        for j in range(i):
            eq_IJ[i,j]=eq_IJ[j,i]

    ijkm=np.zeros((21,4),dtype='int')
    ijkm[ 0,:] = 0,0,0,0 # C11 <-> 0
    ijkm[ 1,:] = 0,0,1,1 # C12 <-> 1
    ijkm[ 2,:] = 0,0,2,2 # C13 <-> 2
    ijkm[ 3,:] = 0,0,1,2 # C14 <-> 3
    ijkm[ 4,:] = 0,0,0,2 # C15 <-> 4
    ijkm[ 5,:] = 0,0,0,1 # C16 <-> 5
    ijkm[ 6,:] = 1,1,1,1 # C22 <-> 6
    ijkm[ 7,:] = 1,1,2,2 # C23 <-> 7
    ijkm[ 8,:] = 1,1,1,2 # C24 <-> 8
    ijkm[ 9,:] = 1,1,0,2 # C25 <-> 9
    ijkm[10,:] = 1,1,0,1 # C26 <-> 10
    ijkm[11,:] = 2,2,2,2 # C33 <-> 11
    ijkm[12,:] = 2,2,1,2 # C34 <-> 12
    ijkm[13,:] = 2,2,0,2 # C35 <-> 13
    ijkm[14,:] = 2,2,0,1 # C36 <-> 14
    ijkm[15,:] = 1,2,1,2 # C44 <-> 15
    ijkm[16,:] = 1,2,0,2 # C45 <-> 16
    ijkm[17,:] = 1,2,0,1 # C46 <-> 17
    ijkm[18,:] = 0,2,0,2 # C55 <-> 18
    ijkm[19,:] = 0,2,0,1 # C56 <-> 19
    ijkm[20,:] = 0,1,0,1 # C66 <-> 20

    def Cij(C_21, i:int,j:int):
        """i,j=0..5"""
        return C_21[ElasticMaterial.eq_IJ[i,j]]

    def Cijkm(C_21, i:int,j:int,k:int,m:int):
        """i,j,k,m=0..2"""
        ij = i if i==j else 6-i-j
        km = k if k==m else 6-k-m
        return ElasticMaterial.Cij(C_21, ij,km)



    ### ### ### ### ###
    ### non-static  ###
    ### ### ### ### ###
    
    def __init__(self, functionspace_tags_marker, rho, C_21, sigma, epsilon, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            C_21: list of the 21 independent elastic constants
            sigma: Stress function
            epsilon: Strain function
            kwargs:
                damping: (default=NoDamping()) An instance of a subclass of Damping
        """
        self._C_21  = C_21
        self._Cij   = ufl.as_matrix( [[ElasticMaterial.Cij(C_21, i,j) for j in range(6)] for i in range(6)] )
        self._Cijkm = ufl.as_tensor( [[[[ElasticMaterial.Cijkm(C_21, i,j,k,m) for m in range(3)] for k in range(3)] for j in range(3)] for i in range(3)] )
        #
        self._sigma = sigma
        self._epsilon = epsilon
        self._DGvariant = kwargs.pop('DGvariant', 'SIPG')
        self._damping   = kwargs.pop('damping', NoDamping())
        if (type(self._damping) == RayleighDamping) and (self._damping.host_material is None):
            self._damping.link_material(self)

        super().__init__(functionspace_tags_marker, rho, is_linear=True, **kwargs)


    def sigma(self, u):
        """Stress function: sigma(u)"""
        return self._sigma(u)
    
    def sigma_n(self, u, n):
        """Stress in the 'n' direction: (sigma(u), n)"""
        return ufl.dot(self._sigma(u), n)

    def diamond(self, v1,v2):
        """out_jk = C_ijkm * v1_i * v2_m"""
        v1 = ufl.as_vector(v1)
        v2 = ufl.as_vector(v2)
        i, j, k, m = ufl.indices(4)
        return ufl.as_matrix( self._Cijkm[i,j,k,m]*v1[i]*v2[m] , (j,k) )
    
    @property
    def k_CG(self) -> 'function':
        """Stiffness form function for a Continuous Galerkin formulation"""
        return lambda u,v: ufl.inner(self._sigma(u), self._epsilon(v)) * self._dx
    
    @property
    def k_DG(self) -> 'function':
        """Stiffness form function for a Discontinuous Galerkin formulation"""
        return lambda u,v: self.k_CG(u,v) + self.DG_numerical_flux(u,v)
        
    def select_DG_numerical_flux(self, variant='SIPG') -> 'function':
        if   variant.upper() == 'SIPG':
            return self.DG_numerical_flux_SIPG
        elif variant.upper() == 'NIPG':
            return self.DG_numerical_flux_NIPG
        elif variant.upper() == 'IIPG':
            return self.DG_numerical_flux_IIPG
        else:
            raise TypeError('Unknown DG variant ' + variant)

    @property
    def DG_numerical_flux(self) -> 'function':
        """Numerical flux for a Disontinuous Galerkin formulation"""
        return self.DG_numerical_flux_SIPG
    
    def DG_SIPG_regularization_parameter(self) -> 'dolfinx.fem.Constant':
        """Regularization parameter for the Symmetric Interior Penalty Galerkin methods (SIPG)"""
        degree = self._function_space.ufl_element().degree()
        gamma  = fem.Constant(self._function_space.mesh, PETSc.ScalarType(degree*(degree+1) + 1)) #+1 otherwise blows with elements of degree 1
        #gamma  = fem.Constant(self._function_space.mesh, PETSc.ScalarType(160)) #+1 otherwise blows with elements of degree 1
        P_mod  = self.P_modulus
        R_     = gamma*P_mod
        return R_
        
    @property
    def DG_numerical_flux_SIPG(self) -> 'function':
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n

        k_int_facets = lambda u,v: \
                       -           inner(avg(sig_n(u,n)), jump(v)        ) * dS \
                       -           inner(jump(u)        , avg(sig_n(v,n))) * dS \
                       + R_/h_avg* inner(jump(u)        , jump(v)        ) * dS

        return k_int_facets
        
    @property
    def DG_numerical_flux_NIPG(self) -> 'function':
        """WARNING: instable for elasticity"""
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n

        k_int_facets = lambda u,v: \
                       -           inner(avg(sig_n(u,n)), jump(v)        ) * dS \
                       +           inner(jump(u)        , avg(sig_n(v,n))) * dS \
                       + R_/h_avg* inner(jump(u)        , jump(v)        ) * dS

        return k_int_facets
        
    @property
    def DG_numerical_flux_IIPG(self) -> 'function':
        """WARNING: instable for elasticity"""
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n

        k_int_facets = lambda u,v: \
                       -           inner(avg(sig_n(u,n)), jump(v)        ) * dS \
                       + R_/h_avg* inner(jump(u)        , jump(v)        ) * dS

        return k_int_facets

    @property
    def c(self) -> 'function':
        """Damping form function"""
        return self._damping.c

    @property
    def P_modulus(self):
        """P-wave modulus
        = rho*c_max**2 where c_max is the highest wave velocity
        = lambda_ + 2*mu for isotropic materials
        = mu for scalar materials"""
        print('supercharge me')



class ScalarLinearMaterial(ElasticMaterial):
    """
    A scalar linear material (e.g. 2D-SH or fluid)
    """
    
    labels = ['scalar', '2d-sh', 'fluid']
    
    def __init__(self, functionspace_tags_marker, rho, mu, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            mu : Shear modulus (or rho*c**2 for a fluid, with c the sound velocity)
            kwargs: Passed to ElasticMaterial
        """
        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements==0, 'ScalarLinearMaterial requires a scalar function space'
        
        self._mu = mu
        sigma = lambda u: self._mu*epsilon_scalar(u)
        
        C11 = C22 = C33 = mu
        C12 = C13 = C23 = mu
        C_21 = [0]*21
        C_21[0] , C_21[6] , C_21[11] = C11, C22, C33
        C_21[1] , C_21[2] , C_21[7]  = C12, C13, C23
        #
        super().__init__(functionspace_tags_marker, rho, C_21, sigma, epsilon_scalar, **kwargs)

    @property
    def mu(self):
        """The shear modulus"""
        return self._mu

    @property
    def Z(self):
        """The Mechanical impedance: rho*c"""
        return ufl.sqrt(self.rho*self.mu)

    @property
    def P_modulus(self):
        return self.mu

    @property
    def DG_numerical_flux_SIPG(self) -> 'function':
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma
        
        k_int_facets = lambda u,v: \
                       -           inner(avg(sigma(u)), jump(v,n)    ) * dS \
                       -           inner(jump(u,n)    , avg(sigma(v))) * dS \
                       + R_/h_avg* inner(jump(u)      , jump(v)      ) * dS

        return k_int_facets

    @property
    def DG_numerical_flux_NIPG(self) -> 'function':
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma
        
        k_int_facets = lambda u,v: \
                       -           inner(avg(sigma(u)), jump(v,n)    ) * dS \
                       +           inner(jump(u,n)    , avg(sigma(v))) * dS \
                       + R_/h_avg* inner(jump(u)      , jump(v)      ) * dS

        return k_int_facets

    @property
    def DG_numerical_flux_IIPG(self) -> 'function':
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_SIPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma
        
        k_int_facets = lambda u,v: \
                       -           inner(avg(sigma(u)), jump(v,n)    ) * dS \
                       + R_/h_avg* inner(jump(u)      , jump(v)      ) * dS

        return k_int_facets
        


class IsotropicElasticMaterial(ElasticMaterial):
    """
    An isotropic linear elastic material
    """

    labels = ['isotropic']
    
    def __init__(self, functionspace_tags_marker, rho, lambda_, mu, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            lambda_: Lame's first parameter
            mu: Lame's second parameter (shear modulus)
            kwargs: Passed to ElasticMaterial
        """
        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements>0, 'IsotropicElasticMaterial requires a vector function space'
        
        self._lambda = lambda_
        self._mu     = mu
        
        if function_space.mesh.geometry.dim == 1: #1D
            di0     = ufl.as_vector([i==0 for i in range(function_space.num_sub_spaces)])
            epsilon = lambda u: 0.5*(u.dx(0) + di0*u[0].dx(0))
            sigma   = lambda u: self._lambda * u[0].dx(0) * di0 + 2*self._mu*u.dx(0)
        else: #2D or 3D
            epsilon = epsilon_vector
            sigma   = lambda u: self._lambda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*self._mu*epsilon(u)
        
        C11 = C22 = C33 = lambda_ + 2*mu
        C12 = C13 = C23 = lambda_
        C44 = C55 = C66 = mu
        C_21 = [0]*21
        C_21[0] , C_21[6] , C_21[11] = C11, C22, C33
        C_21[1] , C_21[2] , C_21[7]  = C12, C13, C23
        C_21[15], C_21[18], C_21[20] = C44, C55, C66
        #
        super().__init__(functionspace_tags_marker, rho, C_21, sigma, epsilon, **kwargs)

    @property
    def lambda_(self):
        """Lame's first parameter"""
        return self._lambda

    @property
    def mu(self):
        """Lame's second parameter (shear modulus)"""
        return self._mu

    @property
    def P_modulus(self):
        return self._lambda + 2*self._mu

    @property
    def Z_N(self):
        """P-wave mechanical impedance: rho*c_L"""
        return ufl.sqrt(self.rho*(self._lambda + 2*self._mu))

    @property
    def Z_T(self):
        """S-wave mechanical impedance: rho*c_S"""
        return ufl.sqrt(self.rho*self._mu)



class Damping():
    """Dummy base class for damping laws"""

    def build(type_, *args):
        """
        Convenience static method that instanciates the desired damping law
        
        Args:
            type_: Available options are:
                'none'
                'rayleigh'
            args: passed to the required damping law
        """
        if   type_.lower() == 'none':     return NoDamping()
        elif type_.lower() == 'rayleigh': return RayleighDamping(*args)
        else:
            raise TypeError("Unknown damping law: {0:s}".format(type_))

    @property
    def c(self): print('supercharge me')

class NoDamping(Damping):
    """no damping"""
        
    @property
    def c(self):
        """The damping form"""
        return None

class RayleighDamping(Damping):
    """Rayleigh damping law: c(u,v) = eta_m*m(u,v) + eta_k(u,v)"""

    def __init__(self, eta_m, eta_k):
        """
        Args:
            eta_m: Parameter of the mass-matrix part of the damping
            eta_k: Parameter of the stiffness-matrix part of the damping
        """
        self._eta_m = eta_m
        self._eta_k = eta_k
        self._material = None

    @property
    def eta_m(self):
        """Parameter of the mass-matrix part of the damping"""
        return self._eta_m

    @property
    def eta_k(self):
        """Parameter of the stiffness-matrix part of the damping"""
        return self._eta_k

    @property
    def c(self):
        """The damping form"""
        return lambda u,v: self.eta_m*self._material.m(u,v) + self.eta_k*self._material.k(u,v)
    
    @property
    def host_material(self):
        """Host material from whom the mass and stiffness matrices are copied"""
        return self._material

    def link_material(self, host_material):
        """
        Connects to a host material from whom the mass and stiffness matrices
        will be copied
        """
        self._material = host_material

