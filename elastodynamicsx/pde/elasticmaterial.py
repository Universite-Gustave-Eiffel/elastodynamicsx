from dolfinx import fem
from petsc4py import PETSc
import ufl

from .material import Material, epsilon_scalar, epsilon_vector
from elastodynamicsx.utils import get_functionspace_tags_marker

class ElasticMaterial(Material):
    """
    Base class for linear elastic materials
    """
    
    def __init__(self, functionspace_tags_marker, rho, sigma, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            sigma: Stress function
            kwargs:
                damping: (default=NoDamping()) An instance of a subclass of Damping
        """
        self._DGvariant = kwargs.pop('DGvariant', 'SIPG')
        self._damping   = kwargs.pop('damping', NoDamping())
        if (type(self._damping) == RayleighDamping) and (self._damping.host_material is None):
            self._damping.link_material(self)
        
        super().__init__(functionspace_tags_marker, rho, sigma, is_linear=True, **kwargs)
    
    @property
    def k(self):
        """Stiffness form function"""
        e = self._function_space.ufl_element()
        if e.is_cellwise_constant() == True:
            return self.k_CG
        else:
            return self.k_DG(self._DGvariant)

    @property
    def k_CG(self):
        """Stiffness form function for a Continuous Galerkin formulation"""
        return lambda u,v: ufl.inner(self._sigma(u), self._epsilon(v)) * self._dx
    
    def DG_IPG_regularization_parameter(self):
        degree = self._function_space.ufl_element().degree()
        gamma  = fem.Constant(self._function_space.mesh, PETSc.ScalarType(degree*(degree+1) + 1)) #+1 otherwise blows with elements of degree 1
        P_mod  = self.P_modulus
        R_     = gamma*P_mod
        return R_
    
    def k_DG(self, variant):
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_IPG_regularization_parameter()
        dS = self._dS
        sig_n = self.sigma_n
        
        if   variant.upper() == 'SIPG':
            k_int_facets = lambda u,v: \
                           -           inner(avg(sig_n(u,n)), jump(v)        ) * dS \
                           -           inner(jump(u)        , avg(sig_n(v,n))) * dS \
                           + R_/h_avg* inner(jump(u)        , jump(v)        ) * dS
        elif variant.upper() == 'NIPG':
            k_int_facets = lambda u,v: \
                           -           inner(avg(sig_n(u,n)), jump(v)        ) * dS \
                           +           inner(jump(u)        , avg(sig_n(v,n))) * dS \
                           + R_/h_avg* inner(jump(u)        , jump(v)        ) * dS
        elif variant.upper() == 'IIPG':
            k_int_facets = lambda u,v: \
                           -           inner(avg(sig_n(u,n)), jump(v)) * dS \
                           + R_/h_avg* inner(jump(u)        , jump(v)) * dS
        else:
            raise TypeError('Unknown DG variant ' + variant)
        
        return lambda u,v: self.k_CG(u,v) + k_int_facets(u,v)

    @property
    def c(self):
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
        #
        super().__init__(functionspace_tags_marker, rho, sigma, epsilon=epsilon_scalar, **kwargs)

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

    def k_DG(self, variant):
        inner, avg, jump = ufl.inner, ufl.avg, ufl.jump
        V = self._function_space
        n = ufl.FacetNormal(V)
        h = ufl.MinCellEdgeLength(V) #works!
        h_avg  = (h('+') + h('-'))/2.0
        R_ = self.DG_IPG_regularization_parameter()
        dS = self._dS
        sigma = self.sigma
        
        if   variant.upper() == 'SIPG':
            k_int_facets = lambda u,v: \
                           -           inner(avg(sigma(u)), jump(v,n)    ) * dS \
                           -           inner(jump(u,n)    , avg(sigma(v))) * dS \
                           + R_/h_avg* inner(jump(u)      , jump(v)      ) * dS
        elif variant.upper() == 'NIPG':
            k_int_facets = lambda u,v: \
                           -           inner(avg(sigma(u)), jump(v,n)    ) * dS \
                           +           inner(jump(u,n)    , avg(sigma(v))) * dS \
                           + R_/h_avg* inner(jump(u)      , jump(v)      ) * dS
        elif variant.upper() == 'IIPG':
            k_int_facets = lambda u,v: \
                           -           inner(avg(sigma(u)), jump(v,n)) * dS \
                           + R_/h_avg* inner(jump(u)      , jump(v)  ) * dS
        else:
            raise TypeError('Unknown DG variant ' + variant)
        
        return lambda u,v: self.k_CG(u,v) + k_int_facets(u,v)



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
        sigma = lambda u: self._lambda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*self._mu*epsilon_vector(u)
        #
        super().__init__(functionspace_tags_marker, rho, sigma, epsilon=epsilon_vector, **kwargs)
        
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
        return self.lambda_ + 2*self.mu

    @property
    def Z_N(self):
        """P-wave mechanical impedance: rho*c_L"""
        return ufl.sqrt(self.rho*(self.lambda_ + 2*self.mu))

    @property
    def Z_T(self):
        """S-wave mechanical impedance: rho*c_S"""
        return ufl.sqrt(self.rho*self.mu)



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

