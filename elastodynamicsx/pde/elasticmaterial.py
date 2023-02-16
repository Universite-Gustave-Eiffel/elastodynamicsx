from dolfinx import fem
from petsc4py import PETSc
import ufl

from .material   import Material
from .kinematics import epsilon_scalar, epsilon_vector
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
        self._sigma = sigma
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


    @property
    def k_CG(self) -> 'function':
        """Stiffness form function for a Continuous Galerkin formulation"""
        return lambda u,v: ufl.inner(self._sigma(u), epsilon_vector(v)) * self._dx
    
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
        #
        super().__init__(functionspace_tags_marker, rho, sigma, **kwargs)

    @property
    def k_CG(self) -> 'function':
        """Stiffness form function for a Continuous Galerkin formulation"""
        return lambda u,v: ufl.inner(self._sigma(u), epsilon_scalar(v)) * self._dx
        
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
        sigma = lambda u: self._lambda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*self._mu*epsilon_vector(u)
        #
        super().__init__(functionspace_tags_marker, rho, sigma, **kwargs)
        
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

