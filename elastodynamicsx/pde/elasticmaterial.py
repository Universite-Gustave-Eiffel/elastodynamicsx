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
        self._damping = kwargs.pop('damping', NoDamping())
        if (type(self._damping) == RayleighDamping) and (self._damping.host_material is None):
            self._damping.link_material(self)
        
        super().__init__(functionspace_tags_marker, rho, sigma, is_linear=True, **kwargs)
    
    @property
    def k(self):
        """Stiffness form function"""
        return lambda u,v: ufl.inner(self._sigma(u), self._epsilon(v)) * self._dx

    @property
    def c(self):
        """Damping form function"""
        return self._damping.c


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

