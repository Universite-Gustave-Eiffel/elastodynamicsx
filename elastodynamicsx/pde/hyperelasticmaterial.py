import ufl

from .material import Material
from elastodynamicsx.utils import get_functionspace_tags_marker

#https://bazaar.launchpad.net/~cbc-core/cbc.solve/main/view/head:/cbc/twist/material_model_base.py
#https://bazaar.launchpad.net/~cbc-core/cbc.solve/main/view/head:/cbc/twist/material_models.py

class HyperelasticMaterial(Material):
    """
    Base class for hyperelastic materials
    """
    
    def __init__(self, functionspace_tags_marker, rho, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            sigma: Stress function
            kwargs:
                
        """
        function_space, _, _ = get_functionspace_tags_marker(functionspace_tags_marker)
        assert function_space.element.num_sub_elements>0, 'HyperelasticMaterial requires a vector function space'
        
        super().__init__(functionspace_tags_marker, rho, is_linear=False, **kwargs)


    def P(self, u):
        """First Piola-Kirchhoff stress"""
        print("supercharge me")

    @property
    def k_CG(self):
        """Stiffness form function for a Continuous Galerkin formulation"""
        return lambda u,v: ufl.inner(self.P(u), ufl.grad(v)) * self._dx
    
    @property
    def k_DG(self) -> 'function':
        """Stiffness form function for a Discontinuous Galerkin formulation"""
        raise NotImplementedError
    
    @property
    def DG_numerical_flux(self) -> 'function':
        """Numerical flux for a Disontinuous Galerkin formulation"""
        raise NotImplementedError

        

class Murnaghan(HyperelasticMaterial):
    """
    Murnaghan's model
    see: https://en.wikipedia.org/wiki/Acoustoelastic_effect
    """

    labels = ['murnaghan']
    
    def __init__(self, functionspace_tags_marker, rho, lambda_, mu, l, m, n, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            lambda_: Lame's first parameter
            mu: Lame's second parameter (shear modulus)
            l, m, n: Murnaghan's third order elastic constants
            kwargs: Passed to HyperelasticMaterial
        """

        self._lambda = lambda_
        self._mu     = mu
        self._l      = l
        self._m      = m
        self._n      = n

        #
        super().__init__(functionspace_tags_marker, rho, **kwargs)

    def P(self, u):
        """First Piola-Kirchhoff stress"""
        # Spatial dimension
        d = len(u)

        # Identity tensor
        I = ufl.variable(ufl.Identity(d))

        # Deformation gradient
        F = ufl.variable(I + ufl.grad(u))

        # Right Cauchy-Green tensor
        C = ufl.variable(F.T * F)

        E = ufl.variable(0.5*(C - I))
        
        # convert Murnaghan's constants into Landau & Lifshitz constants
        A = self._n
        B = self._m - self._n/2
        C = self._l - B
        
        # Strain-energy function
        W = self._lambda/2*(ufl.tr(E)**2) + self._mu*ufl.tr(E*E) + C/3*(ufl.tr(E)**3) + B*(ufl.tr(E)*ufl.tr(E*E)) + A/3*(ufl.tr(E*E*E))
        
        return F*ufl.diff(W, E)
        
    @property
    def Z_N(self):
        """(WARNING: infinitesimal strain asymptotics) P-wave mechanical impedance: rho*c_L"""
        return ufl.sqrt(self.rho*(self._lambda + 2*self._mu))

    @property
    def Z_T(self):
        """(WARNING: infinitesimal strain asymptotics) S-wave mechanical impedance: rho*c_S"""
        return ufl.sqrt(self.rho*self._mu)



class DummyIsotropicElasticMaterial(HyperelasticMaterial):
    """
    A dummy implementation of an isotropic linear elastic material
    based on a HyperelasticMaterial
    """

    labels = ['dummy-isotropic']
    
    def __init__(self, functionspace_tags_marker, rho, lambda_, mu, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            lambda_: Lame's first parameter
            mu: Lame's second parameter (shear modulus)
            kwargs: Passed to HyperelasticMaterial
        """

        self._lambda = lambda_
        self._mu     = mu

        #
        super().__init__(functionspace_tags_marker, rho, **kwargs)

    def P(self, u):
        """Infinitesimal stress function; NOT the first Piola-Kirchhoff stress"""
        # Infinitesimal strain
        e = ufl.variable(0.5*(ufl.grad(u) + ufl.grad(u).T))
        
        # Strain-energy function
        W = self._lambda/2*(ufl.tr(e)**2) + self._mu*ufl.tr(e*e)
        
        return ufl.diff(W, e)

    @property
    def Z_N(self):
        """P-wave mechanical impedance: rho*c_L"""
        return ufl.sqrt(self.rho*(self._lambda + 2*self._mu))

    @property
    def Z_T(self):
        """S-wave mechanical impedance: rho*c_S"""
        return ufl.sqrt(self.rho*self._mu)



class StVenantKirchhoff(HyperelasticMaterial):
    """
    Saint Venant-Kirchhoff model
    see: https://en.wikipedia.org/wiki/Hyperelastic_material
    """

    labels = ['stvenant-kirchhoff', 'saintvenant-kirchhoff']
    
    def __init__(self, functionspace_tags_marker, rho, lambda_, mu, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            lambda_: Lame's first parameter
            mu: Lame's second parameter (shear modulus)
            kwargs: Passed to HyperelasticMaterial
        """

        self._lambda = lambda_
        self._mu     = mu

        #
        super().__init__(functionspace_tags_marker, rho, **kwargs)

    def P(self, u):
        """First Piola-Kirchhoff stress"""
        # Spatial dimension
        d = len(u)

        # Identity tensor
        I = ufl.variable(ufl.Identity(d))

        # Deformation gradient
        F = ufl.variable(I + ufl.grad(u))

        # Right Cauchy-Green tensor
        C = ufl.variable(F.T * F)

        E = ufl.variable(0.5*(C - I))
        
        # Strain-energy function
        W = self._lambda/2*(ufl.tr(E)**2) + self._mu*ufl.tr(E*E)
        
        return F*ufl.diff(W, E)

    @property
    def Z_N(self):
        """(WARNING: infinitesimal strain asymptotics) P-wave mechanical impedance: rho*c_L"""
        return ufl.sqrt(self.rho*(self._lambda + 2*self._mu))

    @property
    def Z_T(self):
        """(WARNING: infinitesimal strain asymptotics) S-wave mechanical impedance: rho*c_S"""
        return ufl.sqrt(self.rho*self._mu)


class MooneyRivlinIncompressible(HyperelasticMaterial):
    """
    Mooney-Rivlin model for an incompressible solid
    see: https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
    """

    labels = ['mooney-rivlin-incomp']
    
    def __init__(self, functionspace_tags_marker, rho, C1, C2, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            C1: first parameter
            C2: second parameter
            kwargs: Passed to HyperelasticMaterial
        """

        self._C1 = C1
        self._C2 = C2

        #
        super().__init__(functionspace_tags_marker, rho, **kwargs)

    def P(self, u):
        """First Piola-Kirchhoff stress"""
        # Spatial dimension
        d = len(u)
        assert d==3, 'The MooneyRivlinIncompressible class is only defined for 3D'

        # Identity tensor
        I = ufl.variable(ufl.Identity(d))

        # Deformation gradient
        F = ufl.variable(I + ufl.grad(u))

        # Right Cauchy-Green tensor
        C = ufl.variable(F.T * F)

        # Invariants
        I1 = ufl.tr(C)
        I2 = 0.5*(ufl.tr(C)**2 - ufl.tr(C*C))
        #I3 = ufl.det(C)
        
        # Strain-energy function
        W = self._C1*(I1 - 3) + self._C2*(I2 - 3)
        
        return ufl.diff(W, F)

    @property
    def C1(self):
        return self._C1

    @property
    def C2(self):
        return self._C2


class MooneyRivlinCompressible(HyperelasticMaterial):
    """
    Mooney-Rivlin model for a compressible solid
    see: https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
    """

    labels = ['mooney-rivlin-comp']
    
    def __init__(self, functionspace_tags_marker, rho, C10, C01, D1, **kwargs):
        """
        Args:
            functionspace_tags_marker: See Material
            rho: Density
            C10: first parameter
            C01: second parameter
            D1 : third parameter
            kwargs: Passed to HyperelasticMaterial
        """

        self._C10 = C10
        self._C01 = C01
        self._D1  = D1

        #
        super().__init__(functionspace_tags_marker, rho, **kwargs)

    def P(self, u):
        """First Piola-Kirchhoff stress"""
        # Spatial dimension
        d = len(u)
        #assert d==3, 'The MooneyRivlinIncompressible class is only defined for 3D'

        # Identity tensor
        I = ufl.variable(ufl.Identity(d))

        # Deformation gradient
        F = ufl.variable(I + ufl.grad(u))
        
        # Jacobian
        J = ufl.variable(ufl.det(F))

        # Right Cauchy-Green tensor
        C = ufl.variable(F.T * F)
        
        # Isochoric C
        Cb = ufl.variable(J**(-2.0/3.0)*C)

        # Invariants
        I1bar = ufl.tr(Cb)
        I2bar = 0.5*(ufl.tr(Cb)**2 - ufl.tr(Cb*Cb)) 
        
        # Strain-energy function
        W = self._C10*(I1bar - 3) + self._C01*(I2bar - 3) + 1/self._D1*(J-1)*(J-1)
        
        return ufl.diff(W, F)

    @property
    def C1(self):
        return self._C1

    @property
    def C2(self):
        return self._C2

# Finite-strain kinematics
# see: https://bazaar.launchpad.net/~cbc-core/cbc.solve/main/view/head:/cbc/twist/kinematics.py
#      https://jsdokken.com/dolfinx-tutorial/chapter2/hyperelasticity.html


# Lagrangian finite strain tensor, or Green-Lagrangian strain tensor
def GreenLagrangeStrain(u):
    # Spatial dimension
    d = len(u)

    # Identity tensor
    I = ufl.variable(ufl.Identity(d))

    # Deformation gradient
    F = ufl.variable(I + ufl.grad(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)
    
    return ufl.variable(0.5*(C - I))

