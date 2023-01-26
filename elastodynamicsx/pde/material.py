from dolfinx import fem
import ufl

from elastodynamicsx.plot  import get_3D_array_from_FEFunction
from elastodynamicsx.utils import get_functionspace_tags_marker

class Material():
    """
    Base class for a representation of a PDE of the kind:
    
        M*a + C*v + K(u) = 0
    
    i.e. the lhs of a pde such as defined in the PDE class. An instance represents
    a single (possibly arbitrarily space-dependent) material.
    """
    
    
    ### --------------------------
    ### --------- static ---------
    ### --------------------------

    labels = ['supercharge me']
    
    def build(functionspace_tags_marker, type_, *args, **kwargs):
        """
        Convenience static method that instanciates the desired material
        
        Args:
            functionspace_tags_marker: Available possibilities are
                (function_space, cell_tags, marker) #Meaning application of the
                    material in the cells whose tag correspond to marker
                function_space #In this case cell_tags=None and marker=None, meaning
                    application of the material in the entire domain
            type_: Available options are:
                     'scalar'
                     'isotropic'
            args:   Passed to the required material
            kwargs: Passed to the required material
        
        Returns:
            An instance of the desired material
        
        Examples of use:
            aluminum = Material.build( (function_space, cell_tags, 1), 'isotropic',
                                        rho=2.8, lambda_=58, mu=26) #restricted to subdomain number 1
            aluminum = Material.build( function_space, 'isotropic',
                                       rho=2.8, lambda_=58, mu=26)  #entire domain
        """
        allMaterials = (ScalarLinearMaterial, IsotropicElasticMaterial)
        for Mat in allMaterials:
          if type_.lower() in Mat.labels: return Mat(functionspace_tags_marker, *args, **kwargs)
        #
        raise TypeError('unknown material type: '+type_)


    ### --------------------------
    ### ------- non-static -------
    ### --------------------------
    
    def __init__(self, functionspace_tags_marker, rho, sigma, is_linear, **kwargs):
        self._rho = rho
        self._sigma = sigma
        self._epsilon = kwargs.get('epsilon', epsilon_vector)
        
        function_space, cell_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)

        self._dx = ufl.Measure("dx", domain=function_space.mesh, subdomain_data=cell_tags)(marker) #also valid if cell_tags or marker are None
    
    @property
    def is_linear(self):
        return self._is_linear
    
    @property
    def m(self):
        """(bilinear) mass form function"""
        return lambda u,v: self._rho* ufl.inner(u, v) * self._dx
    
    @property
    def c(self):
        """(bilinear) damping form function"""
        return None
    
    @property
    def k(self):
        """stiffness form function"""
        print("supercharge me")

    @property
    def rho(self):
        return self._rho



#
def epsilon_vector(u): return ufl.sym(ufl.grad(u))
def epsilon_scalar(u): return ufl.nabla_grad(u)

# -----------------------------------------------------
# Import subclasses -- must be done at the end to avoid loop imports
# -----------------------------------------------------
from .elasticmaterial import ScalarLinearMaterial, IsotropicElasticMaterial

