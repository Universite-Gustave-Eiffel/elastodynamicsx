import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker
from elastodynamicsx.pde import PDE


def material(functionspace_tags_marker, type_, *args, **kwargs):
    """
    Builder method that instanciates the desired material type
    
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
        aluminum = material( (function_space, cell_tags, 1), 'isotropic',
                              rho=2.8, lambda_=58, mu=26) #restricted to subdomain number 1
        aluminum = material( function_space, 'isotropic',
                             rho=2.8, lambda_=58, mu=26)  #entire domain
    """
    for Mat in all_materials:
        if type_.lower() in Mat.labels:
            return Mat(functionspace_tags_marker, *args, **kwargs)
    #
    raise TypeError('unknown material type: '+type_)



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
    
    def build(*args, **kwargs):
        """
        Same as material(*args, **kwargs).
        """
        return material(*args, **kwargs)


    ### --------------------------
    ### ------- non-static -------
    ### --------------------------
    
    def __init__(self, functionspace_tags_marker, rho, is_linear, **kwargs):
        """
        Args:
            functionspace_tags_marker: Available possibilities are
                (function_space, cell_tags, marker) #Meaning application of the
                    material in the cells whose tag correspond to marker
                function_space #In this case cell_tags=None and marker=None, meaning
                    application of the material in the entire domain
            rho:   Density
            is_linear: True for linear, False for hyperelastic
            kwargs:
                metadata: (default=None) The metadata used by the ufl measures (dx, dS)
                    If set to None, uses the PDE.default_metadata
                
        """
        self._rho = rho
        self._is_linear = is_linear
        
        function_space, cell_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)

        domain = function_space.mesh
        md     = kwargs.get('metadata', PDE.default_metadata)
        self._dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags, metadata=md)(marker) #also valid if cell_tags or marker are None
        self._dS = ufl.Measure("dS", domain=domain, subdomain_data=cell_tags, metadata=md)(marker)
        self._function_space = function_space
                
        e = self._function_space.ufl_element()
        if e.discontinuous == True: #True for discontinuous Galerkin
            print('Material: Using discontinuous elements -> DG formulation')
            self._k = self.k_DG
        else:
            print('Material: Using continuous elements -> CG formulation')
            self._k = self.k_CG
            
    
    @property
    def is_linear(self) -> bool:
        return self._is_linear
    
    @property
    def m(self) -> 'function':
        """(bilinear) mass form function"""
        return lambda u,v: self._rho* ufl.inner(u, v) * self._dx
    
    @property
    def c(self) -> 'function':
        """(bilinear) damping form function"""
        return None

    @property
    def k(self) -> 'function':
        """Stiffness form function"""
        return self._k

    @property
    def k_CG(self) -> 'function':
        """Stiffness form function for a Continuous Galerkin formulation"""
        print("supercharge me")
        
    @property
    def k_DG(self) -> 'function':
        """Stiffness form function for a Disontinuous Galerkin formulation"""
        print("supercharge me")

    @property
    def DG_numerical_flux(self) -> 'function':
        """Numerical flux for a Disontinuous Galerkin formulation"""
        print("supercharge me")

    @property
    def rho(self):
        """Density"""
        return self._rho


# -----------------------------------------------------
# Import subclasses -- must be done at the end to avoid loop imports
# -----------------------------------------------------
from .elasticmaterial import ScalarLinearMaterial, IsotropicElasticMaterial
from .hyperelasticmaterial import DummyIsotropicElasticMaterial, Murnaghan, StVenantKirchhoff, MooneyRivlinIncompressible, MooneyRivlinCompressible

all_linear_materials    = [ScalarLinearMaterial, IsotropicElasticMaterial]
all_nonlinear_materials = [DummyIsotropicElasticMaterial, Murnaghan, StVenantKirchhoff, MooneyRivlinIncompressible, MooneyRivlinCompressible]
all_materials           = all_linear_materials + all_nonlinear_materials

