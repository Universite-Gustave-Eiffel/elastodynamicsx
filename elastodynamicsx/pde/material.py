from dolfinx import fem
import ufl

class Material():
    """
    Base class for a representation of a PDE of the kind:
        M*a + C*v + K(u) = 0
    i.e. the lhs of a pde such as defined in the PDE class. An instance represents a single (possibly arbitrarily space-dependent) material.
    """
    
    def __init__(self, functionspace_tags_marker, rho, sigma, is_linear, **kwargs):
        self._rho = rho
        self._sigma = sigma
        self._epsilon = kwargs.get('epsilon', epsilon_vector)
        
        if type(functionspace_tags_marker) == fem.FunctionSpace:
            function_space, cell_tags, marker = functionspace_tags_marker, None, None
        else:
            function_space, cell_tags, marker = functionspace_tags_marker

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
