import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker


class BodyForce():
    """
    Representation of the rhs term (the 'F' term) of a pde such as defined
    in the PDE class. An instance represents a single source.
    """
    
    def __init__(self, functionspace_tags_marker, value):
        self._value = value
        function_space, cell_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)
        self._dx = ufl.Measure("dx", domain=function_space.mesh, subdomain_data=cell_tags)(marker)
    
    @property
    def L(self):
        """The linear form function"""
        return lambda v: ufl.inner(self._value, v) * self._dx

    @property
    def value(self):
        return self._value


