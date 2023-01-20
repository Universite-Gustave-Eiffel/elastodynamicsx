from dolfinx import fem
import ufl

class PDE():
    """
    Representation of a PDE of the kind:
        M*a + C*v + K(u) = F
    as an assembly of materials and forces defined over different subdomains
    """
    
    def __init__(self, materials=[], bodyforces=[]):
        self.materials = materials
        self.bodyforces= bodyforces
    
    def add_material(self, material):
        self.materials.append(material)
    
    def add_bodyforce(self, bodyforce):
        self.bodyforces.append(bodyforce)

    @property
    def m(self):
        """(bilinear) mass form function"""
        return lambda u,v: sum([mat.m(u,v) for mat in self.materials])
    
    @property
    def c(self):
        """(bilinear) damping form function"""
        non0dampings = [mat.c for mat in self.materials if not(mat.c) is None]
        if len(non0dampings)==0:
            return None
        else:
            return lambda u,v: sum([c(u,v) for c in non0dampings])
    
    @property
    def k(self):
        """stiffness form function"""
        return lambda u,v: sum([mat.k(u,v) for mat in self.materials])
    
    @property
    def L(self):
        """linear form function"""
        if len(self.bodyforces)==0:
            return None
        else:
            return lambda v: sum([f.L(v) for f in self.bodyforces])


class BodyForce():
    """
    Representation of the rhs term (the 'F' term) of a pde such as defined in the PDE class. An instance represents a single source.
    """
    
    def __init__(self, functionspace_tags_marker, value):
        self._value = value
        if type(functionspace_tags_marker) == fem.FunctionSpace:
            function_space, cell_tags, marker = functionspace_tags_marker, None, None
        else:
            function_space, cell_tags, marker = functionspace_tags_marker

        self._dx = ufl.Measure("dx", domain=function_space.mesh, subdomain_data=cell_tags)(marker) #also valid if cell_tags or marker are None
    
    @property
    def L(self):
        return lambda v: ufl.inner(self._value, v) * self._dx

    @property
    def value(self):
        return self._value


