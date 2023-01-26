import numpy as np
from dolfinx import fem
from petsc4py import PETSc
import ufl

from elastodynamicsx.utils import get_functionspace_tags_marker

class BoundaryCondition():
    """
    Representation of a variety of boundary conditions
    
    Examples of use:
        #####       #####
        # Free or Clamp #
        #####       #####
        
        # Imposes sigma(u).n=0 on boundary n°1
        bc = BoundaryCondition((V, facet_tags, 1), 'Free' )
        
        # Imposes u=0 on boundaries n°1 and 2
        bc = BoundaryCondition((V, facet_tags, (1,2)), 'Clamp')
        
        # Apply BC to all boundaries when facet_tags and marker are not specified
        # or set to None
        bc = BoundaryCondition(V, 'Clamp')


        ##### #####
        # General #
        ##### #####
        
        # Imposes u=u_D on boundary n°1
        bc = BoundaryCondition((V, facet_tags, 1), 'Dirichlet', u_D)
        
        # Imposes sigma(u).n=T_N on boundary n°1
        bc = BoundaryCondition((V, facet_tags, 1), 'Neumann', T_N)
        
        # Imposes sigma(u).n=r*(u-s) on boundary n°1
        bc = BoundaryCondition((V, facet_tags, 1), 'Robin', (r, s))
    
    
        #####                    #####
        # BCs involving the velocity #
        #####                    #####
        
        # (for a scalar function_space)
        # Imposes sigma(u).n = z*v on boundary n°1, with v=du/dt. Usually z=rho*c
        bc = BoundaryCondition((V, facet_tags, 1), 'Dashpot', z)
        
        # (for a vector function_space)
        # Imposes sigma(u).n = z_N*v_N+z_T*v_T on boundary n°1,
        # with v=du/dt, v_N=(v.n)n, v_T=v-v_N
        # Usually z_N=rho*c_L and z_T=rho*c_S
        bc = BoundaryCondition((V, facet_tags, 1), 'Dashpot', (z_N, z_T))
    
    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
    """
    def __init__(self, functionspace_tags_marker, type_, values=None):
        #
        function_space, facet_tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)

        type_ = type_.lower()
        nbcomps = function_space.element.num_sub_elements #number of components if vector space, 0 if scalar space
        
        # shortcuts: Free->Neumann with value=0; Clamp->Dirichlet with value=0
        if type_ == "free":
            type_   = "neumann"
            z0s     = np.array([0]*nbcomps, dtype=PETSc.ScalarType) if nbcomps>0 else PETSc.ScalarType(0)
            values  = fem.Constant(function_space.mesh, z0s)
        elif type_ == "clamp":
            type_   = "dirichlet"
            z0s     = np.array([0]*nbcomps, dtype=PETSc.ScalarType) if nbcomps>0 else PETSc.ScalarType(0)
            values  = fem.Constant(function_space.mesh, z0s)
        
        self._type   = type_
        self._values = values
        ds = ufl.Measure("ds", domain=function_space.mesh, subdomain_data=facet_tags)(marker) #also valid if facet_tags or marker are None

        if type_ == "dirichlet":
            fdim   = function_space.mesh.topology.dim - 1
            #facets = facet_tags.find(marker) #not available on dolfinx v0.4.1
            facets = facet_tags.indices[ facet_tags.values == marker ]
            dofs   = fem.locate_dofs_topological(function_space, fdim, facets)
            self._bc = fem.dirichletbc(values, dofs, function_space) #fem.dirichletbc
        elif type_ == "neumann":
            self._bc = lambda v  : ufl.inner(values, v) * ds #Linear form
        elif type_ == "robin":
            self._bc = lambda u,v: values[0] * ufl.inner(u-values[1], v)* ds #Bilinear form
        elif type_ == 'dashpot':
            if nbcomps == 0: #scalar function space
                self._bc = lambda u_t,v: values * ufl.inner(u_t, v)* ds #Bilinear form, to be applied on du/dt
            else: #vector function space
                n = ufl.FacetNormal(function_space)
                self._bc = lambda u_t,v: ((values[0]-values[1])*ufl.inner(u_t, n)*ufl.inner(n, v) + values[1]*ufl.inner(u_t, v))* ds #Bilinear form, to be applied on du/dt
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type_))

    @property
    def bc(self):
        """The boundary condition"""
        return self._bc

    @property
    def type(self):
        return self._type

    @property
    def values(self):
        return self._values

