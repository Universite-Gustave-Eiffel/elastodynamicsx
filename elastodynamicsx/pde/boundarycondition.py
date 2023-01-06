from dolfinx import fem
from petsc4py import PETSc
import ufl
import numpy as np

class BoundaryCondition():
    """
    Representation of a variety of boundary conditions
    
    # Shortcuts
    Free     : bc = BoundaryCondition(V, facet_tags, 'Free' , 1)             <- imposes sigma(u).n=0 on boundary n°1
    Clamp    : bc = BoundaryCondition(V, facet_tags, 'Clamp', 1)             <- imposes u=0 on boundary n°1

    # General
    Dirichlet: bc = BoundaryCondition(V, facet_tags, 'Dirichlet', 1, u_D)    <- imposes u=u_D on boundary n°1
    Neumann  : bc = BoundaryCondition(V, facet_tags, 'Neumann'  , 1, T_N)    <- imposes sigma(u).n=T_N on boundary n°1
    Robin    : bc = BoundaryCondition(V, facet_tags, 'Robin'    , 1, (r, s)) <- imposes sigma(u).n=r*(u-s) on boundary n°1
    
    # Specific to time domain:
    Dashpot  : bc = BoundaryCondition(V, facet_tags, 'td-Dashpot', 1, z)          <- (scalar) imposes sigma(u).n=z*v on boundary n°1, with v=du/dt. Usually z=rho*c
    Dashpot  : bc = BoundaryCondition(V, facet_tags, 'td-Dashpot', 1, (z_N, z_T)) <- (vector) imposes sigma(u).n=z_N*v_N+z_T*v_T on boundary n°1, with v=du/dt, v_N=(v.n)n, v_T=v-v_N
    
    Adapted from https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
    """
    def __init__(self, function_space, facet_tags, type_, marker, values=None):
        type_ = type_.lower()
        nbcomps = function_space.element.num_sub_elements #number of components if vector space, 0 if scalar space
        #
        if type_ == "free":
            type_   = "neumann"
            z0s     = np.array([0]*nbcomps, dtype=PETSc.ScalarType) if nbcomps>0 else PETSc.ScalarType(0)
            values  = fem.Constant(function_space.mesh, z0s)
        elif type_ == "clamp":
            type_   = "dirichlet"
            z0s     = np.array([0]*nbcomps, dtype=PETSc.ScalarType) if nbcomps>0 else PETSc.ScalarType(0)
            values  = fem.Constant(function_space.mesh, z0s)
        #
        self._type   = type_
        self._values = values
        ds = ufl.Measure("ds", domain=function_space.mesh, subdomain_data=facet_tags)
        #
        if type_ == "dirichlet":
            fdim   = function_space.mesh.topology.dim - 1
            #facets = facet_tags.find(marker) #not available on dolfinx v0.4.1
            facets = facet_tags.indices[ facet_tags.values == marker ]
            dofs   = fem.locate_dofs_topological(function_space, fdim, facets)
            self._bc = fem.dirichletbc(values, dofs, function_space) #fem.dirichletbc
        elif type_ == "neumann":
            self._bc = lambda v  : ufl.inner(values, v) * ds(marker) #Linear form
        elif type_ == "robin":
            self._bc = lambda u,v: values[0] * ufl.inner(u-values[1], v)* ds(marker) #Bilinear form
        elif type_ == 'td-dashpot':
            if nbcomps == 0: #scalar function space
                self._bc = lambda u_t,v: values * ufl.inner(u_t, v)* ds(marker) #Bilinear form, to be applied on du/dt
            else: #vector function space
                n = ufl.FacetNormal(function_space)
                self._bc = lambda u_t,v: ((values[0]-values[1])*ufl.inner(u_t, n)*ufl.inner(n, v) + values[1]*ufl.inner(u_t, v))* ds(marker) #Bilinear form, to be applied on du/dt
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type_))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

    @property
    def values(self):
        return self._values

