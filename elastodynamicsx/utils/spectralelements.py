import basix
from dolfinx.mesh import CellType

#https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_lagrange_variants.py
#https://github.com/FEniCS/dolfinx/issues/2537

### ### ###  ###
### Elements ###
### ### ###  ###

_cell_type_mesh2basix = {CellType.point        : basix.CellType.point, \
                         CellType.interval     : basix.CellType.interval, \
                         CellType.triangle     : basix.CellType.triangle, \
                         CellType.quadrilateral: basix.CellType.quadrilateral, \
                         CellType.tetrahedron  : basix.CellType.tetrahedron, \
                         CellType.prism        : basix.CellType.prism, \
                         CellType.pyramid      : basix.CellType.pyramid, \
                         CellType.hexahedron   : basix.CellType.hexahedron \
                         }
def _suitable_cell_type_format(cell_type):
    if   type(cell_type) == basix.CellType:
        return cell_type
    elif type(cell_type) == CellType:
        return _cell_type_mesh2basix[cell_type]
    elif type(cell_type) == str:
        return basix.cell.string_to_type(cell_type)
    else:
        raise TypeError("Unknown cell type: {0:s}".format(cell_type))

def GLL_element(cell_type, degree:int) -> basix.ufl_wrapper.BasixElement:
    """Element defined using the Gauss-Lobatto-Legendre points"""
    cell_type = _suitable_cell_type_format(cell_type)
    element   = basix.create_element( basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.gll_warped )
    return basix.ufl_wrapper.BasixElement(element)

def GL_element(cell_type, degree:int) -> basix.ufl_wrapper.BasixElement:
    """(discontinuous) Element defined using the Gauss-Legendre points"""
    cell_type = _suitable_cell_type_format(cell_type)
    element   = basix.create_element( basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.gl_warped, True )
    return basix.ufl_wrapper.BasixElement(element)

def Legendre_element(cell_type, degree:int) -> basix.ufl_wrapper.BasixElement:
    """(discontinuous) Element whose basis functions are the orthonormal Legendre polynomials"""
    cell_type = _suitable_cell_type_format(cell_type)
    element   = basix.create_element( basix.ElementFamily.P, cell_type, degree, basix.LagrangeVariant.legendre, True )
    return basix.ufl_wrapper.BasixElement(element)



### ### ### ### ###
### Quadrature  ###
### ### ### ### ###

_qd = {"2": 3, "3": 4, "4": 6, "5": 8, "6": 10, "7": 12, "8": 14, "9": 16, "10": 18}

def GLL_quadrature(degree:int) -> dict:
    return {"quadrature_rule": "GLL", "quadrature_degree": _qd[str(degree)]}

def GL_quadrature(degree:int) -> dict:
    return {"quadrature_rule": "GL",  "quadrature_degree": _qd[str(degree)]}

def Legendre_quadrature(degree:int) -> dict: #TODO
    raise NotImplementedError
    #return {"quadrature_degree": _qd[str(degree)]}



### ### ### ### ###
### ALL IN ONE  ###
### ### ### ### ###

def spectral_element(name:str, cell_type, degree:int) -> basix.ufl_wrapper.BasixElement:
    """
    A spectral element that can be used in a dolfinx.fem.FunctionSpace
    
    Args:
        name: One of ("GLL", "GL", "Legendre")
        cell_type: Elements can be defined from any type, but diagonal mass matrices
            can only be obtained using GLL / GL quadratures that require cell types
            interval (1D) / quadrilateral (2D) / hexahedron (3D)
        degree: The maximum degree of basis functions
    
    Example of use:
        from mpi4py import MPI
        from dolfinx import mesh, fem
        from elastodynamicsx.utils import spectral_element, spectral_quadrature
        #
        degree = 4
        specFE = spectral_element("GLL", mesh.CellType.quadrilateral, degree)
        specmd = spectral_quadrature("GLL", degree)
        V = fem.FunctionSpace( mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=mesh.CellType.quadrilateral), specFE )
        
        # compile mass matrices using pure fenicsx code
        import ufl
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        
        a1_non_diag = fem.form(ufl.inner(u,v) * ufl.dx)
        M1_non_diag = fem.petsc.assemble_matrix(a1_non_diag)
        M1_non_diag.assemble()
        
        a2_diag     = fem.form(ufl.inner(u,v) * ufl.dx(metadata=specmd))
        M2_diag     = fem.petsc.assemble_matrix(a2_diag)
        M2_diag.assemble()
        
        # compile mass matrices using elastodynamicsx.pde classes
        from elastodynamicsx.pde import PDE, material
        
        mat3          = material(V, 'scalar', 1, 1)
        pde3_non_diag = PDE([mat3])
        M3_non_diag   = pde3_non_diag.M()
        
        PDE.metadata  = md # default metadata for all forms in the pde package (materials, BCs, ...)
        mat4          = material(V, 'scalar', 1, 1)
        pde4_diag     = PDE([mat4])
        M4_diag       = pde4_diag.M()
        
        # spy mass matrices
        from elastodynamicsx.plot import spy_petscMatrix
        import matplotlib.pyplot as plt
        for i,M in enumerate([M1_non_diag, M2_diag, M3_non_diag, M4_diag]):
            fig = plt.figure()
            fig.suptitle('M'+str(i+1))
            spy_petscMatrix(M)
        plt.show()
    """
    if   name.lower() == "gll":
        return GLL_element(cell_type, degree)
    elif name.lower() == "gl":
        return GL_element(cell_type, degree)
    elif name.lower() == "legendre":
        return Legendre_element(cell_type, degree)
    else:
        raise TypeError("Unknown element name: {0:s}".format(name))

def spectral_quadrature(name:str, degree:int) -> dict:
    """
    A quadrature metadata to build diagonal mass matrices when used with
    corresponding spectral elements.
    
    Args:
        name: One of ("GLL", "GL", "Legendre")
        degree: The maximum degree of basis functions
    
    Example of use:
        See doc of 'spectral_element'.
    """
    if   name.lower() == "gll":
        return GLL_quadrature(degree)
    elif name.lower() == "gl":
        return GL_quadrature(degree)
    elif name.lower() == "legendre":
        return Legendre_quadrature(degree)
    else:
        raise TypeError("Unknown element name: {0:s}".format(name))

