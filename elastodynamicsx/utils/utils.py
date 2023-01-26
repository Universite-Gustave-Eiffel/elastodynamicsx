#

import numpy as np
from dolfinx import geometry, mesh, fem

def find_points_and_cells_on_proc(points, domain):
    """
    Get the points and corresponding cells that pertain to the current MPI proc.
    The output is ready for use in a fem.Function.eval call.
    
    Args:
        points: Array of size (3, number of points) -> coordinates of points
            that may or may not belong to the current MPI proc
        domain: The mesh belonging to the current MPI proc
    
    Returns:
        (points_on_proc, cells_on_proc)
        
    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
    
    Example of use:
        import numpy as np
        from mpi4py import MPI
        from dolfinx.mesh import create_unit_square
        from dolfinx import fem
        #
        V = fem.FunctionSpace( create_unit_square(MPI.COMM_WORLD, 10, 10), ("CG", 1) )
        P1 = [0.1, 0.2, 0] #point P1 with coordinates (x1, y1, z1)=([0.1, 0.2, 0)
        P2 = [0.3, 0.4, 0] #point P2
        points = np.array([P1, P2]).T
        points_on_proc, cells_on_proc = find_points_and_cells_on_proc(points, V.mesh)
        u = fem.Function(V)
        u.eval(points_on_proc, cells_on_proc) #evaluates u at points P1 and P2
    """
    cells = []
    points_on_proc = []
    
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions( geometry.BoundingBoxTree(domain, domain.topology.dim), points.T )
    
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    
    return points_on_proc, cells

def make_facet_tags(domain, boundaries):
    """Shortcut for make_tags(domain, locators, type_='boundaries')"""
    return make_tags(domain, boundaries, type_='boundaries')

def make_cell_tags( domain, subdomains):
    """Shortcut for make_tags(domain, locators, type_='domains')"""
    return make_tags(domain, subdomains, type_='domains')

def make_tags(domain, locators, type_='unknown'):
    """
    Args:
        domain: A mesh
        locators: A list of tuples of the type (tag, fn), where:
            tag: is an int to be assigned to the cells or facets
            fn: is a boolean function that take 'x' as argument
        type_: either 'domains' or 'boundaries'
    
    Returns:
        A MeshTags object (dolfinx.mesh)
        
    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
    
    Example of use:
        import numpy as np
        from mpi4py import MPI
        from dolfinx.mesh import create_unit_square
        #
        domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    
        boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                      (2, lambda x: np.isclose(x[0], 1)),
                      (3, lambda x: np.isclose(x[1], 0)),
                      (4, lambda x: np.isclose(x[1], 1))]
        facet_tags = make_tags(domain, boundaries, 'boundaries')
    
        Omegas = [(1, lambda x: x[1] <= 0.5),
                  (2, lambda x: x[1] >= 0.5)]
        cell_tags = make_tags(domain, Omegas, 'domains')
    """
    if   type_.lower() == 'boundaries':
        fdim = domain.topology.dim - 1
    elif type_.lower() == 'domains':
        fdim = domain.topology.dim
    else:
        raise TypeError("Unknown type: {0:s}".format(type_))
    
    loc_indices, loc_markers = [], []
    for (marker, locator) in locators:
        loc = mesh.locate_entities(domain, fdim, locator)
        loc_indices.append(loc)
        loc_markers.append(np.full_like(loc, marker))
        
    loc_indices = np.hstack(loc_indices).astype(np.int32)
    loc_markers = np.hstack(loc_markers).astype(np.int32)
    sorted_loc = np.argsort(loc_indices)
    loc_tags = mesh.meshtags(domain, fdim, loc_indices[sorted_loc], loc_markers[sorted_loc])
    
    return loc_tags

def get_functionspace_tags_marker(functionspace_tags_marker):
    """
    This is a convenience function for several classes/functions of other packages.
    It is not intended to be used in other context.
    
    Example of use:
        function_space, tags, marker = get_functionspace_tags_marker(functionspace_tags_marker)
    
        where functionspace_tags_marker can be:
            functionspace_tags_marker = (function_space, facet_tags, marker)
            functionspace_tags_marker = (function_space, cell_tags, marker)
            functionspace_tags_marker = function_space #means tags=None and marker=None
    """
    if type(functionspace_tags_marker) == fem.FunctionSpace:
        return functionspace_tags_marker, None, None
    else:
        return functionspace_tags_marker

