#

import numpy as np
from dolfinx import geometry, mesh

def find_points_and_cells_on_proc(points, domain):
    """
    Adapted from: https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html
    
    example of use:
    
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
    """
    Adapted from: https://jsdokken.com/dolfinx-tutorial/chapter3/robin_neumann_dirichlet.html
    
    example of use:
    
    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square
    #
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),\
                  (2, lambda x: np.isclose(x[0], 1)),\
                  (3, lambda x: np.isclose(x[1], 0)),\
                  (4, lambda x: np.isclose(x[1], 1))]
    facet_tags = make_facet_tags(domain, boundaries)
    """
    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    return facet_tags

