# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import numpy as np
from dolfinx.geometry import compute_collisions, BoundingBoxTree, compute_colliding_cells
from dolfinx.mesh import Mesh

from typing import Tuple


class ParallelEvaluator:
    """
    Convenience class to evaluate functions (fem.Function) when the mesh is
    distributed over several processes
    """
    def __init__(self, domain: Mesh, points: np.ndarray):
        pts_, cells_, indxs_ = find_points_and_cells_on_proc_new(domain, points)
        self.comm          = domain.comm
        self.points        = points
        self.points_local  = pts_
        self.cells_local   = cells_
        self.indexes_local = indxs_

    @property
    def nb_points_local(self) -> int:
        return len(self.points_local)

    def gather(self, eval_results: np.ndarray, root=0) -> np.ndarray:
        recv_indx = self.comm.gather(self.indexes_local, root=0)
        recv_eval = self.comm.gather(eval_results, root=0)

        rank = self.comm.Get_rank()
        if rank == 0:
            recv_indx = np.concatenate(recv_indx)
            recv_eval = np.concatenate(recv_eval, axis=0)
            out_eval = np.empty(recv_eval.shape, dtype=recv_eval.dtype)
            out_eval[recv_indx] = recv_eval
        else:
            out_eval = None

        return out_eval


def find_points_and_cells_on_proc(points: np.ndarray,
                                  domain: Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        V = fem.FunctionSpace( create_unit_square(MPI.COMM_WORLD, 10, 10), ("P", 1) )
        P1 = [0.1, 0.2, 0]  # point P1 with coordinates (x1, y1, z1)=(0.1, 0.2, 0)
        P2 = [0.3, 0.4, 0]  # point P2
        points = np.array([P1, P2]).T
        points_on_proc, cells_on_proc = find_points_and_cells_on_proc(points, V.mesh)
        u = fem.Function(V)
        u.eval(points_on_proc, cells_on_proc)  # evaluates u at points P1 and P2
    """
    cells = []
    points_on_proc = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates = compute_collisions(BoundingBoxTree(domain, domain.topology.dim), points.T)

    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)

    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cells          = np.array(cells, dtype='i')

    return points_on_proc, cells


def find_points_and_cells_on_proc_new(domain: Mesh,
                                      points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        V = fem.FunctionSpace( create_unit_square(MPI.COMM_WORLD, 10, 10), ("P", 1) )
        P1 = [0.1, 0.2, 0]  # point P1 with coordinates (x1, y1, z1)=(0.1, 0.2, 0)
        P2 = [0.3, 0.4, 0]  # point P2
        points = np.array([P1, P2]).T
        points_on_proc, cells_on_proc = find_points_and_cells_on_proc(points, V.mesh)
        u = fem.Function(V)
        u.eval(points_on_proc, cells_on_proc)  # evaluates u at points P1 and P2
    """
    cells = []
    points_on_proc = []
    pt_indexes = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates = compute_collisions(BoundingBoxTree(domain, domain.topology.dim), points.T)

    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points.T)

    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            pt_indexes.append(i)

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cells          = np.array(cells, dtype='i')
    pt_indexes     = np.array(pt_indexes, dtype='i')

    return points_on_proc, cells, pt_indexes
