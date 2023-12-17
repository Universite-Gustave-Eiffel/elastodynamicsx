# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

from mpi4py import MPI

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.cpp.geometry import determine_point_ownership  # type: ignore


class ParallelEvaluator:
    """
    Convenience class to evaluate functions (fem.Function) when the mesh is
    distributed over several processes

    Args:
        domain: a distributed mesh (MPI.COMM_WORLD)
        points: shape=(3, nbpts). Should be defined either for all processes,
            or for the proc no 0. Does not support 'points' being scattered on
            several processes.
        padding: close-to-zero parameter used in dolfinx.cpp.geometry.determine_point_ownership

    Example:
        .. highlight:: python
        .. code-block:: python

          # Define some output points
          x = np.linspace(0, 1, num=30)
          y = np.zeros_like(x)
          z = np.zeros_like(x)
          points = np.array([x, y, z])

          # Initialize the evaluator
          paraEval = ParallelEvaluator(domain, points)

          # Perform function evaluations
          u_eval_local = u.eval(paraEval.points_local, paraEval.cells_local)

          # Gather all procs
          u_eval = paraEval.gather(u_eval_local, root=0)

          # Do something, e.g. export to file
          if domain.comm.rank == 0:
              np.savez('u_eval.npz', x=points.T, u=u_eval)

    Adapted from:
        https://github.com/jorgensd/dolfinx-tutorial/issues/116
    """
    def __init__(self, domain: Mesh, points: np.ndarray, padding: float = 1.e-4):
        if domain.comm.rank == 0:
            pass
        else:
            # Only add points on one process
            points = np.zeros((3, 0))

        src_owner, dest_owner, dest_points, dest_cells = \
            determine_point_ownership(domain._cpp_object, points.T, padding)

        self.comm: MPI.Comm = domain.comm
        self.points: np.ndarray = points
        self.src_owner = src_owner
        self.dest_owner = dest_owner
        self.points_local: np.ndarray = np.array(dest_points).reshape(len(dest_points) // 3, 3)
        self.cells_local = dest_cells

    @property
    def nb_points_local(self) -> int:
        return len(self.points_local)

    def gather(self, eval_results: np.ndarray, root=0) -> typing.Union[np.ndarray, None]:
        recv_eval = self.comm.gather(eval_results, root=root)
        rank = self.comm.Get_rank()
        if rank == 0:
            assert isinstance(recv_eval, np.ndarray)
            # concatenate
            recv_eval = np.concatenate(recv_eval, axis=0)
            # re-order
            recv_indx = np.argsort(np.asarray(self.src_owner) + np.linspace(0, 0.1, num=len(self.src_owner)))
            out_eval = np.empty(recv_eval.shape, dtype=recv_eval.dtype)
            out_eval[recv_indx] = recv_eval
        else:
            out_eval = None

        return out_eval
