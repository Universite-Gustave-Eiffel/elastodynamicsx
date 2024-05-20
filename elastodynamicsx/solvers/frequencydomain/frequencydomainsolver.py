# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from typing import Callable, List, Union

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(x):  # type: ignore[no-redef]
        return x


class FrequencyDomainSolver:
    """
    Class for solving frequency domain problems.

    Args:
        comm: The MPI communicator
        M: The mass matrix
        C: The damping matrix
        K: The stiffness matrix
        b: The load vector
        b_update_function: A function that updates the load vector (in-place)
            The function must take b,omega as parameters.
            e.g.: b_update_function = lambda b,omega: b[:]=omega
            If set to None, the call is ignored.

    Keyword Args:
        petsc_options: Options that are passed to the linear
            algebra backend PETSc. For available choices for the
            'petsc_options' kwarg, see the `PETSc documentation
            <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.

    Example:
        .. highlight:: python
        .. code-block:: python

          from mpi4py import MPI

          from dolfinx import mesh, fem
          import ufl

          from elastodynamicsx.solvers import FrequencyDomainSolver
          from elastodynamicsx.pde import material, BoundaryCondition, PDE

          # domain
          length, height = 10, 10
          Nx, Ny = 10, 10
          domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0,0], [length,height]], [Nx,Ny])
          V      = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (2,)))

          # material
          rho, lambda_, mu = 1, 2, 1
          mat = material(V, rho, lambda_, mu)

          # absorbing boundary condition
          Z_N, Z_T = mat.Z_N, mat.Z_T  # P and S mechanical impedances
          bcs = [ BoundaryCondition(V, 'Dashpot', (Z_N, Z_T)) ]

          # gaussian source term
          F0     = fem.Constant(domain, PETSc.ScalarType([1, 0]))  # polarization
          R0     = 0.1  # radius
          x0, y0 = length/2, height/2  # center
          x      = ufl.SpatialCoordinate(domain)
          gaussianBF = F0 * ufl.exp(-((x[0]-x0)**2 + (x[1]-y0)**2) /2 /R0**2) / (2 * 3.141596*R0**2)
          bf         = BodyForce(V, gaussianBF)

          # PDE
          pde = PDE(V, materials=[mat], bodyforces=[bf], bcs=bcs)

          # solve
          fdsolver = FrequencyDomainSolver(V.mesh.comm, pde.M(), pde.C(), pde.K(), pde.b())
          omega    = 1.0
          u        = fem.Function(V, name='solution')
          fdsolver.solve(omega=omega, out=u.x.petsc_vec)
    """

    default_petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}  # "pc_factor_mat_solver_type": "mumps"

    def __init__(self, comm: MPI.Comm, M: PETSc.Mat, C: PETSc.Mat, K: PETSc.Mat, b: PETSc.Vec,  # type: ignore
                 b_update_function: Union[Callable, None] = None, **kwargs):
        self._M = M
        self._C = C
        self._K = K
        self._b = b
        self._b_update_function = b_update_function

        # ### ### #
        # Initialize the PETSc solver
        petsc_options = kwargs.get('petsc_options', FrequencyDomainSolver.default_petsc_options)
        self.solver = PETSc.KSP().create(comm)  # type: ignore

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self.solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(problem_prefix)

        for k, v in petsc_options.items():
            opts[k] = v

        opts.prefixPop()
        self.solver.setFromOptions()
        # ### ### #

    def solve(self, omega: Union[float, np.ndarray],
              out: PETSc.Vec = None,  # type: ignore
              callbacks: List[Callable] = [],
              **kwargs) -> PETSc.Vec:  # type: ignore
        """
        Solve the linear problem

        Args:
            omega: The angular frequency (scalar or array)
            out: The solution (displacement field) to the last solve. If
                None a new PETSc.Vec is created
            callbacks: If omega is an array, list of callback functions
                to be called after each solve (e.g. plot, store solution, ...).
                Ignored if omega is a scalar.

        Keyword Args:
            live_plotter: a plotter object that can refresh through
                a live_plotter.live_plotter_update_function(i, out) function

        Returns:
            out
        """
        if out is None:
            out = self._M.createVecRight()

        if hasattr(omega, '__iter__'):
            omega = np.asarray(omega)
            return self._solve_multiple_omegas(omega, out, callbacks, **kwargs)

        else:
            return self._solve_single_omega(omega, out)

    def _solve_single_omega(self, omega: float, out: PETSc.Vec) -> PETSc.Vec:  # type: ignore
        # Update load vector at angular frequency 'omega'
        if not (self._b_update_function is None):
            self._b_update_function(self._b, omega)
            # Assume self._b.ghostUpdate(...) has already been done

        # Update PDE matrix
        w = omega
        A = PETSc.ScalarType(-w * w) * self._M + PETSc.ScalarType(1J * w) * self._C + self._K  # type: ignore
        self.solver.setOperators(A)

        # Solve
        self.solver.solve(self._b, out)

        # Update the ghosts in the solution
        out.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        return out

    def _solve_multiple_omegas(self, omegas: np.ndarray,
                               out: PETSc.Vec,  # type: ignore
                               callbacks: List[Callable] = [], **kwargs) \
            -> PETSc.Vec:  # type: ignore

        # Loop on values in omegas -> _solve_single_omega

        call_on_leave = kwargs.get('call_on_leave', [])

        live_plt = kwargs.get('live_plotter', None)

        if not (live_plt is None):
            if isinstance(live_plt, dict):
                # from elastodynamicsx.plot import live_plotter
                # live_plt = live_plotter(self.u, live_plt.pop('refresh_step', 1), **live_plt)
                raise NotImplementedError

            callbacks.append(live_plt.live_plotter_update_function)
            call_on_leave.append(live_plt.live_plotter_stop)
            live_plt.live_plotter_start()

        for i in tqdm(range(len(omegas))):
            self._solve_single_omega(omegas[i], out)
            for callback in callbacks:
                callback(i, out)  # <- store solution, plot, print, ...

        for function in call_on_leave:
            function()

        return out

    @property
    def M(self) -> PETSc.Mat:  # type: ignore
        """The mass matrix"""
        return self._M

    @property
    def C(self) -> PETSc.Mat:  # type: ignore
        """The damping matrix"""
        return self._C

    @property
    def K(self) -> PETSc.Mat:  # type: ignore
        """The stiffness matrix"""
        return self._K

    @property
    def b(self) -> PETSc.Vec:  # type: ignore
        """The load vector"""
        return self._b
