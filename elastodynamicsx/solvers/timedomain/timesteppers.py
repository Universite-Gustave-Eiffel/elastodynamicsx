# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from typing import Union

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
import ufl  # type: ignore

from elastodynamicsx.pde.timeschemes import TimeScheme, timescheme

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(x):  # type: ignore[no-redef]
        return x


class DiagonalSolver:
    """
    A solver with a diagonal left hand side

    Args:
        A: A PETSc vector that represents a diagonal matrix
    """
    def __init__(self, A: PETSc.Vec):  # type: ignore
        self._A = A

    def solve(self, b: PETSc.Vec, out: PETSc.Vec) -> None:  # type: ignore
        """
        Solve (in-place) the linear system
        :math:`\mathbf{A} * \mathbf{out} = \mathbf{b}`
        """
        out.setArray(b / self._A)


class TimeStepper:
    """
    Base class for solving time-dependent problems.
    """

    # --------------------------
    # --------- static ---------
    # --------------------------

    petsc_options_explicit_scheme = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options_implicit_scheme_linear = {"ksp_type": "preonly", "pc_type": "lu"}
    # petsc_options_implicit_scheme_nonlinear =  # TODO

    def build(*args, **kwargs):
        """
        Convenience static method that instanciates the required time-stepping scheme

        Args:
            args: (passed to the required scheme)

        Keyword Args:
            scheme: (required) Available options are 'leapfrog', 'midpoint'
                'linear-acceleration-method', 'newmark', 'hht-alpha', 'generalized-alpha'
            **kwargs: (passed to the required scheme)
        """
        tscheme = timescheme(*args, **kwargs)
        comm = tscheme.u.function_space.mesh.comm

        if tscheme.linear_ODE is True:
            if tscheme.nbsteps == 1:
                return OneStepTimeStepper(comm, tscheme, tscheme.A(), tscheme.init_b(), **kwargs)
            else:
                raise NotImplementedError  # TODO: multi-steps?
        else:
            return NonlinearTimeStepper(comm, tscheme, **kwargs)

    @staticmethod
    def Courant_number(domain: mesh.Mesh, c_max, dt):
        """
        The Courant number: :math:`C = c_{max} \, \mathrm{d}t / h`, with :math:`h` the cell diameter

        Related to the Courant-Friedrichs-Lewy (CFL) condition.

        Args:
            domain: the mesh
            c_max: ufl-compatible, the maximum velocity
            dt: ufl-compatible, the time step

        See:
            https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
        """
        V = fem.functionspace(domain, ("DG", 0))
        c_number = fem.Function(V)
        pts = V.element.interpolation_points()  # DOLFINx.__version__ >=0.5
        h = ufl.MinCellEdgeLength(V.mesh)  # or rather ufl.CellDiameter?
        c_number.interpolate(fem.Expression(dt * c_max / h, pts))
        c_number_max = max(c_number.x.array)
        return V.mesh.comm.allreduce(c_number_max, op=MPI.MAX)

    # --------------------------
    # ------- non-static -------
    # --------------------------

    def __init__(self, comm: MPI.Comm, tscheme: TimeScheme, **kwargs):
        """
        Args:
            comm: The MPI communicator
            tscheme: Time scheme
        """
        self._tscheme = tscheme
        self._comm: MPI.Comm = comm
        self._t = 0
        self._dt = self._tscheme.dt
        self._out = self._tscheme.out

    @property
    def timescheme(self):
        return self._tscheme

    @property
    def t(self):
        return self._t

    @property
    def dt(self):
        return self._dt

    def set_initial_condition(self, u0, v0, t0=0) -> None:
        """
        .. role:: python(code)
           :language: python

        Apply initial conditions

        Args:
            u0: u at t0
            v0: du/dt at t0
            t0: start time (default: 0)

        u0 and v0 can be:
            - Python callable -> will be evaluated at nodes
                -> e.g. :python:`u0 = lambda x: np.zeros((dim, x.shape[1]), dtype=PETSc.ScalarType)`
            - scalar (int, float, complex, PETSc.ScalarType)
                -> e.g. :python:`u0 = 0`
            - array (list, tuple, np.ndarray) or fem.function.Constant
                -> e.g. :python:`u0 = [0,0,0]`
            - fem.function.Function
                -> e.g. :python:`u0 = fem.Function(V)`
        """
        self._tscheme.set_initial_condition(u0, v0)
        self._t = t0

    def solve(self, num_steps, **kwargs):  # supercharge me
        raise NotImplementedError


class NonlinearTimeStepper(TimeStepper):
    """
    Base class for solving nonlinear problems using implicit time schemes. Not implemented yet.
    """
    def __init__(self, comm: MPI.Comm, tscheme: TimeScheme, **kwargs):
        super().__init__(comm, tscheme, **kwargs)
        raise NotImplementedError


class LinearTimeStepper(TimeStepper):
    """
    Base class for solving linear problems. Note that nonlinear problems formulated
    with an explicit scheme come down to linear problems; they are handled by this class.
    """
    def __init__(self, comm: MPI.Comm, tscheme: TimeScheme,
                 A: PETSc.Mat, b: PETSc.Vec, **kwargs):  # type: ignore
        super().__init__(comm, tscheme, **kwargs)

        if kwargs.get('diagonal', False) and isinstance(A, PETSc.Mat):  # type: ignore
            A = A.getDiagonal()

        self._A = A  # Time-independent operator
        self._b = b  # Time-dependent right-hand side
        self._explicit = tscheme.explicit
        self._solver = None

        if isinstance(A, PETSc.Vec):  # type: ignore
            self._init_solver_diagonal(A)
        else:
            if self._explicit:
                default_petsc_options = TimeStepper.petsc_options_explicit_scheme
            else:
                default_petsc_options = TimeStepper.petsc_options_implicit_scheme_linear
            petsc_options = kwargs.get('petsc_options', default_petsc_options)
            self._init_solver(comm, petsc_options)

    @property
    def A(self) -> Union[PETSc.Mat, PETSc.Vec]:  # type: ignore
        return self._A

    @property
    def b(self) -> PETSc.Vec:  # type: ignore
        return self._b

    @property
    def explicit(self) -> bool:
        return self._explicit

    @property
    def solver(self) -> Union[PETSc.KSP, DiagonalSolver]:  # type: ignore
        return self._solver

    def _init_solver_diagonal(self, A: PETSc.Vec) -> None:  # type: ignore
        self._solver = DiagonalSolver(A)  # type: ignore

    def _init_solver(self, comm: MPI.Comm, petsc_options={}):
        # see https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/fem/petsc.py
        # see also https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html
        # ##   ###   ###   ###

        # Solver
        self._solver = PETSc.KSP().create(comm)  # type: ignore
        self._solver.setOperators(self._A)  # type: ignore

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)  # type: ignore

        # Set PETSc options
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(problem_prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()  # type: ignore

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()


class OneStepTimeStepper(LinearTimeStepper):
    """
    Base class for solving time-dependent problems with one-step algorithms (e.g. Newmark-beta methods).
    """

    def __init__(self, comm: MPI.Comm, tscheme: TimeScheme, A: PETSc.Mat, b: PETSc.Vec, **kwargs):  # type: ignore
        super().__init__(comm, tscheme, A, b, **kwargs)
        self._b_update_function = tscheme.b_update_function

        # self._i0 = 1 for explicit schemes because solving the initial
        # value problem a0=M_inv.(F(t0)-K(u0)-C.v0) also yields u1=u(t0+dt)
        self._i0 = 1 * self._explicit
        self._intermediate_dt = tscheme.intermediate_dt  # non zero for generalized-alpha

    def solve(self, num_steps, **kwargs):
        """
        .. role:: python(code)
           :language: python

        Run the loop on time steps

        Args:
            num_steps: number of time steps to integrate

        Keyword Args:
            callfirsts: (default=[]) list of functions to be called at the beginning
                of each iteration (before solving). For instance: update a source term.
                Each callfirst if of the form: cf = lambda t: do_something
                where t is the time at which to evaluate the sources
            callbacks:  (detault=[]) similar to callfirsts, but the callbacks are called
                at the end of each iteration (after solving). For instance: store/save, plot, print, ...
                Each callback is of the form :python:`cb = lambda i, out: -> do_something()`
                where :python:`i` is the iteration index and :python:`out` is the PETSc.Vec
                vector just solved for
            live_plotter: a plotter object that can refresh through
                a live_plotter.live_plotter_update_function(i, out) function
            verbose: (default=0) Verbosity level. >9 means an info msg before each step
        """
        verbose = kwargs.get('verbose', 0)

        callfirsts = kwargs.get('callfirsts', [])
        callbacks = kwargs.get('callbacks', [])
        call_on_leave = kwargs.get('call_on_leave', [])

        live_plt = kwargs.get('live_plotter', None)
        if not (live_plt is None):
            if isinstance(live_plt, dict):
                from elastodynamicsx.plot import live_plotter
                u_init = self._tscheme.out_fenicsx  # assume timescheme is of type FEniCSxTimeScheme
                live_plt = live_plotter(u_init, live_plt.pop('refresh_step', 1), **live_plt)

            callbacks.append(live_plt.live_plotter_update_function)
            call_on_leave.append(live_plt.live_plotter_stop)
            live_plt.live_plotter_start()

        t0 = self.t
        self._tscheme.initialStep(t0, callfirsts, callbacks, verbose=verbose)

        for i in tqdm(range(self._i0, num_steps)):
            self._t += self.dt
            t_calc = self.t - self.dt * self._intermediate_dt

            if verbose >= 10:
                PETSc.Sys.Print('Callfirsts...')

            for callfirst in callfirsts:
                callfirst(t_calc)  # <- update stuff (e.g. sources)

            # Update the right hand side reusing the initial vector
            if verbose >= 10:
                PETSc.Sys.Print('Update the right hand side reusing the initial vector...')

            self._tscheme.b_update_function(self._b, t_calc)
            # assume the following has already been done:
            # self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            # Solve linear problem
            if verbose >= 10:
                PETSc.Sys.Print('Solving...')

            self._solver.solve(self._b, self._out)

            # update the ghosts in the solution
            self._out.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            if verbose >= 10:
                PETSc.Sys.Print('Time-stepping for next iteration...')

            self._tscheme.prepareNextIteration()

            if verbose >= 10:
                PETSc.Sys.Print('Callbacks...')

            for callback in callbacks:
                callback(i, self._out)  # <- store solution, plot, print, ...

        for function in call_on_leave:
            function()
