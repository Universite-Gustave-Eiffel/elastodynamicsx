# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

import numpy as np

from petsc4py import PETSc

from dolfinx import fem

try:
    import dolfinx_mpc
    from dolfinx_mpc import MultiPointConstraint
except ImportError:
    dolfinx_mpc = None
    MultiPointConstraint = None


class TimeScheme():
    """
    .. role:: python(code)
      :language: python

    Abstract base class for time schemes as needed by the :python:`TimeStepper` solvers
    """

    labels: typing.List[str] = ['supercharge me']

    def build_timestepper(*args, **kwargs) -> 'TimeStepper':  # supercharge me
        raise NotImplementedError

    def __init__(self, dt, out: PETSc.Vec, **kwargs):
        self._dt = dt
        self._out: PETSc.Vec = out
        self._explicit: bool = kwargs.get('explicit', False)
        self._intermediate_dt: float = kwargs.get('intermediate_dt', 0.)

    @property
    def explicit(self) -> bool:
        return self._explicit

    @property
    def dt(self):
        """The time step"""
        return self._dt

    @property
    def intermediate_dt(self):
        return self._intermediate_dt

    @property
    def out(self) -> PETSc.Vec:
        """The solution vector"""
        return self._out

    def b_update_function(self, b: PETSc.Vec, t) -> None:  # supercharge me
        raise NotImplementedError

    def prepareNextIteration(self) -> None:  # supercharge me
        raise NotImplementedError

    def set_initial_condition(self, u0, v0) -> None:  # supercharge me
        raise NotImplementedError

    def initialStep(self,
                    t0,
                    callfirsts: typing.List[typing.Callable] = [],
                    callbacks: typing.List[typing.Callable] = [],
                    verbose: int = 0) -> None:  # supercharge me
        """Specific to the initial value step"""
        raise NotImplementedError


class FEniCSxTimeScheme(TimeScheme):
    """Abstract base class based on FEniCSx's form language"""

    def __init__(self, dt, out: fem.Function,
                 bilinear_form: fem.forms.Form,
                 linear_form: fem.forms.Form,
                 mpc: MultiPointConstraint = None,
                 bcs=[],
                 **kwargs):
        super().__init__(dt, out.vector, **kwargs)
        self._bilinear_form = bilinear_form
        self._linear_form = linear_form
        self._mpc = mpc
        self._bcs = bcs  # dirichlet BCs only
        self._out_fenicsx = out

        if self._mpc is None:
            self.b_update_function = self._b_update_function_WO_MPC
        else:
            self.b_update_function = self._b_update_function_WITH_MPC

    @property
    def out_fenicsx(self) -> fem.Function:
        """The solution vector"""
        return self._out_fenicsx

    def A(self) -> PETSc.Mat:
        """The time-independent matrix (bilinear form)"""
        if self._mpc is None:
            A = fem.petsc.assemble_matrix(self._bilinear_form, bcs=self._bcs)
        else:
            A = dolfinx_mpc.assemble_matrix(self._bilinear_form, self._mpc, bcs=self._bcs)
        A.assemble()
        return A

    def init_b(self) -> PETSc.Vec:
        """Declares a zero vector compatible with the linear form"""
        if self._mpc is None:
            return fem.petsc.create_vector(self._linear_form)
        else:
            return dolfinx_mpc.assemble_vector(self._linear_form, self._mpc)

    # def b_update_function(self, b: PETSc.Vec, t) -> None:  # NOW SET TO EITHER METHOD BELOW IN __init__

    def _b_update_function_WO_MPC(self, b: PETSc.Vec, t) -> None:  # TODO: use t?
        """Updates the b vector (in-place) for a given time t"""
        with b.localForm() as loc_b:
            loc_b.set(0)

        # fill with values
        fem.petsc.assemble_vector(b, self._linear_form)

        # BC modifyier
        fem.petsc.apply_lifting(b, [self._bilinear_form], [self._bcs])

        # ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # apply BC value
        fem.petsc.set_bc(b, self._bcs)

    def _b_update_function_WITH_MPC(self, b: PETSc.Vec, t) -> None:  # TODO: use t?
        """Updates the b vector (in-place) for a given time t"""
        with b.localForm() as loc_b:
            loc_b.set(0)

        # fill with values
        dolfinx_mpc.assemble_vector(self._linear_form, self._mpc, b)

        # BC modifyier
        dolfinx_mpc.apply_lifting(b, [self._bilinear_form], [self._bcs], self._mpc)

        # ghost
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # apply BC value
        fem.petsc.set_bc(b, self._bcs)  # not modified by dolfinx_mpc

    def set_initial_condition(self, u0, v0) -> None:
        """
        .. role:: python(code)
          :language: python

        Apply initial conditions

        Args:
            u0: u at t0
            v0: du/dt at t0

                u0 and v0 can be:

                - Python Callable -> interpolated at nodes
                    -> e.g. :python:`u0 = lambda x: np.zeros((domain.topology.dim, x.shape[1]), dtype=PETSc.ScalarType)`

                - scalar (:python:`int`, :python:`float`, :python:`complex`, :python:`PETSc.ScalarType`)
                    -> e.g. :python:`u0 = 0`

                - array (:python:`list`, :python:`tuple`, :python:`np.ndarray`)
                    -> e.g. :python:`u0 = [0,0,0]`

                - :python:`fem.function.Function` or :python:`fem.function.Constant`
                    -> e.g. :python:`u0 = fem.Function(V)`
        """
        for selfVal, val in ((self._u0, u0), (self._v0, v0)):
            if callable(val):
                selfVal.interpolate(val)
            elif issubclass(type(val), fem.function.Constant):
                selfVal.x.array[:] = np.tile(val.value, np.size(selfVal.x.array) // np.size(val.value))
            elif type(val) in (list, tuple, np.ndarray):
                selfVal.x.array[:] = np.tile(val, np.size(selfVal.x.array) // np.size(val))
            elif type(val) in (int, float, complex, PETSc.ScalarType):
                selfVal.x.array[:] = val
            elif issubclass(type(val), fem.function.Function):
                selfVal.x.array[:] = val.x.array
            else:
                raise TypeError("Unknown type of initial value " + str(type(val)))
