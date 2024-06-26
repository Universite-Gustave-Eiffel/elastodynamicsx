# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

# TODO: optimize SLEPc default options

import typing

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc  # type: ignore

import numpy as np

from dolfinx.fem import FunctionSpace

from elastodynamicsx.solutions import ModalBasis

# see https://slepc.upv.es/documentation/
#     https://slepc.upv.es/documentation/current/docs/manualpages/EPS/index.html
#     https://slepc.upv.es/documentation/current/docs/manualpages/PEP/index.html
#     https://slepc4py.readthedocs.io/en/stable/


class EigenmodesSolver(SLEPc.EPS):  # SLEPc.PEP for polynomial eigenvalue problem
    """
    Convenience class inhereted from SLEPc.EPS, with methods and
    default parameters that are relevant for computing the resonances
    of an elastic component.

    Args:
        comm: The MPI communicator
        M: The mass matrix
        C: The damping matrix. C=None means no dissipation. C!=None is not supported yet.
        K: The stiffness matrix

    Keyword Args:
        nev: The number of eigenvalues to be computed

    Example:
        .. highlight:: python
        .. code-block:: python

          # ###
          # Free resonances of an elastic cube
          # ###
          from mpi4py import MPI
          from dolfinx import mesh, fem
          from elastodynamicsx.solvers import EigenmodesSolver
          from elastodynamicsx.pde import material, PDE

          domain = mesh.create_box(MPI.COMM_WORLD, [[0,0,0], [1,1,1]], [10,10,10])
          V      = dolfinx.fem.functionspace(domain, ("CG", 1, (3,)))

          rho, lambda_, mu = 1, 2, 1
          mat = material(V, rho, lambda_, mu)
          pde = PDE(V, materials=[mat])
          nev = 6 + 6  # the first 6 resonances are rigid body motion
          eps = EigenmodesSolver(V.mesh.comm, pde.M(), None, pde.K(), nev=nev)
          eps.solve()
          eps.plot(V)
          freqs = eps.getEigenfrequencies()
          print('First resonance frequencies:', freqs)
    """

    def __init__(self, comm: MPI.Comm, M: PETSc.Mat, C: PETSc.Mat, K: PETSc.Mat, **kwargs):  # type: ignore
        super().__init__()

        if not (C is None):  # TODO
            raise NotImplementedError

        #
        self.create(comm)
        self.setOperators(K, M)
        self.setProblemType(SLEPc.EPS.ProblemType.GHEP)  # GHEP = Generalized Hermitian Eigenvalue Problem
        # self.setTolerances(tol=1e-9)
        self.setType(SLEPc.EPS.Type.KRYLOVSCHUR)  # Note that Krylov-Schur is the default solver

        # ## Spectral transform
        st = self.getST()

        # SINVERT = Shift and invert. By default, Slepc computes the largest eigenvalue,
        # while we are interested in the smallest ones
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(1e-8)  # can be set to a different value if the focus is set on another part of the spectrum

        # ## Number of eigenvalues to be computed
        nev = kwargs.get('nev', 10)
        self.setDimensions(nev=nev)

    def getWn(self) -> np.ndarray:
        """The eigen angular frequencies from the computed eigenvalues"""
        # abs because rigid body motions may lead to minus zero: -0.00000
        return np.array([np.sqrt(abs(self.getEigenvalue(i).real)) for i in range(self._getNout())])

    def getEigenfrequencies(self) -> np.ndarray:
        """The eigenfrequencies from the computed eigenvalues"""
        return self.getWn() / (2 * np.pi)

    def getEigenmodes(self, which='all') \
            -> typing.List[PETSc.Vec]:  # type: ignore
        """
        Returns the desired modeshapes

        Args:
            which: 'all', or an integer, or a list of integers, or a slice object

        Example:
            .. highlight:: python
            .. code-block:: python

              getEigenmodes()   # returns all computed eigenmodes
              getEigenmodes(3)  # returns mode number 4
              getEigenmodes([3,5])  # returns modes number 4 and 6
              getEigenmodes(slice(0,None,2))  # returns even modes
        """
        K, M = self.getOperators()
        indexes = _slice_array(np.arange(self._getNout()), which)
        eigenmodes = [K.createVecRight() for i in range(np.size(indexes))]

        for i, eigM in zip(indexes, eigenmodes):
            self.getEigenpair(i, eigM)  # Save eigenvector in eigM

        return eigenmodes

    def getModalBasis(self) -> ModalBasis:
        return ModalBasis(self.getWn(), self.getEigenmodes())

    def getErrors(self) -> np.ndarray:
        """Returns the error estimate on the computed eigenvalues"""
        return np.array([self.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
                         for i in range(self._getNout())])  # Compute error for i-th eigenvalue

    def plot(self, function_space: FunctionSpace, which='all', **kwargs) -> None:
        """
        Plots the desired modeshapes

        Args:
            function_space: The underlying function space
            which: 'all', or an integer, or a list of integers, or a slice object
                -> the same as for getEigenmodes
        """
        self.getModalBasis().plot(function_space, which, **kwargs)

    def printEigenvalues(self) -> None:
        """Prints the computed eigenvalues and error estimates"""
        v = [self.getEigenvalue(i) for i in range(self._getNout())]
        e = self.getErrors()
        PETSc.Sys.Print("       eigenvalue \t\t\t error ")  # type: ignore
        for cv, ce in zip(v, e):
            PETSc.Sys.Print(cv, '\t', ce)  # type: ignore

    def _getNout(self):
        """Returns the number of eigenpairs that can be returned. Usually equal to 'nev'."""
        nconv = self.getConverged()
        nev, _, _ = self.getDimensions()
        nout = min(nev, nconv)
        return nout


def _slice_array(a, which):
    """Not intended to be called by user"""
    if which == 'all':
        which = slice(0, None, None)

    if isinstance(which, int):
        which = slice(which, which + 1, None)

    return a[which]
