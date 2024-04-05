# Copyright (C) 2024 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
import ufl

from elastodynamicsx.pde import PDE, material, PDECONFIG
from elastodynamicsx.utils import spectral_element, spectral_quadrature


def test():
    nbelts = 3
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nbelts, nbelts,
                                     cell_type=mesh.CellType.quadrilateral)

    degrees = range(1, 10)
    names = ('GLL', 'GL', 'LEGENDRE')
    for degree in degrees:
        for name in names:
            specFE = spectral_element(name, mesh.CellType.quadrilateral, degree)
            specmd = spectral_quadrature(name, degree)
            V = fem.FunctionSpace(domain, specFE)

            #######
            # Compile mass matrices using pure fenicsx code
            u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

            # here we specify the quadrature metadata -> the matrix becomes diagonal
            a1_diag = fem.form(ufl.inner(u, v) * ufl.dx(metadata=specmd))
            M1_diag = fem.petsc.assemble_matrix(a1_diag)
            M1_diag.assemble()

            #######
            # Compile mass matrices using elastodynamicsx.pde classes
            # elastodynamicsx offers two ways to specify the quadrature metadata

            # 1. by passing the metadata as kwargs to each material / boundary condition / bodyforce

            mat2 = material(V, 'scalar', 1, 1, metadata=specmd)
            pde2_diag = PDE(V, [mat2])
            M2_diag = pde2_diag.M()

            # 2. by passing the metadata to the PDE.default_metadata, which is being used by
            # all materials / boundary conditions / bodyforces at __init__ call

            PDECONFIG.default_metadata = specmd  # default for all forms in the pde package (materials, BCs, ...)

            mat3 = material(V, 'scalar', 1, 1)
            pde3_diag = PDE(V, [mat3])
            M3_diag = pde3_diag.M()

            for M_ in (M1_diag, M2_diag, M3_diag):
                d_ = M_.getDiagonal()
                MforceDiag = PETSc.Mat()
                MforceDiag.createAIJ(M_.size)
                MforceDiag.assemble()
                MforceDiag.setDiagonal(d_)
                assert np.isclose((M_ - MforceDiag).norm(), 0.), \
                    f"Matrix should be diagonal; Something went wrong for elts:{name}, deg:{degree}"
