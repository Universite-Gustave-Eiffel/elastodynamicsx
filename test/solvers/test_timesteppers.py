# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

# TODO: currently only testing for build & compile, not for the validity of a result.

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl

from elastodynamicsx.pde import material, PDE
from elastodynamicsx.solvers import TimeStepper


def create_mesh(dim):
    if dim == 1:
        return mesh.create_unit_interval(MPI.COMM_WORLD, 5)
    elif dim == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    elif dim == 3:
        return mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    else:
        raise ValueError('Dimension should be one of (1,2,3). Got: ' + str(dim))


def test():
    dim = 2
    nbcomps = 2
    eltname = "Lagrange"

    # FE domain
    V = fem.FunctionSpace(create_mesh(dim), (eltname, 1, (nbcomps,)))

    # Material
    def const(x):
        return fem.Constant(V.mesh, default_scalar_type(x))

    rho = const(1)
    Coo = const(1)  # a dummy Cij
    mat = material(V, 'isotropic', rho, Coo, Coo)

    # PDE
    pde = PDE(V, materials=[mat])

    dt = 1
    cmax = ufl.sqrt(mat.P_modulus / rho)
    TimeStepper.Courant_number(V.mesh, cmax, dt)  # Courant number
    num_steps = 5

    for scheme in ('leapfrog', 'newmark', 'hht', 'generalized-alpha', 'midpoint', 'linear-acceleration-method'):
        tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=pde.bcs, scheme=scheme)
        tStepper.set_initial_condition(u0=[0, 0], v0=[0, 0], t0=0)
        tStepper.solve(num_steps)
