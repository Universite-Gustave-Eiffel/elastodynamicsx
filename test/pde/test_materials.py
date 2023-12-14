# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

# TODO: currently only testing for build & compile, not for the validity of a result.

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type

from elastodynamicsx.pde import material, PDE, damping


def create_mesh(dim):
    if dim == 1:
        return mesh.create_unit_interval(MPI.COMM_WORLD, 5)
    elif dim == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    elif dim == 3:
        return mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    else:
        raise ValueError('Dimension should be one of (1,2,3). Got: ' + str(dim))


def tst_scalar_material(dim, eltname="Lagrange"):
    # FE domain
    V = fem.FunctionSpace(create_mesh(dim), (eltname, 1))

    # Material
    def const(x):
        return fem.Constant(V.mesh, default_scalar_type(x))

    mats = []
    mats.append(material(V, 'scalar', rho=const(1), mu=const(1)))

    # Damping laws
    mats.append(material(V, 'scalar', rho=const(1), mu=const(1), damping=damping('Rayleigh', const(1), const(1))))

    # PDE
    pde = PDE(V, materials=mats)

    # Compile some matrices
    _, _, _ = pde.M(), pde.C(), pde.K()
    _, _, _ = pde.K1(), pde.K2(), pde.K3()

    # The end


def tst_vector_materials(dim, nbcomps, eltname="Lagrange"):
    # FE domain
    V = fem.FunctionSpace(create_mesh(dim), (eltname, 1, (nbcomps,)))

    # Material
    def const(x):
        return fem.Constant(V.mesh, default_scalar_type(x))

    rho = const(1)
    Coo = const(1)  # a dummy Cij
    types = ('isotropic', 'cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthotropic', 'monoclinic', 'triclinic')
    nbCij = (2, 3, 5, 7, 7, 9, 13, 21)
    mats = []

    for type_, nb in zip(types, nbCij):
        Cij = [Coo] * nb
        mats.append(material(V, type_, rho, *Cij))

    # Damping laws
    mats.append(material(V, 'isotropic', rho, Coo, Coo, damping=damping('Rayleigh', const(1), const(1))))

    # PDE
    pde = PDE(V, materials=mats)

    # Compile some matrices
    _, _, _ = pde.M(), pde.C(), pde.K()

    if dim == 2 and nbcomps == 2:
        print('skipping invalid case for waveguides')

    elif V.element.basix_element.discontinuous is True:
        print('skipping not implemented case for waveguides (DG)')

    else:
        _, _, _ = pde.K1(), pde.K2(), pde.K3()

    # The end


def test_all():
    for eltname in ["Lagrange"]:  # , "DG"]:
        for dim in range(3):
            tst_scalar_material(dim + 1, eltname)
            for nbcomps in range(max(dim, 1), 3):  # avoids dim=1 and nbcomps=1
                print('test_all:: dim=' + str(dim + 1) + ', nbcomps=' + str(nbcomps + 1))
                tst_vector_materials(dim + 1, nbcomps + 1, eltname)
