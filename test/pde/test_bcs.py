# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

# TODO: currently only testing for build & compile, not for the validity of a result.

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type

from elastodynamicsx.pde import material, PDE, BoundaryCondition
from elastodynamicsx.utils import make_facet_tags


tag_left, tag_right = 1, 2
tagged_boundaries = [(tag_left, lambda x: np.isclose(x[0], 0)),
                     (tag_right, lambda x: np.isclose(x[0], 1))]


def create_mesh(dim):
    if dim == 1:
        return mesh.create_unit_interval(MPI.COMM_WORLD, 5)
    elif dim == 2:
        return mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)
    elif dim == 3:
        return mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    else:
        raise ValueError('Dimension should be one of (1,2,3). Got: ' + str(dim))


def tst_bcs_scalar_material(dim, eltname="Lagrange"):
    # FE domain
    domain = create_mesh(dim)
    facet_tags = make_facet_tags(domain, tagged_boundaries)
    V = fem.FunctionSpace(domain, (eltname, 1))

    # Material
    def const(x):
        return fem.Constant(V.mesh, default_scalar_type(x))

    mat = material(V, 'scalar', rho=const(1), mu=const(1))

    # BCs
    dummy_value = const(1)
    supported_bcs = []
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Free'))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Clamp'))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Dirichlet', dummy_value))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Neumann', dummy_value))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Robin', (dummy_value, dummy_value)))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Dashpot', mat.Z))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Periodic', [1, 0, 0]))

    for bc in supported_bcs:
        # PDE
        pde = PDE(V, materials=[mat], bcs=[bc])

        # Compile some matrices
        _, _, _ = pde.M(), pde.C(), pde.K()

        if bc.type in ("robin", "dashpot"):
            print('tst_bcs_scalar_material:: skipping ' + bc.type + ' BC test for K1, K2, K3 matrices')
            pass

        else:
            _, _, _ = pde.K1(), pde.K2(), pde.K3()

    # The end


def tst_bcs_vector_materials(dim, nbcomps, eltname="Lagrange"):
    # FE domain
    domain = create_mesh(dim)
    facet_tags = make_facet_tags(domain, tagged_boundaries)
    V = fem.FunctionSpace(domain, (eltname, 1, (nbcomps,)))

    # Material
    def const(x):
        return fem.Constant(V.mesh, default_scalar_type(x))

    rho = const(1)
    Coo = const(1)  # a dummy Cij
    mat = material(V, 'isotropic', rho, Coo, Coo)

    # BCs
    dummy_vector = const([1] * nbcomps)
    dummy_matrix = const([[1] * nbcomps] * nbcomps)
    supported_bcs = []
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Free'))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Clamp'))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Dirichlet', dummy_vector))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Neumann', dummy_vector))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Robin', (dummy_matrix, dummy_vector)))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Dashpot', (mat.Z_N, mat.Z_T)))
    supported_bcs.append(BoundaryCondition((V, facet_tags, tag_left), 'Periodic', [1, 0, 0]))

    for bc in supported_bcs:
        # PDE
        print(bc.type, end='; ')
        pde = PDE(V, materials=[mat], bcs=[bc])

        # Compile some matrices
        _, _, _ = pde.M(), pde.C(), pde.K()

        if bc.type in ("robin", "dashpot"):
            print('tst_bcs_vector_materials:: skipping ' + bc.type + ' BC test for K1, K2, K3 matrices')
            pass

        elif dim == 2 and nbcomps == 2:
            print('tst_bcs_vector_materials:: skipping invalid case for waveguides')
            pass

        elif V.element.basix_element.discontinuous is True:
            print('tst_bcs_vector_materials:: skipping not implemented case for waveguides (DG)')

        else:
            _, _, _ = pde.K1(), pde.K2(), pde.K3()

    # The end


def test_all():
    for eltname in ["Lagrange"]:  # , "DG"]:
        for dim in range(3):
            tst_bcs_scalar_material(dim + 1, eltname)
            for nbcomps in range(max(dim, 1), 3):  # avoids dim=1 and nbcomps=1
                print('test_all:: dim=' + str(dim + 1) + ', nbcomps=' + str(nbcomps + 1))
                tst_bcs_vector_materials(dim + 1, nbcomps + 1, eltname)
