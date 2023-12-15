# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

try:
    import dolfinx_mpc
except ImportError:
    import warnings
    warnings.warn("Can't import dolfinx_mpc. Some features are not available", Warning)
    dolfinx_mpc = None

from .boundarycondition import BoundaryCondition


def _build_mpc(function_space, bcs):
    bcs_strong = BoundaryCondition.get_dirichlet_BCs(bcs)
    bcs_mpc = BoundaryCondition.get_mpc_BCs(bcs)

    if len(bcs_mpc) == 0:
        return None

    mpc = dolfinx_mpc.MultiPointConstraint(function_space)

    for bc in bcs_mpc:
        if bc.type == 'periodic':
            facet_tags, marker, slave_to_master_map = bc.bc
            mpc.create_periodic_constraint_topological(function_space,
                                                       facet_tags,
                                                       marker,
                                                       slave_to_master_map,
                                                       bcs_strong)
        else:
            raise TypeError("Unsupported boundary condition {0:s}".format(bc.type))

    mpc.finalize()
    return mpc
