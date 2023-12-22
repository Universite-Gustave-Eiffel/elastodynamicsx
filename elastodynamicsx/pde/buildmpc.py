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
    dolfinx_mpc = None  # type: ignore

from .boundaryconditions import get_dirichlet_BCs, get_mpc_BCs, BCPeriodic


def _build_mpc(bcs):
    bcs_strong = get_dirichlet_BCs(bcs)
    bcs_mpc = get_mpc_BCs(bcs)

    if len(bcs_mpc) == 0:
        return None

    function_space = bcs_mpc[0].function_space
    mpc = dolfinx_mpc.MultiPointConstraint(function_space)

    for bc in bcs_mpc:
        if isinstance(bc, BCPeriodic):
            facet_tags, marker, slave_to_master_map = bc.bc
            mpc.create_periodic_constraint_topological(function_space,
                                                       facet_tags,
                                                       marker,
                                                       slave_to_master_map,
                                                       bcs_strong)
        else:
            raise TypeError("Unsupported boundary condition")

    mpc.finalize()
    return mpc
