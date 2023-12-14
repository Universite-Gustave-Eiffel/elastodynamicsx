# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
The *kinematics* module is mainly designed for internal use and will probably not
be used by an external user. It contains convenience functions to build some
useful differential operators at the required ufl format, for many possible
space dimensions and number of components of the function space.
"""

from typing import Callable

import ufl


def epsilon_vector(u):
    return ufl.sym(ufl.grad(u))  # requires 'space dimension' == 'number of components'


def epsilon_scalar(u):
    return ufl.nabla_grad(u)


def get_epsilon_function(dim, nbcomps):
    if nbcomps == 0:  # scalar function space
        return epsilon_scalar

    elif dim == nbcomps:
        return epsilon_vector

    if dim == 1:
        if nbcomps == 2:  # [ [exx, exy], [eyx, eyy] ]
            return lambda u: ufl.as_matrix([[u[0].dx(0), 0.5 * u[1].dx(0)],
                                            [0.5 * u[1].dx(0), 0]])
        elif nbcomps == 3:  # [ [exx, exy, exz], [eyx, eyy, eyz], [ezx, ezy, ezz] ]
            return lambda u: ufl.as_matrix([[u[0].dx(0), 0.5 * u[1].dx(0), 0.5 * u[2].dx(0)],
                                            [0.5 * u[1].dx(0), 0, 0],
                                            [0.5 * u[2].dx(0), 0, 0]])

    elif dim == 2:
        if nbcomps == 3:  # [ [exx, exy, exz], [eyx, eyy, eyz], [ezx, ezy, ezz] ]
            return lambda u: ufl.as_matrix([[u[0].dx(0), 0.5 * (u[1].dx(0) + u[0].dx(1)), 0.5 * u[2].dx(0)],
                                            [0.5 * (u[1].dx(0) + u[0].dx(1)), u[1].dx(1), 0.5 * u[2].dx(1)],
                                            [0.5 * u[2].dx(0), 0.5 * u[2].dx(1), 0]])

    else:
        raise NotImplementedError('dim = ' + str(dim) + ', nbcomps = ' + str(nbcomps))


def get_epsilonVoigt_function(dim, nbcomps):
    epsV: Callable  # epsilonVoigt

    if nbcomps == 0:  # scalar function space
        epsV = epsilon_scalar  # TODO: pas sur que ca marche partout...
        return epsV

    if dim == 1:  # 1D
        if nbcomps == 1:
            epsV = epsilon_scalar

        elif nbcomps == 2:  # [exx, 0, uy_x]
            def epsV(u):
                return ufl.as_vector([u[0].dx(0), 0, u[1].dx(0)])

        elif nbcomps == 3:  # [exx, 0, 0, 0, uz_x, uy_x]
            def epsV(u):
                return ufl.as_vector([u[0].dx(0), 0, 0, 0, u[2].dx(0), u[1].dx(0)])

    elif dim == 2:  # 2D
        if nbcomps == 2:  # [exx, eyy, 2exy]
            def epsV(u):
                return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])

        elif nbcomps == 3:  # [exx, eyy, 0, uz_y, uz_x, 2exy]
            def epsV(u):
                return ufl.as_vector([u[0].dx(0), u[1].dx(1), 0, u[2].dx(1), u[2].dx(0), u[0].dx(1) + u[1].dx(0)])

        else:
            def epsV(u):
                print('kinematics:: ERROR, dim=2:: epsilon operator requires 3 components')

    elif dim == 3:  # 3D
        if nbcomps == 3:  # [exx, eyy, ezz, 2eyz, 2exz, 2exy]
            def epsV(u):
                return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2),
                                      u[1].dx(2) + u[2].dx(1), u[0].dx(2) + u[2].dx(0), u[0].dx(1) + u[1].dx(0)])

        else:
            def epsV(u):
                print('kinematics:: ERROR, dim=3:: epsilon operator requires 3 components')

    return epsV


def get_L_operators(dim, nbcomps, k_nrm=None):
    """
    .. role:: python(code)
       :language: python

    Args:
        dim: space dimension
        nbcomps: number of components
        k_nrm: (dim==1 only) A unitary vector (len=3) representing the phase direction.

            typically: :python:`ufl.as_vector([0,ay,az])`
            default  : :python:`ufl.as_vector([0,1,0])`
    """
    L_cs: Callable  # cross section
    L_oa: Callable  # on axis

    if nbcomps == 0:
        L_cs = epsilon_scalar

        def L_oa(u):
            return u
        return L_cs, L_oa

    if dim == 1:
        if nbcomps == 2:
            def L_cs(u):
                return ufl.as_vector([u[0].dx(0), 0, u[1].dx(0)])

            def L_oa(u):
                return ufl.as_vector([0, u[1], u[0]])

        elif nbcomps == 3:
            if k_nrm is None:
                k_nrm = ufl.as_vector([0, 1, 0])

            def L_cs(u):
                return ufl.as_vector([u[0].dx(0), 0, 0, 0, u[2].dx(0), u[1].dx(0)])

            def L_oa(u):
                return ufl.as_vector([0, k_nrm[1] * u[1], k_nrm[2] * u[2],
                                      k_nrm[2] * u[1] + k_nrm[1] * u[2], k_nrm[2] * u[0], k_nrm[1] * u[0]])

    elif dim == 2:
        if nbcomps == 3:  # 2D, 3 components: ok
            def L_cs(u):
                return ufl.as_vector([u[0].dx(0), u[1].dx(1), 0, u[2].dx(1), u[2].dx(0), u[0].dx(1) + u[1].dx(0)])

            def L_oa(u):
                return ufl.as_vector([0, 0, u[2], u[1], u[0], 0])

        else:  # 2D, 1 or 2 components: not ok -> error
            def L_cs(*args):
                print('kinematics:: ERROR:: L operators require 1 or 3 components')

            L_oa = None

    else:
        if nbcomps == 3:
            # Warning: this is a special case: K1=K, K2=K3=0. The lines below
            # don't compile because L_oa==0 -> treat it differently outside
            L_cs = get_epsilonVoigt_function(dim, nbcomps)

            def L_oa(u):
                return ufl.as_vector([0] * 6)

        else:  # 3D, 1 or 2 components: not ok -> error
            def L_cs(*args):
                print('kinematics:: ERROR:: L operators require 1 or 3 components')

            L_oa = None

    return L_cs, L_oa
