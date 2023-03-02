# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import ufl

def epsilon_vector(u): return ufl.sym(ufl.grad(u))
def epsilon_scalar(u): return ufl.nabla_grad(u)



def get_epsilon_function(dim, nbcomps):
    if nbcomps == 0: #scalar function space
        return epsilon_scalar
    
    elif dim == nbcomps:
        return epsilon_vector
    
    #TODO: more cases
    
    else:
        raise NotImplementedError('dim = ' + str(dim) + ', nbcomps = ' + str(nbcomps))



def get_epsilonVoigt_function(dim, nbcomps):
    if nbcomps == 0: #scalar function space
        return epsilon_scalar #TODO: pas sur que ca marche partout...

    if   dim == 1: # 1D
        if   nbcomps == 1:
            return epsilon_scalar
        elif nbcomps == 2: # [exx, 0, uy_x]
            return lambda u: ufl.as_vector([u[0].dx(0),
                                            0,
                                            u[1].dx(0)])
        elif nbcomps == 3: # [exx, 0, 0, 0, uz_x, uy_x]
            return lambda u: ufl.as_vector([u[0].dx(0),
                                            0,
                                            0,
                                            0,
                                            u[2].dx(0),
                                            u[1].dx(0)])

    if   dim == 2: # 2D
        if   nbcomps == 2: # [exx, eyy, 2exy]
            return lambda u: ufl.as_vector([u[0].dx(0),
                                            u[1].dx(1),
                                            u[0].dx(1)+u[1].dx(0)])
        elif nbcomps == 3: # [exx, eyy, 0, uz_y, uz_x, 2exy]
            return lambda u: ufl.as_vector([u[0].dx(0),
                                            u[1].dx(1),
                                            0,
                                            u[2].dx(1),
                                            u[2].dx(0),
                                            u[0].dx(1)+u[1].dx(0)])
        else:
            return lambda u: print('kinematics:: ERROR:: epsilon operator requires 3 components')

    elif dim == 3: # 3D
        if   nbcomps == 3: # [exx, eyy, ezz, 2eyz, 2exz, 2exy]
            return lambda u: ufl.as_vector([u[0].dx(0),
                                            u[1].dx(1),
                                            u[2].dx(2),
                                            u[1].dx(2)+u[2].dx(1),
                                            u[0].dx(2)+u[2].dx(0),
                                            u[0].dx(1)+u[1].dx(0)])
        else:
            return lambda u: print('kinematics:: ERROR:: epsilon operator requires 3 components')



def get_L_operators(dim, nbcomps, k_nrm=None):
    """
    Args:
        k_nrm: (dim==1 only) A unitary vector (len=3) representing the phase direction
            typically: ufl.as_vector([0,ay,az])
            default  : ufl.as_vector([0,1,0])
    """
    L_cs = lambda *a: print('kinematics, L operators: NotImplemented') #cross section
    L_oa = None #on axis

    if   dim == 1:
        if   nbcomps == 0: #TODO: scalar guide
            pass
        elif nbcomps == 2:
            L_cs = lambda u: ufl.as_vector([u[0].dx(0),
                                            0,
                                            u[1].dx(0)])
            L_oa = lambda u: ufl.as_vector([0,
                                            u[1],
                                            u[0]])
        elif nbcomps == 3:
            if k_nrm is None:
                k_nrm = ufl.as_vector([0,1,0])
            L_cs = lambda u: ufl.as_vector([u[0].dx(0),
                                            0,
                                            0,
                                            0,
                                            u[2].dx(0),
                                            u[1].dx(0)])
            L_oa = lambda u: ufl.as_vector([0,
                                            k_nrm[1]*u[1],
                                            k_nrm[2]*u[2],
                                            k_nrm[2]*u[1] + k_nrm[1]*u[2],
                                            k_nrm[2]*u[0],
                                            k_nrm[1]*u[0]])

    elif dim == 2:
        if   nbcomps == 0: #TODO: scalar guide
            pass
        elif nbcomps == 3: #2D, 3 components: ok
            L_cs = lambda u: ufl.as_vector([u[0].dx(0),
                                            u[1].dx(1),
                                            0,
                                            u[2].dx(1),
                                            u[2].dx(0),
                                            u[0].dx(1)+u[1].dx(0)])
            L_oa = lambda u: ufl.as_vector([0, 0, u[2], u[1], u[0], 0])
        else: #2D, 1 or 2 components: not ok -> default
            L_cs = lambda *a: print('kinematics:: ERROR:: L operators require 1 or 3 components')

    else: #TODO: 3D
        pass

    return L_cs, L_oa

