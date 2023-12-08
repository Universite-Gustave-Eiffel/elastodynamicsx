# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT


class Damping: pass
def damping(type_, *args) -> Damping:
    """
    Builder method that instanciates the desired damping law type

    Args:
        type_: Available options are:
            'none', 'rayleigh'

        args:   Passed to the required damping law

    Returns:
        An instance of the desired damping law

    Examples of use:
        dmp = damping('none')
        dmp = damping('rayleigh', eta_m, eta_k)
    """
    for Damp in all_dampings:
        if type_.lower() in Damp.labels:
            return Damp(*args)

    raise TypeError("Unknown damping law: {0:s}".format(type_))


class Damping():
    """Dummy base class for damping laws"""

    @property
    def c(self):
        print('supercharge me')
        raise NotImplementedError


class NoDamping(Damping):
    """no damping"""
    labels = ['none']

    @property
    def c(self):
        """The damping form"""
        return None


class RayleighDamping(Damping):
    """Rayleigh damping law: c(u,v) = eta_m * m(u,v) + eta_k * k(u,v)"""
    labels = ['rayleigh']

    def __init__(self, eta_m, eta_k):
        """
        Args:
            eta_m: Parameter of the mass-matrix part of the damping
            eta_k: Parameter of the stiffness-matrix part of the damping
        """
        self._eta_m = eta_m
        self._eta_k = eta_k
        self._material = None


    @property
    def eta_m(self):
        """Parameter of the mass-matrix part of the damping"""
        return self._eta_m


    @property
    def eta_k(self):
        """Parameter of the stiffness-matrix part of the damping"""
        return self._eta_k


    @property
    def c(self):
        """The damping form"""
        return lambda u,v: self.eta_m * self._material.m(u,v) + self.eta_k * self._material.k(u,v)


    @property
    def host_material(self):
        """Host material from whom the mass and stiffness matrices are copied"""
        return self._material


    def link_material(self, host_material):
        """
        Connects to a host material from whom the mass and stiffness matrices
        will be copied
        """
        self._material = host_material


# ### ### ###
all_dampings = [NoDamping, RayleighDamping]
