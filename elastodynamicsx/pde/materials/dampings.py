# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
.. role:: python(code)
   :language: python

The *damping* module implements laws to model attenuation. A Damping instance
is meant to be connected to a Material instance and thereby provides the damping form
of the PDE.

The recommended way to instanciate a damping law is to use the :python:`damping` function.
"""

from typing import List


class Damping():
    """Dummy base class for damping laws"""

    labels: List[str]

    def C_fn(self, u, v):
        """The damping form"""
        print('supercharge me')
        raise NotImplementedError


class NoDamping(Damping):
    """No damping"""

    labels: List[str] = ['none']

    def C_fn(self, u, v):
        """
        The damping form

        Returns:
            None
        """
        return None


class RayleighDamping(Damping):
    """Rayleigh damping law, i.e. whose damping form is a linear combination
    of the mass and stiffness forms of the host material

    .. math::
      c(u,v) = \eta_m  m(u,v) + \eta_k  k(u,v)
    """

    labels: List[str] = ['rayleigh']

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

    def C_fn(self, u, v):
        return self.eta_m * self._material.M_fn(u, v) + self.eta_k * self._material.K_fn(u, v)

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


# ## ### ## #
all_dampings = [NoDamping, RayleighDamping]


def damping(type_: str, *args) -> Damping:
    """
    Builder method that instanciates the desired damping law type

    Args:
        type_: Available options are:

            - 'none'
            - 'rayleigh'

        *args: Passed to the required damping law

    Returns:
        An instance of the desired damping law

    Example:
        .. highlight:: python
        .. code-block:: python

          from elastodynamicsx.pde import damping, material

          dmp = damping('none')
          dmp = damping('rayleigh', eta_m, eta_k)

          mat = material(V, 'isotropic', rho, lambda_, mu, damping=dmp)
    """
    for Damp in all_dampings:
        assert issubclass(Damp, Damping)
        if type_.lower() in Damp.labels:
            return Damp(*args)

    raise TypeError("Unknown damping law: {0:s}".format(type_))
