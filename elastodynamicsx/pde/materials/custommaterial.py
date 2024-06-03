# Copyright (C) 2024 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from .material import Material


class CustomMaterial(Material):
    """
    Interface for custom material laws. Mass, damping and stiffness forms
    are provided as callables.

    Args:
        functionspace_tags_marker: (see Material)

    Keyword Args:
        M_fn: (callable, optionnal) Mass form
        C_fn: (callable, optionnal) Damping form
        K_fn: (callable, optionnal) Stiffness form
        K0_fn: (callable, optionnal) K0 part of the stiffness form (waveguide problems)
        K1_fn: (callable, optionnal) K1 part of the stiffness form (waveguide problems)
        K2_fn: (callable, optionnal) K2 part of the stiffness form (waveguide problems)
        is_linear: (bool)

    Example:
        .. highlight:: python
        .. code-block:: python

        # a linear material
        custom_mass_fn = lambda u, v: 1.234 * ufl.inner(u, v) * ufl.dx
        custom_stiff_fn = lambda u, v: ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        custom_mat = material(V,
                              'custom',
                              is_linear=True,
                              M_fn=custom_mass_fn,
                              K_fn=custom_stiff_fn)

        # a nonlinear material
        custom_stiff_fn_nonlinear = lambda u, v: ufl.inner(u*u, v) * ufl.dx
        custom_mat = material(V,
                              'custom',
                              is_linear=False,
                              M_fn=custom_mass_fn,
                              K_fn=custom_stiff_fn_nonlinear)

        # interface to K0, K1, K2 matrices (waveguides)
        K0_fn = K1_fn = K2_fn = lambda u, v: ufl.inner(u, v) * ufl.dx
        custom_mat = material(V,
                              'custom',
                              is_linear=True,
                              M_fn=custom_mass_fn,
                              K0_fn=K0_fn,
                              K1_fn=K1_fn,
                              K2_fn=K2_fn)
    """
    labels = ['custom']

    def __init__(self, functionspace_tags_marker, **kwargs):
        rho = None
        is_linear = kwargs.pop('is_linear', None)
        super().__init__(functionspace_tags_marker, rho, is_linear, **kwargs)

        def fn_None(u, v):
            """Default behavior: returns None"""
            return None

        def fn_Disabled(u, v):
            """Disabled for custom laws"""
            raise NotImplementedError

        self.M_fn = kwargs.pop('M_fn', fn_None)
        self.C_fn = kwargs.pop('C_fn', fn_None)
        self.K_fn = kwargs.pop('K_fn', fn_None)

        self.K0_fn_CG = kwargs.pop('K0_fn', fn_Disabled)
        self.K1_fn_CG = kwargs.pop('K1_fn', fn_Disabled)
        self.K2_fn_CG = kwargs.pop('K2_fn', fn_Disabled)

    @property
    def rho(self):
        """Disabled"""
        raise AttributeError
