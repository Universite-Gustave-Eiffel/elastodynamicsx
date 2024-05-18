# Copyright (C) 2024 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from .material import Material


class CustomMaterial(Material):
    """
    TODO
    """
    labels = ['custom']

    def __init__(self, functionspace_tags_marker, **kwargs):
        rho = None
        is_linear = kwargs.get('is_linear', None)
        super().__init__(functionspace_tags_marker, rho, is_linear, **kwargs)

        def fn_None(u, v):
            return None

        self.M_fn = kwargs.pop('M_fn', fn_None)
        self.C_fn = kwargs.pop('C_fn', fn_None)
        self.K_fn = kwargs.pop('K_fn', fn_None)

    @property
    def rho(self):
        """Disabled"""
        raise AttributeError
