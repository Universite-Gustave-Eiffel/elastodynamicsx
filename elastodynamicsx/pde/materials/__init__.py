# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

"""
.. role:: python(code)
  :language: python

The *materials* module contains classes for a representation of a PDE of the kind:

    M*a + C*v + K(u) = 0

i.e. the lhs of a PDE such as defined in the PDE class. An instance represents
a single (possibly arbitrarily space-dependent) material.

| The preferred way to call a material law is using the :python:`material` function:

.. highlight:: python
.. code-block:: python

  from elastodynamicsx.pde import material
  mat = material( (function_space, cell_tags, marker), type_, *args, **kwargs)
"""

from .material import *
from .elasticmaterial import *
from .anisotropicmaterials import *
from .hyperelasticmaterial import *
from .damping              import *
