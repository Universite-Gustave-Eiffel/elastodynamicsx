.. ElastoDynamiCSx documentation master file, created by
   sphinx-quickstart on Thu Oct  5 14:19:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ElastodynamiCSx documentation
=============================

ElastodynamiCSx is a Python module dedicated to the numerical modeling of wave propagation in solids using the `FEniCSx <https://fenicsproject.org/>`_ Finite Elements library. It deals with the following PDE:

.. math::
  \mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{b},

where :math:`\mathbf{u}`, :math:`\mathbf{v}=\partial_t \mathbf{u}`, :math:`\mathbf{a}=\partial_t^2\mathbf{u}` are the displacement, velocity and acceleration fields, :math:`\mathbf{M}`, :math:`\mathbf{C}` and :math:`\mathbf{K}` are the mass, damping and stiffness forms, and :math:`\mathbf{b}` is an applied force. For time domain problems :math:`\mathbf{K}` may be a non-linear function of :math:`\mathbf{u}`.

The module provides a high level interface to build and solve common problems in a few lines code.

.. toctree::
   :maxdepth: 2

   readme_link
   demos/examples_index
   api/api_index


* :ref:`genindex`

