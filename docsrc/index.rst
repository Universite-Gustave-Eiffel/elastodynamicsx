.. ElastoDynamiCSx documentation master file, created by
   sphinx-quickstart on Thu Oct  5 14:19:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ElastodynamiCSx documentation
=============================

ElastodynamiCSx is a Python library dedicated to the numerical modeling of wave propagation in solids using the `FEniCSx <https://fenicsproject.org/>`_ Finite Elements library. It is a third-party project, not part of the Fenics project.

The library provides a high level interface to build and solve common problems in a few lines code.

.. admonition:: Prerequisite

   This documentation only focuses on features proper to ElastodynamiCSx. Readers should be familiar with:

   * The **FEniCSx library**. See:

      * The `FEniCSx tutorial <https://jsdokken.com/dolfinx-tutorial/>`_,
      * The `dolfinx <https://docs.fenicsproject.org/dolfinx/v0.7.3/python/>`_ module documentation,
      * The `Unified Form Language <https://docs.fenicsproject.org/ufl/2023.2.0/manual/form_language.html>`_ (*ufl*) documentation,
      * The reference page for FEniCSx documentation: https://docs.fenicsproject.org/.

   * A **mesher software**. See for instance:

      * `here <https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html>`_ and `here <https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html>`_ to use GMSH,
      * `here <https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html#convert-msh-files-to-xdmf-using-meshio>`_ to import a mesh file of another format using the `meshio <https://pypi.org/project/meshio/>`_ library.

   Another good starting point can be the `COmputational MEchanics Tours <https://bleyerj.github.io/comet-fenicsx/>`_.


.. toctree::
   :maxdepth: 2
   :caption: Using ElastodynamiCSx
   :hidden:

   gettingstarted/installation
   gettingstarted/overview
   demos/_ln_demo/README
   api/api_index

.. toctree::
   :maxdepth: 2
   :caption: About
   :hidden:

   about/authors
   about/links
   about/licence_link


Pages index
-----------

* :ref:`genindex`

