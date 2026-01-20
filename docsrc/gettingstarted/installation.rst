Installation
============
**Requires FEniCSx binaries installed**

Quick way: pip-git install the package:

.. code-block:: bash

    pip install git+https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git

Or, git-clone and pip-install:

.. code-block:: bash

    git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
    cd elastodynamicsx/
    pip install *

This way, the demos are copied and can be run:

.. code-block:: bash

    python3 demo/weq_2D-SH_FullSpace.py


Dependencies
------------

**Main dependencies:**

* `FEniCSx / DOLFINx <https://github.com/FEniCS/dolfinx#installation>`_.

.. jupyter-execute::
    :hide-code:

    import dolfinx
    print(f"DOLFINx version: {dolfinx.__version__}")

* `DOLFINx-MPC <https://github.com/jorgensd/dolfinx_mpc>`_. This dependency is optional (periodic BCs).

.. jupyter-execute::
    :hide-code:

    from importlib.metadata import version
    print(f"DOLFINx-MPC version: {version('dolfinx_mpc')}")

* ``numpy``

* ``pyvista`` and ``matplotlib`` for 3D/2D plots


| **Optional packages:**
| ``tqdm`` (progress bar for loops)

