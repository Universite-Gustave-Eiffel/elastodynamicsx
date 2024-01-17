Installation
============
**Option 1: With FEniCSx binaries installed**

Clone the repository and install the package:

.. code-block:: bash

    git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
    cd elastodynamicsx/
    pip3 install .

Test by running an a demo:

.. code-block:: bash

    python3 demo/weq_2D-SH_FullSpace.py

**Option 2: Build a dedicated docker image**

Because it can still be somewhat tricky to install ``fenicsx``, the package provides two docker files, for use within a shell (*Dockerfile.shell*) or within a Jupyter notebook (*Dockerfile.lab*). Here we show how to build the docker images and how to use them.

.. tabs::

    .. tab:: Use within a Jupyter notebook

        Clone the repository and build a docker image called 'elastolab:latest':

        .. code-block:: bash

            git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
            cd elastodynamicsx/
            docker build -t elastolab:latest -f Dockerfile.lab .

        Run the image and shares the folder from which the command is executed:

        .. code-block:: bash

            docker run --rm -v $(pwd):/root/shared -p 8888:8888 elastolab:latest

            # Open the URL printed on screen beginning with http://127.0.0.1:8888/?token...
            # The examples are in /root/demo; the shared folder is in /root/shared

        The backend that has been configured by default is the ``ipyvtklink`` one. It has the advantage of being almost fully compatible with the examples. However, as the rendering is performed on the server, the display suffers great lag. Other options are `described here <https://docs.pyvista.org/user-guide/jupyter/index.html>`_. For instance, when live-plotting a ``TimeStepper.solve()`` call, only the first and last images will be seen.



    .. tab:: Use within a shell

        For the shell case the container is given the right to display graphics. The solution adopted to avoid MIT-SHM errors due to sharing host display :0 is to disable IPC namespacing with --ipc=host. It is `given here <https://github.com/jessfraz/dockerfiles/issues/359>`_, although described as not totally satisfactory because of isolation loss. Other more advanced solutions are also given in there.

        Clone the repository and build a docker image called 'elastodynamicsx:latest':

        .. code-block:: bash

            git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
            cd elastodynamicsx/
            docker build -t elastodynamicsx:latest -f Dockerfile.shell .

        Run the image and shares the folder from which the command is executed:

        .. code-block:: bash

            # Grant access to root to the graphical backend (the username inside the container will be 'root')
            # Without this access matplotlib and pyvista won't display
            xhost + si:localuser:root

            # Create a container that will self destroy on close
            docker run -it --rm --ipc=host --net=host --env="DISPLAY" -v $(pwd):/root/shared --volume="$HOME/.Xauthority:/root/.Xauthority:rw" elastodynamicsx:latest bash

            ###
            # At this point we are inside the container
            #
            # The examples are in /root/demo; the shared folder is in /root/shared
            ###

            # Test
            python3 demo/weq_2D-SH_FullSpace.py



Dependencies
------------

**Main dependencies:**

* `FEniCSx / DOLFINx <https://github.com/FEniCS/dolfinx#installation>`_.

.. jupyter-execute::
    :hide-code:

    import dolfinx
    print(f"DOLFINx version: {dolfinx.__version__}")

* `DOLFINx-MPC <https://github.com/jorgensd/dolfinx_mpc>`_ v0.7.0.post1. This dependency is optional (periodic BCs).

| **Packages required for the examples:**
| numpy
| matplotlib
| pyvista
| ipyvtklink (configured pyvista backend in jupyter lab)

| **Optional packages:**
| tqdm (progress bar for loops)

