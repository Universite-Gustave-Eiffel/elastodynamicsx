Demos
=====

The examples can be run in parallel with:

.. code-block:: bash

  # run on 2 nodes:
  mpiexec -n 2 python3 example.py

Several examples rely on functions that are not embedded in the main .py file (models, analytical solutions). Companion files can be found in the `demo/ <https://github.com/Universite-Gustave-Eiffel/elastodynamicsx/tree/main/demo>`_ folder in github.


Time domain
-----------

| **Explicit** scheme
|     *Wave propagation modeling*, *Spectral elements*

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :glob:
   
   weq*


| **Implicit** scheme
|     *Structural dynamics*

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :glob:
   
   tdsdyn*


Frequency domain -- Helmoltz equation
-------------------------------------

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :glob:
   
   freq*


Eigenmodes
----------

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :glob:
   
   eigs*


Guided waves
------------

* *coming soon*
