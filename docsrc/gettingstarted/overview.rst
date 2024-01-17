Overview
========

Introduction
------------
ElastodynamiCSx deals with the following PDE:

.. math::
  \mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{b},

where :math:`\mathbf{u}`, :math:`\mathbf{v}=\partial_t \mathbf{u}`, :math:`\mathbf{a}=\partial_t^2\mathbf{u}` are the displacement, velocity and acceleration fields, :math:`\mathbf{M}`, :math:`\mathbf{C}` and :math:`\mathbf{K}` are the mass, damping and stiffness forms, and :math:`\mathbf{b}` is an applied force. For time domain problems :math:`\mathbf{K}` may be a non-linear function of :math:`\mathbf{u}`.



Build problems
--------------
Using the ``elastodynamicsx.pde`` package:

* Common **material laws**:

  .. code-block:: python

      # V is a dolfinx.fem.function_space
      # cell_tags is a dolfinx.mesh.MeshTags object

      from elastodynamicsx.pde import material

      tag_mat1 = 1        # suppose this refers to cells associated with material no 1
      tags_mat2 = (2, 3)  # same for material no 2
      mat1 = material((V, cell_tags, tag_mat1), 'isotropic', rho=1, lambda_=2, mu=1)
      mat2 = material((V, cell_tags, tag_mat2), 'isotropic', rho=2, lambda_=4, mu=2)

  * linear elasticity:
    *scalar*, *isotropic*, *cubic*, *hexagonal*, *trigonal*, *tetragonal*, *orthotropic*, *monoclinic*, *triclinic*

    * damping laws: *Rayleigh* damping

  * hyperelastic:
    *Saint Venant-Kirchhoff*, *Murnaghan*
  * perfectly matched layers (PML): in the near future...

* Common **boundary conditions** (BCs):

  .. code-block:: python

      # facet_tags is a dolfinx.mesh.MeshTags object

      from elastodynamicsx.pde import boundarycondition

      tag_top = 1  # top boundary
      T_N  = fem.Constant(V.mesh, default_scalar_type([0, 1]))  # boundary load
      bc1  = boundarycondition((V, facet_tags, tag_top) , 'Neumann', T_N)  # prescribed traction

  * BCs involving :math:`\mathbf{u}` and :math:`\boldsymbol{\sigma} . \mathbf{n}`:
    *Free*, *Clamp*, *Dirichlet*, *Neumann*, *Robin*
  * BCs involving :math:`\mathbf{v}` and :math:`\boldsymbol{\sigma} . \mathbf{n}`:
    *Dashpot*
  * Multi-point constraint BCs:
    *Periodic*

* Define **body forces**:

  .. code-block:: python

      from elastodynamicsx.pde import BodyForce

      amplitude = default_scalar_type([0, 0])  # a dummy load amplitude
      def shape_x(x):
          return np.ones_like(x)  # a dummy shape

      f_body = fem.Function(V)
      f_body.interpolate(lambda x: amplitude[:,np.newaxis] * shape_x(x)[np.newaxis,:])
      f1 = BodyForce(V, f_body)  # not specifying cell_tags and a specific tag means the entire domain

* **Assemble** several materials, BCs and body forces into a *PDE* instance:

  .. code-block:: python

      from elastodynamicsx.pde import PDE

      pde = PDE(V, materials=[mat1, mat2], bodyforces=[f1], bcs=[bc1])

      # m, c, k, L form functions: pde.m, pde.c, pde.k, pde.L
      # eigs / freq. domain -> M, C, K matrices:    pde.M(),  pde.C(),  pde.K()
      # waveguides          -> K1, K2, K3 matrices: pde.K1(), pde.K2(), pde.K3()

  * Get the :math:`\mathbf{M}`, :math:`\mathbf{C}`, :math:`\mathbf{K}` weak forms - ``ufl`` format
  * Compile the :math:`\mathbf{M}`, :math:`\mathbf{C}`, :math:`\mathbf{K}` matrices - ``petsc`` format

* Build the weak form of a **time domain** problem

  * Explicit schemes:
    *leapfrog*
  * Implicit schemes (currently restricted to linear PDEs):
    *Newmark-beta*, *midpoint*, *linear acceleration*, *HHT-alpha*, *generalized-alpha*



Solve problems
--------------
Using the ``elastodynamicsx.solvers`` package:

.. tabs::

    .. tab:: Time domain

        .. code-block:: python

            # Time integration
            from elastodynamicsx.solvers import TimeStepper

            dt, num_steps = 0.01, 100  # t=[0..1)

            # Define a function that will update the source term at each time step
            def update_T_N_function(t):
                forceVector = default_scalar_type([0, 1])
                T_N.value   = np.sin(t) * forceVector

            # Initialize the time stepper: compile forms and assemble the mass matrix
            tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, scheme='leapfrog')

            # Define the initial values
            tStepper.set_initial_condition(u0=[0, 0], v0=[0, 0], t0=0)

            # Solve: run the loop on time steps; live-plot the result every 10 steps
            tStepper.solve(num_steps-1,
                           callfirsts=[update_T_N_function],
                           callbacks=[],
                           live_plotter={'refresh_step':10, 'clim':[-1, 1]})

    .. tab:: Frequency domain

        .. code-block:: python

            # Frequency domain
            from elastodynamicsx.solvers import FrequencyDomainSolver

            assert np.issubdtype(default_scalar_type, np.complexfloating), \
                   "Should only be executed with DOLFINx complex mode"

            # MPI communicator
            comm = V.mesh.comm

            # (PETSc) Mass, damping, stiffness matrices
            M, C, K = pde.M(), pde.C(), pde.K()

            # (PETSc) load vector
            b = pde.b()
            b_update_function = pde.update_b_frequencydomain

            # Initialize the solver
            fdsolver = FrequencyDomainSolver(comm,
                                             M,
                                             C,
                                             K,
                                             b,
                                             b_update_function=b_update_function)

            # Solve
            u = fem.Function(V, name='solution')
            fdsolver.solve(omega=1.0, out=u.vector)

            # Plot
            from elastodynamicsx.plot import plotter
            p = plotter(u, complex='real')
            p.show()

    .. tab:: Eigenmodes

        .. code-block:: python

            # Normal modes
            from elastodynamicsx.solvers import EigenmodesSolver

            # MPI communicator
            comm = V.mesh.comm

            # (PETSc) Mass, damping, stiffness matrices
            M, K = pde.M(), pde.K()
            C = None  # Enforce no damping

            nev = 9  # Number of modes to compute

            # Initialize the solver
            eps = EigenmodesSolver(comm,
                                   M,
                                   C,
                                   K,
                                   nev=nev)

            # Solve
            eps.solve()

            # Plot
            eigenfreqs = eps.getEigenfrequencies()  # a np.ndarray
            eps.plot(function_space=V)              # V is a dolfinx.fem.function_space

    .. tab:: Guided waves

        At present it is possible to compile the required matrices to build the eigenvalue problem,
        but a high-level solver is not implemented yet. One has to use ``slepc4py``.

        .. code-block:: python

            # PETSc.Mat matrices
            M = pde.M()
            K1, K2, K3 = pde.K1(), pde.K2(), pde.K3()

            # High-level solver: in the future...


Post-process solutions
----------------------
Using the ``elastodynamicsx.solutions`` package:

* **Eigenmodes** solutions:

.. code-block:: python

    # eps is a elastodynamicsx.solvers.EigenmodesSolver
    # eps.solve() has already been performed

    # Get the solutions
    mbasis = eps.getModalBasis()  # a elastodynamicsx.solutions.ModalBasis

    # Access data
    eigenfreqs = mbasis.fn     # a np.ndarray
    modeshape5 = mbasis.un[5]  # a PETSc.Vec vector

    # Visualize
    mbasis.plot(function_space=V)  # V is a dolfinx.fem.function_space
