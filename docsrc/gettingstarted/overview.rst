Overview
========

Introduction
------------
ElastodynamiCSx deals with the following PDE:

.. math::
  \mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{b},

where :math:`\mathbf{u}`, :math:`\mathbf{v}=\partial_t \mathbf{u}`, :math:`\mathbf{a}=\partial_t^2\mathbf{u}` are the displacement, velocity and acceleration fields, :math:`\mathbf{M}`, :math:`\mathbf{C}` and :math:`\mathbf{K}` are the mass, damping and stiffness forms, and :math:`\mathbf{b}` is an applied force. For time domain problems :math:`\mathbf{K}` may be a non-linear function of :math:`\mathbf{u}`.

The library provides a high level interface to build and solve common problems in a few lines code.



Build problems
--------------
Using the ``elastodynamicsx.pde`` package:

.. jupyter-execute::
    :hide-code:

    import numpy as np

    from dolfinx import mesh, fem, default_scalar_type
    from mpi4py import MPI

    degElement = 1
    length, height = 1, 1
    Nx, Ny = 10 // degElement, 10 // degElement

    # create the mesh
    extent = [[0., 0.], [length, height]]
    domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny], mesh.CellType.triangle)

    # create the function space
    V = fem.functionspace(domain, ("Lagrange", degElement, (domain.geometry.dim,)))

    from elastodynamicsx.utils import make_facet_tags, make_cell_tags
    # define some tags
    tag_top = 1
    boundaries = [(tag_top, lambda x: np.isclose(x[1], height)),]
    subdomains = [(1, lambda x: x[0] <= length/2),\
                  (2, lambda x: np.logical_and(x[0] >= length/2, x[1] <= height/2)),
                  (3, lambda x: np.logical_and(x[0] >= length/2, x[1] >= height/2))]

    cell_tags = make_cell_tags(domain, subdomains)
    facet_tags = make_facet_tags(domain, boundaries)


* Common **material laws**:

  .. jupyter-execute::

      # V is a dolfinx.fem.FunctionSpace
      # cell_tags is a dolfinx.mesh.MeshTags object

      from elastodynamicsx.pde import material

      tag_mat1 = 1        # suppose this refers to cells associated with material no 1
      tags_mat2 = (2, 3)  # same for material no 2
      mat1 = material((V, cell_tags, tag_mat1), 'isotropic', rho=1, lambda_=2, mu=1)
      mat2 = material((V, cell_tags, tags_mat2), 'isotropic', rho=2, lambda_=4, mu=2)

  * linear elasticity:
    *scalar*, *isotropic*, *cubic*, *hexagonal*, *trigonal*, *tetragonal*, *orthotropic*, *monoclinic*, *triclinic*

    * damping laws: *Rayleigh* damping

  * hyperelastic:
    *Saint Venant-Kirchhoff*, *Murnaghan*
  * perfectly matched layers (PML): in the near future...

* Common **boundary conditions** (BCs):

  .. jupyter-execute::

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

* User-defined material laws and BCs, using the ```ufl``` library:

  .. dropdown:: Custom material laws

    Specify :math:`\mathbf{M}`, :math:`\mathbf{C}` and :math:`\mathbf{K}`:

    .. jupyter-execute::

        import ufl

        # ###
        # Here we re-implement mat1 using the interface for custom material laws
        dx_mat1 = ufl.Measure("dx", domain=V.mesh, subdomain_data=cell_tags)(tag_mat1)

        # mass form
        m = lambda u, v: 1 * ufl.inner(u, v) * dx_mat1

        # stiffness form
        epsilon = lambda u: ufl.sym(ufl.grad(u))
        sigma = lambda u: 2 * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * 1 * epsilon(u)
        k = lambda u, v: ufl.inner(sigma(u), epsilon(v)) * dx_mat1

        mat1_user_defined = material(V, 'custom', is_linear=True, M_fn=m, K_fn=k)

  .. dropdown:: Custom BCs

    Specify :math:`\mathbf{C}`, :math:`\mathbf{K}` and :math:`\mathbf{b}`:

    .. jupyter-execute::

        import ufl

        # ###
        # Here we re-implement bc1 using the interface for custom BCs
        ds_bc1 = ufl.Measure("ds", domain=V.mesh, subdomain_data=facet_tags)(tag_top)

        # right hand side term
        b = lambda v: ufl.inner(T_N, v) * ds_bc1

        bc1_user_defined = boundarycondition(V , 'custom', b_fn=b)

* Define **body forces**:

  .. jupyter-execute::

      from elastodynamicsx.pde import BodyForce

      amplitude = default_scalar_type([0, 0])  # a dummy load amplitude
      def shape_x(x):
          x1, x2 = 0.2, 0.3
          y1, y2 = 0.4, 0.5
          return (x[0] >= x1) * (x[0] <= x2) * (x[1] >= y1) * (x[1] <= y2)  # a dummy shape

      f_body = fem.Function(V)
      f_body.interpolate(lambda x: amplitude[:, np.newaxis] * shape_x(x)[np.newaxis, :])
      f1 = BodyForce((V, cell_tags, None), f_body)  # None for the entire domain

* **Assemble** several materials, BCs and body forces into a *PDE* instance:

  .. jupyter-execute::

      from elastodynamicsx.pde import PDE

      pde = PDE(V, materials=[mat1, mat2], bodyforces=[f1], bcs=[bc1])

      # M, C, K, b form functions: pde.M_fn, pde.C_fn, pde.K_fn, pde.b_fn
      # eigs / freq. domain -> M, C, K matrices:    pde.M(),  pde.C(),  pde.K()
      # waveguides          -> K0, K1, K2 matrices: pde.K0(), pde.K1(), pde.K2()

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

        .. jupyter-execute::
            :hide-output:

            # Time integration
            from elastodynamicsx.solvers import TimeStepper

            dt, num_steps = 0.01, 100  # t=[0..1)

            # Define a function that will update the source term at each time step
            def update_T_N_function(t):
                forceVector = default_scalar_type([0, 1])
                T_N.value   = np.sin(t) * forceVector

            # Initialize the time stepper: compile forms and assemble the mass matrix
            tStepper = TimeStepper.build(V,
                                         pde.M_fn, pde.C_fn, pde.K_fn, pde.b_fn, dt, bcs=pde.bcs,
                                         scheme='newmark')

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
            fdsolver.solve(omega=1.0, out=u.x.petsc_vec)

            # Plot
            from elastodynamicsx.plot import plotter
            p = plotter(u, complex='real')
            p.show()

    .. tab:: Eigenmodes

        .. jupyter-execute::
            :hide-output:

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
            eps.plot(function_space=V)              # V is a dolfinx.fem.FunctionSpace

    .. tab:: Guided waves

        At present it is possible to compile the required matrices to build the eigenvalue problem,
        but a high-level solver is not implemented yet. One has to use ``slepc4py``.

        .. code-block:: python

            # PETSc.Mat matrices
            M = pde.M()
            K0, K1, K2 = pde.K0(), pde.K1(), pde.K2()

            # High-level solver: in the future...


Post-process solutions
----------------------
Using the ``elastodynamicsx.solutions`` package:

* **Eigenmodes** solutions:

.. jupyter-execute::
    :hide-output:

    # eps is a elastodynamicsx.solvers.EigenmodesSolver
    # eps.solve() has already been performed

    # Get the solutions
    mbasis = eps.getModalBasis()  # a elastodynamicsx.solutions.ModalBasis

    # Access data
    eigenfreqs = mbasis.fn     # a np.ndarray
    modeshape5 = mbasis.un[5]  # a PETSc.Vec vector

    # Visualize
    mbasis.plot(function_space=V)  # V is a dolfinx.fem.FunctionSpace

