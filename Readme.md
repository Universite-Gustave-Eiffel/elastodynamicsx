# ElastodynamiCSx 
ElastodynamiCSx is dedicated to the numerical modeling of wave propagation in solids using the [FEniCSx](https://fenicsproject.org/) Finite Elements library. It deals with the following PDE:

$$\mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{b},$$

where $\mathbf{u}$, $\mathbf{v}=\partial_t \mathbf{u}$, $\mathbf{a}=\partial_t^2\mathbf{u}$ are the displacement, velocity and acceleration fields, $\mathbf{M}$, $\mathbf{C}$ and $\mathbf{K}$ are the mass, damping and stiffness forms, and $\mathbf{b}$ is an applied force. For time domain problems $\mathbf{K}$ may be a non-linear function of $\mathbf{u}$.

The module provides high level classes to build and solve common problems in a few lines code.

## Build problems
Using the **pde** package:
  * Common **material laws**, using the *Material* class
    * linear: *scalar*, *isotropic* elasticity
      * damping laws: *Rayleigh* damping
    * hyperelastic: *Saint Venant-Kirchhoff*, *Murnaghan*
    * perfectly matched layers (PML): in the near future...
  * Define **body forces**, using the *BodyForce* class
  * **Assemble** several materials and body forces, using the *PDE* class
  * Common **boundary conditions**, using the *BoundaryCondition* class
    * BCs involving $\mathbf{u}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Free*, *Clamp*, *Dirichlet*, *Neumann*, *Robin*
    * BCs involving $\mathbf{v}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Dashpot*
```python
#V is a dolfinx.fem.function_space
#cell_tags is a dolfinx.mesh.MeshTags object

from elastodynamicsx.pde import material, BodyForce, BoundaryCondition, PDE

#MATERIALS
tag_mat1 = 1 #suppose tag_mat1 refers to cells associated with material no 1
tag_mat2 = 2 #same for material no 2
mat1 = material((V, cell_tags, tag_mat1), 'isotropic', rho=1, lambda_=2, mu=1)
mat2 = material((V, cell_tags, tag_mat2), 'isotropic', rho=2, lambda_=4, mu=2)

#BODY LOADS
f_body = fem.Constant(V.mesh, np.array([0, 0], dtype=PETSc.ScalarType)) #body load
f1     = BodyForce(V, f_body) #not specifying cell_tags and a specific tag means the entire domain

#BOUNDARY CONDITIONS
tag_top  = 1       #top boundary
tags_lbr = (2,3,4) #left, bottom, right boundaries
T_N  = fem.Constant(V.mesh, PETSc.ScalarType([0,1])) #boundary load
bc1  = BoundaryCondition((V, facet_tags, tag_top) , 'Neumann', T_N)
bc2  = BoundaryCondition((V, facet_tags, tags_lbr), 'Dashpot', (mat1.Z_N, mat1.Z_T)) #plane-wave absorbing conditions with P-wave & S-wave impedances of material no1
bcs  = [bc1, bc2]

#PDE
pde  = PDE(V, materials=[mat1, mat2], bodyforces=[f1], bcs=bcs)
#m, c, k, L form functions: pde.m, pde.c, pde.k, pde.L
#compiled M, C, K matrices: pde.M(), pde.C(), pde.K()
```

## Solve problems
Using the **solvers** package:
  * **Time-domain problems**, using the *TimeStepper* class
    * Explicit schemes (linear & non-linear):  
       *leap frog*
    * Implicit schemes (linear):  
       *Newmark-beta*, *midpoint*, *linear acceleration*, *HHT-alpha*, *generalized-alpha*
```python
#Time integration
from elastodynamicsx.solvers import TimeStepper

dt, num_steps = 0.01, 100 #t=[0..1)

#Initialize the time stepper
tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, scheme='leapfrog')
tStepper.initial_condition(u0=[0,0], v0=[0,0], t0=0)

#Define a function that will update the source term at each time step
def update_T_N_function(t, timeStepper):
    forceVector = PETSc.ScalarType([0,1])
    T_N.value   = np.sin(t)*forceVector

#Loop on time, and live-plot the result
tStepper.run(num_steps-1, callfirsts=[update_T_N_function], callbacks=[], live_plotter={'refresh_step':1, 'clim':[-1,1]})

#the end
```

  * **Frequency domain problems**, using the *FrequencyDomainSolver* class
```python
#Frequency domain
from elastodynamicsx.solvers import FrequencyDomainSolver

assert np.issubdtype(PETSc.ScalarType, np.complexfloating), \
       "Should only be executed with DOLFINx complex mode"

#MPI communicator
comm = V.mesh.comm

#(PETSc) Mass, damping, stiffness matrices
M, C, K = pde.M(), pde.C(), pde.K()

#(PETSc) load vector
b = pde.b()
b_update_function = pde.update_b_frequencydomain

#Initialize the solver
fdsolver = FrequencyDomainSolver(comm, M, C, K, b, b_update_function=b_update_function)

#Solve
u = fem.Function(V, name='solution')
fdsolver.solve(omega=1.0, out=u.vector)

#Plot
from elastodynamicsx.plot import plotter
p = plotter(u, complex='real')
p.show()

#the end
```

  * **Eigenmodes problems**, using the *ElasticResonanceSolver* class
```python
#Normal modes
from elastodynamicsx.solvers import ElasticResonanceSolver

#MPI communicator
comm = V.mesh.comm

#(PETSc) Mass, damping, stiffness matrices
M, K = pde.M(), pde.K()

nev = 9 #number of modes to compute

#Initialize the solver
eps = ElasticResonanceSolver(comm, M, None, K, nev=nev)
eps.solve()

eigenfreqs = eps.getEigenfrequencies()
eps.plot(function_space=V) #V is a dolfinx.fem.function_space

#the end
```

  * **Guided waves problems**, in the near future...

## Post process solutions
Using the **solutions** package:
  * **Eigenmodes solutions**, using the *ModalBasis* class
```python
#eps is a elastodynamicsx.solvers.ElasticResonanceSolver
#eps.solve() has already been performed

#get the solutions
mbasis = eps.getModalBasis() #a elastodynamicsx.solutions.ModalBasis

#access data
eigenfreqs = mbasis.fn    #a np.array
modeshape5 = mbasis.un[5] #a PETSc.Vec vector

#visualize
mbasis.plot(function_space=V) #V is a dolfinx.fem.function_space
```

## Dependencies
ElastodynamiCSx requires FEniCSx / DOLFINx -> see [instructions here](https://github.com/FEniCS/dolfinx#installation). Tested with v0.4.1, v0.5.1, v0.5.2, v0.6.0.

### Packages required for the examples
numpy  
matplotlib  
pyvista  
ipyvtklink (configured pyvista backend in jupyter lab)  

### Optional packages
tqdm (progress bar for time-domain calculations)

## Installation
### Option 1: With FEniCSx binaries installed
Clone the repository and install the package:
```bash
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
cd elastodynamicsx/
pip3 install .

#test
python3 elastodynamicsx/examples/weq_2D-SH_FullSpace.py
```

### Option 2: Inside a FEniCSx Docker image
The package provides two docker files, for use with shell commands (Dockerfile.shell) or with a Jupyter notebook (Dockerfile.lab). Here we show how to build the docker images and how to use them.

##### For use with a Jupyter notebook
Clone the repository and build a docker image called 'elastolab:latest':
```bash
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
cd elastodynamicsx/

#the image relies on dolfinx/lab:stable (see Dockerfile.lab)
docker build -t elastolab:latest -f Dockerfile.lab .
```
Run the image and shares the folder from which the command is executed:
```bash
docker run --rm -v $(pwd):/root/shared -p 8888:8888 elastolab:latest

#Copy the URL printed on screen beginning with http://127.0.0.1:8888/?token...
#The examples are in /root/examples; the shared folder is in /root/shared
```
The backend that has been configured by default is the 'ipyvtklink' one. It has the advantage of being almost fully compatible with the examples. However, as the rendering is performed on the server, the display suffers great lag. Other options are described [here](https://docs.pyvista.org/user-guide/jupyter/index.html). For instance, when live-plotting a TimeStepper.run() call, only the first and last images will be seen -- in this case the Dockerfile.shell image should be preferred. Note ongoing works are [being pursued](https://github.com/pyvista/pyvista/issues/3690) to have a unique pyvista-jupyter backend.

##### For use with shell commands
For the shell case the container is given the right to display graphics. The solution adopted to avoid MIT-SHM errors due to sharing host display :0 is to disable IPC namespacing with --ipc=host. It is given [here](https://github.com/jessfraz/dockerfiles/issues/359), although described as not totally satisfactory because of isolation loss. Other more advanced solutions are also given in there.

Clone the repository and build a docker image called 'elastodynamicsx:latest':
```bash
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
cd elastodynamicsx/

#the image relies on dolfinx/dolfinx:stable (see Dockerfile.shell)
docker build -t elastodynamicsx:latest -f Dockerfile.shell .
```
Run the image and shares the folder from which the command is executed:
```bash
#grant access to root to the graphical backend (the username inside the container will be 'root')
#without this access matplotlib and pyvista won't display
xhost + si:localuser:root

#create a container that will self destroy on close
docker run -it --rm --ipc=host --net=host --env="DISPLAY" -v $(pwd):/root/shared --volume="$HOME/.Xauthority:/root/.Xauthority:rw" elastodynamicsx:latest bash

###
#at this point we are inside the container
#
#The examples are in /root/examples; the shared folder is in /root/shared
###

#test
python3 examples/weq_2D-SH_FullSpace.py
```

## Examples
Several examples are provided in the **examples** subfolder:
  * Wave equation, time domain:
    * (2D) homogeneous space, anti-plane line load (SH waves): *weq_2D-SH_FullSpace.py*
    * (2D) homogeneous space, in-plane line load (P-SV waves): *weq_2D-PSV_FullSpace.py*
  * Wave equation, frequency domain (Helmoltz equation):
    * (2D) homogeneous space, anti-plane line load (SH waves): *freq_2D-SH_FullSpace.py*
    * (2D) homogeneous space, in-plane line load (P-SV waves): *freq_2D-PSV_FullSpace.py*
  * Structural dynamics, time domain:
    * (3D) forced vibration of an elastic beam clamped at one end, with Rayleigh damping - adapted from [COMET](https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html): *tdsdyn_3D_ElasticBeam.py*
  * Eigenmodes:
    * (3D) resonances of an elastic beam clamped at one end - adapted from [COMET](https://comet-fenics.readthedocs.io/en/latest/demo/modal_analysis_dynamics/cantilever_modal.html): *eigs_3D_ElasticBeam.py*
    * (3D) resonances of an aluminum cube: *eigs_3D_AluminumCube.py*

Reference for the analytical solutions:
  * Kausel, E. (2006). Fundamental solutions in elastodynamics: a compendium. Cambridge University Press.

## Useful links
Part of the code is largely inspired from:
  * [The FEniCSx tutorial](https://jorgensd.github.io/dolfinx-tutorial/)

Other useful references:
  * FEniCSx:
    * [FEniCSx electromagnetic demos](https://mikics.github.io/)
    * [NewFrac FEniCSx Training](https://newfrac.gitlab.io/newfrac-fenicsx-training/index.html)
    * [multiphenicsx](https://github.com/multiphenics/multiphenicsx)
    * [Multi-point constraints with FEniCS-X](https://github.com/jorgensd/dolfinx_mpc)
  * legacy Fenics:
    * [The COmputational MEchanics Toolbox - COMET](https://comet-fenics.readthedocs.io/en/latest/)
    * [The FEniCS solid tutorial](https://fenics-solid-tutorial.readthedocs.io/en/latest/)
    * [FEniCS tutorial, by Jan Blechta and Jaroslav Hron](https://www2.karlin.mff.cuni.cz/~hron/fenics-tutorial/index.html)
    * [CBC.Solve](https://code.launchpad.net/cbc.solve)


