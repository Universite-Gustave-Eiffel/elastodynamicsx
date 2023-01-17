# ElastodynamiCSx 
ElastodynamiCSx is dedicated to the numerical modeling of wave propagation in solids using the [FEniCSx](https://fenicsproject.org/) Finite Elements library. It deals with the following PDE:

$$\mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{F},$$

where $\mathbf{u}$, $\mathbf{v}=\partial_t \mathbf{u}$, $\mathbf{a}=\partial_t^2\mathbf{u}$ are the displacement, velocity and acceleration fields, $\mathbf{M}$, $\mathbf{C}$ and $\mathbf{K}$ are the mass, damping and stiffness forms, and $\mathbf{F}$ is an applied force. $\mathbf{K}$ may be a non-linear function of $\mathbf{u}$. Various kinds of boundary conditions are supported.

The module provides high level classes to build and solve common problems in a few lines code.

## Build problems
Using the **pde** package:
  * Common boundary conditions, using the **BoundaryCondition** class
    * BCs involving $\mathbf{u}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Free*, *Clamp*, *Dirichlet*, *Neumann*, *Robin*
    * BCs involving $\mathbf{v}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Dashpot*
  * Common material laws, using the **Material** class
    * linear: scalar, isotropic elasticity
      * damping laws: Rayleigh damping
    * hyperelastic: in the near future...
  * **BodyForce** class
  * **PDE** class: assembles several materials and body forces

## Solve problems
Using the **solvers** package:
  * Time-domain problems, using the **TimeStepper** class
    * Explicit schemes: *leap frog*
    * Implicit schemes: *Newmark-beta*, *midpoint*, *linear acceleration*, *HHT-alpha*, *generalized-alpha*
  * Eigenmodes problems, using the **ElasticResonanceSolver** class

## Dependencies
ElastodynamiCSx requires FEnicsX / dolfinx -> see [instructions here](https://github.com/FEniCS/dolfinx#installation). Tested with v0.4.1 and v0.5.1.

### Other required packages
**numpy**  
**scipy**  
**matplotlib**  
**pyvista**  

### Optional packages
**tqdm**

## Installation
### With Fenicsx binaries installed
Clone the repository and install the package:
```bash
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
pip3 install ./elastodynamicsx/

#test
python3 elastodynamicsx/examples/weq_2D-SH_FullSpace.py
```

### Inside a Fenicsx Docker image
Here we show how to build the docker image and propose two ways to use it. In each case the container is given the right to display graphics. The solution adopted to avoid MIT-SHM errors due to sharing host display :0 is to disable IPC namespacing with --ipc=host. It is given [here](https://github.com/jessfraz/dockerfiles/issues/359), although described as not totally satisfacory because of isolation loss. Other more advanced solutions are also given in there.

1. Clone the repository and build a docker image called 'ElastodynamiCSx:latest':
```bash
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git

#the image relies on dolfinx/dolfinx:stable (see Dockerfile)
docker build -t ElastodynamiCSx:latest ./elastodynamicsx
```
2. Create a single-use container from this image and allows it to display graphics:
```bash
#creates a folder meant to be shared with the docker container
shareddir=docker_shared
mkdir $shareddir

#copy examples into that shared folder
cp -r elastodynamicsx/examples $shareddir

#grant access to root to the graphical backend (the username inside the container will be 'root')
#without this access matplotlib and pyvista won't display
xhost + si:localuser:root

#creates a container that will self destroy on close
docker run -it --rm --ipc=host --net=host --env="DISPLAY" -v $(shareddir):/root/$(shareddir) -w /root/$(shareddir) --volume="$HOME/.Xauthority:/root/.Xauthority:rw" ElastodynamiCSx:latest /bin/bash

###
#at this point we are inside the container
###

#test
python3 examples/weq_2D-SH_FullSpace.py
```
2. (alternative) Create a container called 'elastoCSx' that will remain after close and can be accessed through several tabs simultaneously:
```bash
#creates a folder meant to be shared with the docker container
shareddir=docker_shared
mkdir $shareddir

#copy examples into that shared folder
cp -r elastodynamicsx/examples $shareddir

#grant access to root to the graphical backend (the username inside the container will be 'root')
#without this access matplotlib and pyvista won't display
xhost + si:localuser:root

#creates a container that will remain after close
docker run -it --name elastoCSx --ipc=host --net=host --env="DISPLAY" -v $(shareddir):/root/$(shareddir) -w /root/$(shareddir) --volume="$HOME/.Xauthority:/root/.Xauthority:rw" ElastodynamiCSx:latest bash

###
#at this point we are inside the 'elastoCSx' container
###

#test
python3 examples/weq_2D-SH_FullSpace.py
```
To re-use the 'elastoCSx' container, possibly from several shell tabs or windows simultaneously, use the following:
```bash
#this needs to be re-executed once after a reboot
xhost + si:localuser:root

#starts the container
docker start elastoCSx

#enters the container; this line can be repeated in another tab or window
docker exec -it elastoCSx bash

###
#at this point we are inside the 'elastoCSx' container
###

#test
python3 examples/weq_2D-SH_FullSpace.py
```

## Examples
Several examples are provided in the **examples** subfolder:
  * Wave equation, time domain:
    * (2D) homogeneous space, anti-plane line load (SH waves): *weq_2D-SH_FullSpace.py*
    * (2D) homogeneous space, in-plane line load (P-SV waves): *weq_2D-PSV_FullSpace.py*
  * Structural dynamics, time domain:
    * (3D) forced vibration of an elastic beam clamped at one end; with Rayleigh damping: *tdsdyn_3D_ElasticBeam.py*
  * Eigenmodes:
    * (3D) resonances of an elastic beam clamped at one end: *eigs_3D_ElasticBeam.py*
    * (3D) resonances of an aluminum cube: *eigs_3D_AluminumCube.py*

Reference for the analytical solutions:
  * Kausel, E. (2006). Fundamental solutions in elastodynamics: a compendium. Cambridge University Press.

## Useful links
Part of the code is largely inspired from:
  * [The FEniCSx tutorial](https://jorgensd.github.io/dolfinx-tutorial/)

Other useful references:
  * Fenicsx:
    * [FEniCSx electromagnetic demos](https://mikics.github.io/)
    * [NewFrac FEniCSx Training](https://newfrac.gitlab.io/newfrac-fenicsx-training/index.html)
    * [multiphenicsx](https://github.com/multiphenics/multiphenicsx)
    * [Multi-point constraints with FEniCS-X](https://github.com/jorgensd/dolfinx_mpc)
  * legacy Fenics:
    * [The COmputational MEchanics Toolbox - COMET](https://comet-fenics.readthedocs.io/en/latest/)
    * [The FEniCS solid tutorial](https://fenics-solid-tutorial.readthedocs.io/en/latest/)
    * [Fenics tutorial, by Jan Blechta and Jaroslav Hron](https://www2.karlin.mff.cuni.cz/~hron/fenics-tutorial/index.html)


