# ElastodynamiCSx 
ElastodynamiCSx is dedicated to the numerical modeling of wave propagation in solids using the [FEniCSx](https://fenicsproject.org/) Finite Elements library. It deals with the following PDE:

$$\mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{F},$$

where $\mathbf{u}$, $\mathbf{v}=\partial_t \mathbf{u}$, $\mathbf{a}=\partial_{t^2}\mathbf{u}$ are the displacement, velocity and acceleration fields, $\mathbf{M}$, $\mathbf{C}$ and $\mathbf{K}$ are the mass, damping and stiffness forms, and $\mathbf{F}$ is an applied force. $\mathbf{K}$ may be a non-linear function of $\mathbf{u}$. Various kinds of boundary conditions are supported.

The module provides high level classes to build and solve common problems in a few lines code:

build problems using the **pde** package:
  * Common boundary conditions, using the **BoundaryCondition** class
    * BCs involving $\mathbf{u}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Free*, *Clamp*, *Dirichlet*, *Neumann*, *Robin*
    * BCs involving $\mathbf{v}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Dashpot*
  * Common material laws, using the **Material** class
    * linear: scalar, isotropic elasticity
      * damping laws: Rayleigh damping
    * hyperelastic: in the near future...
  * **BodyForce** class
  * **PDE** class: Automatic assembly of several materials and body loads

solve problems using the **solvers** package:
  * Time-domain problems, using the **TimeStepper** class
    * Explicit schemes: *leap frog*
    * Implicit schemes: *Newmark-beta*, *midpoint*, *linear acceleration*, *HHT-alpha*, *generalized-alpha*
  * Eigenmodes problems, using the **ElasticResonanceSolver** class

## Dependencies
ElastodynamiCSx requires FEnicsX / dolfinx. Tested with v0.4.1 and v0.5.1 -> see [instructions here](https://github.com/FEniCS/dolfinx#installation)  

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
pip3 install .
```

### Inside a Fenicsx Docker image
For the time being the idea is to create a container from a dolfinx image and add elastodynmicsx into it. In the future the following lines should be replaced with a Dockerfile.

At first time:
```bash
#create a shared directory for the docker container
shareddir=docker_elastodynamicsx
mkdir $shareddir
cd $shareddir

#pull the code
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git

#grant access to root to the graphical backend (the username inside the container will be 'root')
#without this access matplotlib and pyvista won't display
xhost + si:localuser:root

#create a container named 'ElastodynamiCSx' from the dolfinx/dolfinx:stable image
#to avoid MIT-SHM errors due to sharing host display :0, we adopt the solution of disabling IPC namespacing with --ipc=host.
#This solution is given in https://github.com/jessfraz/dockerfiles/issues/359 , although described as not totally satisfacory because of isolation loss. Other more advanced solutions are also given there.
docker run -it --name ElastodynamiCSx --ipc=host --net=host --env="DISPLAY" -v $(pwd):/root/shared -w /root/shared --volume="$HOME/.Xauthority:/root/.Xauthority:rw" dolfinx/dolfinx:stable /bin/bash

###
#at this point we are inside the 'ElastodynamiCSx' container
#
#-> expand the container with new packages
###

#get tkinter for the 'TkAgg' matplotlib backend
apt-get update
apt-get install -y python3-tk

#install the code
cd elastodynamicsx/
pip3 install .
pip3 install tqdm
cd ..

#test
python3 elastodynamicsx/elastodynamicsx/examples/weq_2D-SH_FullSpace.py
```
The other times we just re-use the container:
```bash
#grant access to root to graphical backend for plotting inside the container
#this command needs to be re-run only once after reboot
xhost + si:localuser:root

#start the 'ElastodynamiCSx' container
docker start ElastodynamiCSx

#executes the container from the current shell tab or from any other one, possibly from several tabs or windows simultaneously
docker exec -ti ElastodynamiCSx bash

#test
python3 elastodynamicsx/elastodynamicsx/examples/weq_2D-SH_FullSpace.py
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


