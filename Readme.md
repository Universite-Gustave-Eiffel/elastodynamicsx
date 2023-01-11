# ElastodynamiCSx 
ElastodynamiCSx is dedicated to the numerical modeling of wave propagation in solids using the [FEniCSx](https://fenicsproject.org/) Finite Elements library. It deals with the following PDE:

$$\mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{F},$$

where $\mathbf{u}$, $\mathbf{v}=\partial_t \mathbf{u}$, $\mathbf{a}=\partial_{t^2}\mathbf{u}$ are the displacement, velocity and acceleration fields, $\mathbf{M}$, $\mathbf{C}$ and $\mathbf{K}$ are the mass, damping and stiffness forms, and $\mathbf{F}$ is an applied force. $\mathbf{K}$ may be a non-linear function of $\mathbf{u}$. Various kinds of boundary conditions are supported.

The module provides high level classes to solve common problems in a few lines code:
  * Time-domain problems, using the **TimeStepper** class
    * Explicit schemes: *leap frog*
    * Implicit schemes: *Newmark-beta*, *midpoint*, *linear acceleration*, *HHT-alpha*, *generalized-alpha*
  * Eigenmodes problems, using the **ElasticResonanceSolver** class
  * Common boundary conditions, using the **BoundaryCondition** class
    * BCs involving $\mathbf{u}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Free*, *Clamp*, *Dirichlet*, *Neumann*, *Robin*
    * BCs involving $\mathbf{v}$ and $\boldsymbol{\sigma} . \mathbf{n}$: *Dashpot*

GitHub repository:
https://github.com/Universite-Gustave-Eiffel/elastodynamicsx

## Dependencies
ElastodynamiCSx requires FEnicsX / dolfinx v0.4.1 -> see [instructions here](https://github.com/FEniCS/dolfinx#installation)  

### Other required packages
**numpy**  
**scipy**  
**matplotlib**  
**pyvista**  

### Optional packages
**tqdm**

## Installation
Clone the repository:
```bash
git clone https://github.com/Universite-Gustave-Eiffel/elastodynamicsx.git
```
and install the package:
```bash
pip3 install .
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


