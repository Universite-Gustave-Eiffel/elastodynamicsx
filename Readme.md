[![Generic badge](https://github.com/Universite-Gustave-Eiffel/elastodynamicsx/actions/workflows/pages/pages-build-deployment/badge.svg)](https://universite-gustave-eiffel.github.io/elastodynamicsx/)	

# ElastodynamiCSx 
ElastodynamiCSx is dedicated to the numerical modeling of wave propagation in solids using the [FEniCSx](https://fenicsproject.org/) Finite Elements library. It deals with the following PDE:

$$\mathbf{M}\mathbf{a} + \mathbf{C}\mathbf{v} + \mathbf{K}(\mathbf{u}) = \mathbf{b},$$

where $\mathbf{u}$, $\mathbf{v}=\partial_t \mathbf{u}$, $\mathbf{a}=\partial_t^2\mathbf{u}$ are the displacement, velocity and acceleration fields, $\mathbf{M}$, $\mathbf{C}$ and $\mathbf{K}$ are the mass, damping and stiffness forms, and $\mathbf{b}$ is an applied force. For time domain problems $\mathbf{K}$ may be a non-linear function of $\mathbf{u}$.

The library provides high level classes to build and solve common problems in a few lines code.

## Documentation
Documentation can be viewed at https://universite-gustave-eiffel.github.io/elastodynamicsx/. A variety of examples can be found in the [demo/](https://github.com/Universite-Gustave-Eiffel/elastodynamicsx/tree/main/demo) subfolder.
