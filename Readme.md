# ElastodynamiCSx 
ElastodynamiCSx is dedicated to the numerical modeling of wave propagation in solids using the [FEniCSx](https://fenicsproject.org/) Finite Elements library.    

GitHub repository:
https://github.com/Universite-Gustave-Eiffel/elastodynamicsx

## Dependencies
ElastodynamiCSx requires FEnicsX / dolfinx v0.4.1 -> see [instructions here](https://github.com/FEniCS/dolfinx#installation)  

### Other required packages
**numpy**  
**scipy**  
**matplotlib**  
**pyvista**  

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
    * 2D, homogeneous space, anti-plane line load (SH waves): weq_2D-SH_FullSpace.py
    * 2D, homogeneous space, in-plane line load (P-SV waves): weq_2D-PSV_FullSpace.py

Reference for the analytical solutions:
  * Kausel, E. (2006). Fundamental solutions in elastodynamics: a compendium. Cambridge University Press.

## Useful links
Part of the code is largely inspired from:
  * [The FEniCSx tutorial](https://jorgensd.github.io/dolfinx-tutorial/)

Other useful references:
  * Fenicsx:
    * https://mikics.github.io/
    * https://newfrac.gitlab.io/newfrac-fenicsx-training/index.html
    * https://github.com/multiphenics/multiphenicsx
  * legacy Fenics:
    * https://www2.karlin.mff.cuni.cz/~hron/fenics-tutorial/index.html
    * https://fenics-solid-tutorial.readthedocs.io/en/latest/

## Authors and contributors
Author: Pierric Mora    

