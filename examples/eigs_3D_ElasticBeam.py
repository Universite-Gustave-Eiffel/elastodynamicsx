"""
Eigenmodes

Free resonances of a beam clamped at one end, compared against beam theory

adapted from (legacy Fenics): https://comet-fenics.readthedocs.io/en/latest/demo/modal_analysis_dynamics/cantilever_modal.html
also inspired from (Fenicsx): https://mikics.github.io/gsoc-jupyterbook/chapter3/demo_half_loaded_waveguide.html
"""

from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from elastodynamicsx.pde import BoundaryCondition, Material
from elastodynamicsx.solvers import ElasticResonanceSolver
from elastodynamicsx.utils import make_facet_tags

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
L_, B_, H_ = 20., 0.5, 1.

Nx = 20
Ny = int(B_/L_*Nx)+1
Nz = int(H_/L_*Nx)+1

extent = [[0., 0., 0.], [L_, B_, H_]]
domain = mesh.create_box(MPI.COMM_WORLD, extent, [Nx, Ny, Nz])
boundaries = [(1, lambda x: np.isclose(x[0], 0 )),\
              (2, lambda x: np.isclose(x[0], L_)),\
              (3, lambda x: np.isclose(x[1], 0 )),\
              (4, lambda x: np.isclose(x[1], B_)),\
              (5, lambda x: np.isclose(x[2], 0 )),\
              (6, lambda x: np.isclose(x[2], H_))]
facet_tags = make_facet_tags(domain, boundaries)
#
V = fem.VectorFunctionSpace(domain, ("CG", 2))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary condition
# -----------------------------------------------------
bc_l = BoundaryCondition((V, facet_tags, 1), 'Clamp')
bcs  = [bc_l]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
scaleRHO = 1e6
scaleFREQ= np.sqrt(scaleRHO)
E, nu = 1e5, 0.
rho = 1e-3*scaleRHO
lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)
mu      = E / 2 / (1 + nu)
rho     = fem.Constant(domain, PETSc.ScalarType(rho))
lambda_ = fem.Constant(domain, PETSc.ScalarType(lambda_))
mu      = fem.Constant(domain, PETSc.ScalarType(mu))

material = Material.build(V, 'isotropic', rho, lambda_, mu)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### Initialize the solver
eps = ElasticResonanceSolver(material.m, material.k, V, bcs=bcs, nev=6)

### Run the big calculation!
eps.solve()
### End of big calc.

### Get the result
#eps.printEigenvalues()
eigenfreqs = eps.getEigenfrequencies()
#eigenmodes = eps.getEigenmodes()
eps.plot(wireframe=True, factor=50)

verbose = False
if verbose:
    eps.view()
    eps.errorView()
    PETSc.Sys.Print("Number of converged eigenpairs %d" % nconv)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                Compare with beam theory
# -----------------------------------------------------
# Exact solution computation
from scipy.optimize import root
from math import cos, cosh
falpha = lambda x: cos(x)*cosh(x)+1
alpha  = lambda n: root(falpha, (2*n+1)*np.pi/2)['x'][0]

nev = eigenfreqs.size
I_bend = H_*B_**3/12*(np.arange(nev)%2==0) + B_*H_**3/12*(np.arange(nev)%2==1)
freq_beam = np.array([alpha(i//2) for i in range(nev)])**2 *np.sqrt(E*I_bend/(rho.value*B_*H_*L_**4))/2/np.pi

print('Eigenfrequencies: comparison with beam theory')
print('FE:         ', eigenfreqs*scaleFREQ)
print('Beam theory:', freq_beam*scaleFREQ)
#
# -----------------------------------------------------


