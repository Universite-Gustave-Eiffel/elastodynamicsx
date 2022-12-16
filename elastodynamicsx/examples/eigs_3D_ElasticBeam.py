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

from elastodynamicsx.eigensolver import ElasticResonanceSolver

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
L, B, H = 20., 0.5, 1.

Nx = 20
Ny = int(B/L*Nx)+1
Nz = int(H/L*Nx)+1

extent = [[0., 0., 0.], [L, B, H]]
domain = mesh.create_box(MPI.COMM_WORLD, extent, [Nx, Ny, Nz])
#
V = fem.VectorFunctionSpace(domain, ("CG", 3))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary condition
# -----------------------------------------------------
def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0,0,0], dtype=PETSc.ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
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
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### Initialize the solver
eps = ElasticResonanceSolver.build_isotropicMaterial(rho, lambda_, mu, V, bcs=[bc], nev=6)

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
alpha = lambda n: root(falpha, (2*n+1)*np.pi/2)['x'][0]

nev = eigenfreqs.size
I_bend = H*B**3/12*(np.arange(nev)%2==0) + B*H**3/12*(np.arange(nev)%2==1)
freq_beam = np.array([alpha(i//2) for i in range(nev)])**2 *np.sqrt(E*I_bend/(rho.value*B*H*L**4))/2/np.pi

print('Eigenfrequencies: comparison with beam theory')
print('FE:         ', eigenfreqs*scaleFREQ)
print('Beam theory:', freq_beam*scaleFREQ)
#
# -----------------------------------------------------


