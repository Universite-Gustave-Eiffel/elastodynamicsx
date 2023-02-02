"""
Wave equation (frequency-domain)

Propagation of P and SV elastic waves in a 2D, homogeneous isotropic solid, and comparison with an analytical solution
"""

from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import matplotlib.pyplot as plt

from elastodynamicsx.pde import material, BodyForce, BoundaryCondition, PDE
from elastodynamicsx.solvers import FrequencyDomainSolver
from elastodynamicsx.plot import CustomVectorPlotter
from elastodynamicsx.utils import find_points_and_cells_on_proc, make_facet_tags, make_cell_tags
from analyticalsolutions import u_2D_PSV_xw, int_Fraunhofer_2D

assert np.issubdtype(PETSc.ScalarType, np.complexfloating), "Demo should only be executed with DOLFINx complex mode"
    
# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
degElement = 1

length, height = 10, 10
Nx, Ny = 100//degElement, 100//degElement
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny], mesh.CellType.triangle)
boundaries = [(1, lambda x: np.isclose(x[0], 0     )),\
              (2, lambda x: np.isclose(x[0], length)),\
              (3, lambda x: np.isclose(x[1], 0     )),\
              (4, lambda x: np.isclose(x[1], height))]
facet_tags = make_facet_tags(domain, boundaries)
#
V = fem.VectorFunctionSpace(domain, ("CG", degElement))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho     = fem.Constant(domain, PETSc.ScalarType(1))
mu      = fem.Constant(domain, PETSc.ScalarType(1))
lambda_ = fem.Constant(domain, PETSc.ScalarType(2))

mat   = material(V, 'isotropic', rho, lambda_, mu)
materials = [mat]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
Z_N, Z_T = mat.Z_N, mat.Z_T #P and S mechanical impedances
bc  = BoundaryCondition((V, facet_tags, (1,2,3,4)), 'Dashpot', (Z_N, Z_T))
bcs = [bc]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                Gaussian Source term
# -----------------------------------------------------
F0 = fem.Constant(domain, PETSc.ScalarType([1,0])) #amplitude
R0 = 0.1 #radius
X0 = np.array([length/2, height/2, 0]) #center
x  = ufl.SpatialCoordinate(domain)
gaussianBF = F0 * ufl.exp(-((x[0]-X0[0])**2+(x[1]-X0[1])**2)/2/R0**2) / (2*np.pi*R0**2)
bf         = BodyForce(V, gaussianBF)

bodyforces = [bf]
# -----------------------------------------------------


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(materials=materials, bodyforces=bodyforces)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                  Initialize solver
# -----------------------------------------------------
fdsolver = FrequencyDomainSolver(V, pde.m, pde.c, pde.k, pde.L, bcs=bcs)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
omega = 1.0
u = fdsolver.solve(omega=omega)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    Post process
# -----------------------------------------------------
### -> Extract field at few points
pts = np.linspace(0, length/2, endpoint=False)[1:]
points_output = X0[:,np.newaxis] + np.array([pts, np.zeros_like(pts), np.zeros_like(pts)])
points_output_on_proc, cells_output_on_proc = find_points_and_cells_on_proc(points_output, domain)

u_at_pts = u.eval(points_output_on_proc, cells_output_on_proc)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                        Plot
# -----------------------------------------------------
p = CustomVectorPlotter(u, complex='real')
p.show()

if len(points_output_on_proc)>0:
    ### -> Exact solution, At few points
    x = points_output_on_proc
    fn_kdomain_finite_size = int_Fraunhofer_2D['gaussian'](R0) #accounts for the size of the source in the analytical formula
    u_at_pts_anal = u_2D_PSV_xw(x-X0[np.newaxis,:], omega, F0.value, rho.value, lambda_.value, mu.value, fn_kdomain_finite_size)
    
    #
    fn = np.real
    
    fig, ax = plt.subplots(1,1)
    ax.set_title('u at few points')
    ax.plot(x[:,0]-X0[0], fn(u_at_pts[:,0]), ls='-' , label='FEM')
    ax.plot(x[:,0]-X0[1], fn(u_at_pts_anal[:,0,0]), ls='--', label='analytical')
    ax.set_xlabel('Distance to source')
    ax.legend()
    plt.show()
#
# -----------------------------------------------------

