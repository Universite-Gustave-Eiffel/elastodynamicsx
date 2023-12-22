# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT


"""
Wave equation (frequency-domain)

Propagation of SH elastic waves in a 2D, homogeneous isotropic solid,
and comparison with an analytical solution
"""

from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import matplotlib.pyplot as plt

from elastodynamicsx.pde import material, BodyForce, boundarycondition, PDE
from elastodynamicsx.solvers import FrequencyDomainSolver
from elastodynamicsx.plot import plotter, live_plotter
from elastodynamicsx.utils import make_facet_tags, make_cell_tags, ParallelEvaluator
from analyticalsolutions import green_2D_SH_rw, int_Fraunhofer_2D

assert np.issubdtype(PETSc.ScalarType, np.complexfloating), \
       "Demo should only be executed with DOLFINx complex mode"


# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
degElement = 1

length, height = 10, 10
Nx, Ny = 100//degElement, 100//degElement
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny], mesh.CellType.triangle)

tag_left, tag_top, tag_right, tag_bottom = 1, 2, 3, 4
all_tags = (tag_left, tag_top, tag_right, tag_bottom)
boundaries = [(tag_left  , lambda x: np.isclose(x[0], 0     )),\
              (tag_right , lambda x: np.isclose(x[0], length)),\
              (tag_bottom, lambda x: np.isclose(x[1], 0     )),\
              (tag_top   , lambda x: np.isclose(x[1], height))]
facet_tags = make_facet_tags(domain, boundaries)
#
V  = fem.FunctionSpace(domain, ("Lagrange", degElement))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho = fem.Constant(domain, PETSc.ScalarType(1))
mu  = fem.Constant(domain, PETSc.ScalarType(1))
mat = material(V, 'scalar', rho, mu)
materials = [mat]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
Z   = mat.Z  # mechanical impedance
bc  = boundarycondition((V, facet_tags, all_tags), 'Dashpot', Z)
bcs = [bc]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                Gaussian Source term
# -----------------------------------------------------
F0 = fem.Constant(domain, PETSc.ScalarType(1))  # amplitude
R0 = 0.1  # radius
X0 = np.array([length/2, height/2, 0])  # center
x  = ufl.SpatialCoordinate(domain)
gaussianBF = F0 * ufl.exp(-((x[0]-X0[0])**2+(x[1]-X0[1])**2)/2/R0**2) / (2*np.pi*R0**2)
bf         = BodyForce(V, gaussianBF)

bodyforces = [bf]
# -----------------------------------------------------


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(V, materials=materials, bodyforces=bodyforces, bcs=bcs)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                  Initialize solver
# -----------------------------------------------------
fdsolver = FrequencyDomainSolver(V.mesh.comm, pde.M(), pde.C(), pde.K(),
                                 pde.init_b(), b_update_function=pde.update_b_frequencydomain)
#
# -----------------------------------------------------


# -----------------------------------------------------
#          Ex 1: Solve for a single frequency
# -----------------------------------------------------
# Solve
omega = 1.0
u_res = fem.Function(V, name='solution')
fdsolver.solve(omega=omega, out=u_res.vector)

# Plot
if domain.comm.rank == 0:
    p = plotter(u_res, complex='real')
    p.show()
#
# ------------------ end of Ex 1 ----------------------


# -----------------------------------------------------
#          Ex 2: Solve for several frequencies
# -----------------------------------------------------
# Frequencies to solve for
omegas = np.linspace(0.5, 3, num=5)

# Declare solution
u_res = fem.Function(V, name='solution')

# Prepare post processing
### -> Extract field at few points
from scipy.spatial.transform import Rotation as R
theta = np.radians(35)
pts = np.linspace(0, length/2, endpoint=False)[1:]
points_out = X0[:,np.newaxis] + R.from_rotvec([0,0,theta]).as_matrix() @ np.array([pts,
                                                                                   np.zeros_like(pts),
                                                                                   np.zeros_like(pts)])

# Declare a convenience ParallelEvaluator
paraEval = ParallelEvaluator(domain, points_out)

# Declare data (local)
u_at_pts_local = np.zeros((paraEval.nb_points_local,
                           1,
                           omegas.size),
                           dtype=PETSc.ScalarType)  # <- output stored here

# Callback function: post process solution
def cbck_storeAtPoints(i, out):
    if paraEval.nb_points_local > 0:
        u_at_pts_local[:,:,i] = u_res.eval(paraEval.points_local, paraEval.cells_local)

# Live plotting
enable_plot = True
if domain.comm.rank == 0 and enable_plot:
    p = live_plotter(u_res, clim=0.25*np.linalg.norm(mu.value*F0.value)*np.array([-1, 1]))
    if paraEval.nb_points_local > 0:
        p.add_points(paraEval.points_local)  # add points to live_plotter
else:
    p = None

# Solve
fdsolver.solve(omega=omegas, out=u_res.vector, callbacks=[cbck_storeAtPoints], live_plotter=p)

# Plot
u_at_pts = paraEval.gather(u_at_pts_local, root=0)

if domain.comm.rank == 0:
    ### -> Exact solution, At few points
    x = points_out.T
    r = np.linalg.norm(x - X0[np.newaxis,:], axis=1)
    
    # account for the size of the source in the analytical formula
    fn_kdomain_finite_size = int_Fraunhofer_2D['gaussian'](R0)
    u_at_pts_anal = green_2D_SH_rw(r, omegas, rho.value, mu.value, fn_kdomain_finite_size)
    
    #
    fn = np.real
    
    fig, ax = plt.subplots(len(omegas),1)
    fig.suptitle(r'u at few points, $\theta$=' + str(int(round(np.degrees(theta), 0))) + r'$^{\circ}$')
    for i in range(len(omegas)):
        ax[i].text(0.15,0.95, r'$\omega$='+str(round(omegas[i],2)),
                   ha='left', va='top', transform=ax[i].transAxes)
        ax[i].plot(r, fn(u_at_pts[:,0,i]), ls='-' , label='FEM')
        ax[i].plot(r, fn(u_at_pts_anal[:,i]), ls='--', label='analytical')
    ax[0].legend()
    ax[-1].set_xlabel('Distance to source')
    plt.show()
#
# ------------------ end of Ex 2 ----------------------

