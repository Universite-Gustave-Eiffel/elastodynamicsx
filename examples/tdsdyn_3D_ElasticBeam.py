"""
Time-domain structural dynamics

Vibration of a beam clamped at one end. The beam is increasingly loaded at its other end, and then suddenly released.

adapted from (legacy Fenics): https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
"""

from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import matplotlib.pyplot as plt

from elastodynamicsx.pde import material, BodyForce, BoundaryCondition, PDE, Damping
from elastodynamicsx.solvers import TimeStepper
from elastodynamicsx.utils import find_points_and_cells_on_proc, make_facet_tags

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
L_, B_, H_ = 1, 0.04, 0.1

Nx, Ny, Nz = 60, 5, 10

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
V = fem.VectorFunctionSpace(domain, ("CG", 1))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
T_N  = fem.Constant(domain, np.array([0]*3, dtype=PETSc.ScalarType)) #normal traction (Neumann boundary condition)
bc_l    = BoundaryCondition((V, facet_tags, 1), 'Clamp')
bc_r    = BoundaryCondition((V, facet_tags, 2), 'Neumann', T_N)
bc_free = BoundaryCondition((V, facet_tags, (3,4,5,6)), 'Free')
bcs = [bc_l, bc_r, bc_free]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
E, nu = 1000, 0.3
rho = 1
lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)
mu      = E / 2 / (1 + nu)
rho     = fem.Constant(domain, PETSc.ScalarType(rho))
lambda_ = fem.Constant(domain, PETSc.ScalarType(lambda_))
mu      = fem.Constant(domain, PETSc.ScalarType(mu))

# Rayleigh damping coefficients
eta_m = fem.Constant(domain, PETSc.ScalarType(0.01))
eta_k = fem.Constant(domain, PETSc.ScalarType(0.01))

material = material(V, 'isotropic', rho, lambda_, mu, damping=Damping.build('Rayleigh', eta_m, eta_k))
#
# -----------------------------------------------------


# -----------------------------------------------------
#               (boundary) Source term
# -----------------------------------------------------
### -> Time function
#
p0  = 1. #max amplitude
F_0 = p0 * np.array([0,0,1], dtype=PETSc.ScalarType) #source orientation
cutoff_Tc = 4/5 #release time
#
src_t        = lambda t: t/cutoff_Tc * (t>0) * (t<=cutoff_Tc)
T_N_function = lambda t: src_t(t) * F_0
#
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
T       = 4
Nsteps  = 50
dt = T/Nsteps

# Generalized-alpha method parameters
alpha_m = 0.2
alpha_f = 0.4
kwargsTScheme = dict(scheme='g-a-newmark', alpha_m=alpha_m, alpha_f=alpha_f)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(materials=[material], bodyforces=[])

#  Time integration
tStepper = TimeStepper.build(pde.m, pde.c, pde.k, pde.L, dt, V, bcs=bcs, **kwargsTScheme)
tStepper.initial_condition(u0=[0,0,0], v0=[0,0,0], t0=0)
#tStepper.solver.view()
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    define outputs
# -----------------------------------------------------
### -> Extract signals at few points
points_output = np.array([[L_, B_/2, 0]]).T
points_output_on_proc, cells_output_on_proc = find_points_and_cells_on_proc(points_output, domain)
signals_at_points = np.zeros((points_output.shape[1], domain.topology.dim, Nsteps)) #<- output stored here

### -> Energies
energies = np.zeros((Nsteps, 4))
E_damp   = 0
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### define callfirsts and callbacks
def cfst_updateSources(t, tStepper):
    T_N.value = T_N_function(t)

def cbck_storeAtPoints(i, tStepper):
    if len(points_output_on_proc)>0: signals_at_points[:,:,i+1] = tStepper.u.eval(points_output_on_proc, cells_output_on_proc)

def cbck_energies(i, tStepper):
    global E_damp
    E_elas = tStepper.Energy_elastic()
    E_kin  = tStepper.Energy_kinetic()
    E_damp+= tStepper.Energy_damping()
    E_tot  = E_elas + E_kin + E_damp
    energies[i+1,:] = np.array([E_elas, E_kin, E_damp, E_tot])

### live plotting params
clim = 0.4 * L_*B_*H_/(E*B_*H_**3/12) * np.amax(F_0)*np.array([0, 1])

### Run the big time loop!
tStepper.run(Nsteps-1, callfirsts=[cfst_updateSources], callbacks=[cbck_storeAtPoints, cbck_energies], live_plotter={'live_plotter_step':1, 'clim':clim})
### End of big calc.
#
# -----------------------------------------------------

# Plot energies evolution
plt.figure()
plt.plot(dt*np.arange(energies.shape[0]), energies, marker='o', ms=5)
plt.legend(("elastic", "kinetic", "damping", "total"))
plt.xlabel("Time")
plt.ylabel("Energies")
plt.show()

