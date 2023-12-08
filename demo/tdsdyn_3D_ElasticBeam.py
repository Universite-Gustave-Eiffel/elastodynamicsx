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

from elastodynamicsx.pde import material, BodyForce, BoundaryCondition, PDE, damping
from elastodynamicsx.solvers import TimeStepper
from elastodynamicsx.utils import make_facet_tags, ParallelEvaluator

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
L_, B_, H_ = 1, 0.04, 0.1

Nx, Ny, Nz = 60, 5, 10

extent = [[0., 0., 0.], [L_, B_, H_]]
domain = mesh.create_box(MPI.COMM_WORLD, extent, [Nx, Ny, Nz])

tag_left, tag_top, tag_right, tag_bottom, tag_back, tag_front = 1, 2, 3, 4, 5, 6
boundaries = [(tag_left  , lambda x: np.isclose(x[0], 0 )),\
              (tag_right , lambda x: np.isclose(x[0], L_)),\
              (tag_bottom, lambda x: np.isclose(x[1], 0 )),\
              (tag_top   , lambda x: np.isclose(x[1], B_)),\
              (tag_back  , lambda x: np.isclose(x[2], 0 )),\
              (tag_front , lambda x: np.isclose(x[2], H_))]
facet_tags = make_facet_tags(domain, boundaries)
#
V = fem.FunctionSpace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
T_N  = fem.Constant(domain, np.array([0]*3, dtype=PETSc.ScalarType))  # normal traction (Neumann boundary condition)
bc_l = BoundaryCondition((V, facet_tags, tag_left ), 'Clamp')
bc_r = BoundaryCondition((V, facet_tags, tag_right), 'Neumann', T_N)
bcs  = [bc_l, bc_r]
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

material = material(V, 'isotropic', rho, lambda_, mu, damping=damping('Rayleigh', eta_m, eta_k))
#
# -----------------------------------------------------


# -----------------------------------------------------
#               (boundary) Source term
# -----------------------------------------------------
### -> Time function
#
p0  = 1.  # max amplitude
F_0 = p0 * np.array([0,0,1], dtype=PETSc.ScalarType)  # source orientation
cutoff_Tc = 4/5  # release time
#
src_t        = lambda t: t/cutoff_Tc * (t>0) * (t<=cutoff_Tc)
T_N_function = lambda t: src_t(t) * F_0
#
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
T       = 4  # difference with original example: here t=[0,T-dt]
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
pde = PDE(V, materials=[material], bodyforces=[], bcs=bcs)

# Time integration
tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, **kwargsTScheme)
tStepper.set_initial_condition(u0=[0,0,0], v0=[0,0,0], t0=0)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    define outputs
# -----------------------------------------------------
### -> Extract signals at few points
# Define points
points_out = np.array([[L_, B_/2, 0]]).T

# Declare a convenience ParallelEvaluator
paraEval = ParallelEvaluator(domain, points_out)

# Declare data (local)
signals_local = np.zeros((paraEval.nb_points_local,
                          V.num_sub_spaces,
                          Nsteps))  # <- output stored here

### -> Energies
energies = np.zeros((Nsteps, 4))
E_damp   = 0

u_n = tStepper.timescheme.u  # The displacement at time t_n
v_n = tStepper.timescheme.v
comm = domain.comm
Energy_elastic = lambda *a: comm.allreduce( fem.assemble_scalar(fem.form( 1/2* pde.k(u_n, u_n) )) , op=MPI.SUM)
Energy_kinetic = lambda *a: comm.allreduce( fem.assemble_scalar(fem.form( 1/2* pde.m(v_n, v_n) )) , op=MPI.SUM)
Energy_damping = lambda *a: dt*comm.allreduce( fem.assemble_scalar(fem.form( pde.c(v_n, v_n) )) , op=MPI.SUM)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### define callfirsts and callbacks
def cfst_updateSources(t):
    T_N.value = T_N_function(t)

def cbck_storeAtPoints(i, out):
    if paraEval.nb_points_local > 0:
        signals_local[:,:,i+1] = u_n.eval(paraEval.points_local, paraEval.cells_local)

def cbck_energies(i, out):
    global E_damp
    E_elas = Energy_elastic()
    E_kin  = Energy_kinetic()
    E_damp+= Energy_damping()
    E_tot  = E_elas + E_kin + E_damp
    energies[i+1,:] = np.array([E_elas, E_kin, E_damp, E_tot])

### live plotting params
clim = 0.4 * L_*B_*H_/(E*B_*H_**3/12) * np.amax(F_0)*np.array([0, 1])
live_plotter = {'refresh_step':1, 'clim':clim} if domain.comm.rank == 0 else None

### Run the big time loop!
tStepper.solve(Nsteps-1,
               callfirsts=[cfst_updateSources],
               callbacks=[cbck_storeAtPoints, cbck_energies],
               live_plotter=live_plotter)
### End of big calc.
#
# -----------------------------------------------------

# Plot tip displacement and energies evolution
all_signals = paraEval.gather(signals_local, root=0)

if domain.comm.rank == 0:
    t = dt*np.arange(energies.shape[0])

    # Tip displacement
    u_tip = all_signals[0]
    plt.figure()
    plt.plot(t, u_tip[2,:])
    plt.xlabel('Time')
    plt.ylabel('Tip displacement')
    plt.ylim(-0.5, 0.5)

    # Energies
    plt.figure()
    plt.plot(t, energies, marker='o', ms=5)
    plt.legend(("elastic", "kinetic", "damping", "total"))
    plt.xlabel("Time")
    plt.ylabel("Energies")
    plt.ylim(0, 0.0011)
    plt.show()
