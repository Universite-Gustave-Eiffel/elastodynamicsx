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

from elastodynamicsx.timestepper import TimeStepper
from elastodynamicsx.utils import find_points_and_cells_on_proc

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
L_, B_, H_ = 1, 0.04, 0.1

Nx, Ny, Nz = 60, 5, 10

extent = [[0., 0., 0.], [L_, B_, H_]]
domain = mesh.create_box(MPI.COMM_WORLD, extent, [Nx, Ny, Nz])
#
V = fem.VectorFunctionSpace(domain, ("CG", 1))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
def left_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1

u_D_clamp = np.array([0,0,0], dtype=PETSc.ScalarType)
bc_l = fem.dirichletbc(u_D_clamp, fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(domain, fdim, left_boundary) ), V)
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
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    Source term
# -----------------------------------------------------
### -> Time function
#
p0  = 1. #max amplitude
F_0 = p0 * np.array([0,0,1], dtype=PETSc.ScalarType) #source orientation
cutoff_Tc = 4/5 #release time
#
src_t = lambda t: t/cutoff_Tc * (t>0) * (t<=cutoff_Tc)
def T_N_function(t): return lambda x: src_t(t) * F_0[:,np.newaxis] * np.isclose(x[0], 1)[np.newaxis,:]
#
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
T       = 4
Nsteps  = 50
dt = T/Nsteps
#
# -----------------------------------------------------


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
### Body force 'F_body' and normal traction 'T_N'
F_body = fem.Constant(domain, PETSc.ScalarType((0,0,0))) #body force
T_N    = fem.Function(V) #normal traction (Neumann boundary condition)

def epsilon(u): return ufl.sym(ufl.grad(u))
def sigma(u): return lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

m_ = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
k_ = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
c_ = lambda u,v: eta_m*m_(u,v) + eta_k*k_(u,v)
L  = lambda v  : ufl.dot(F_body, v) * ufl.dx   +   ufl.dot(T_N, v) * ufl.ds
###

###
# Initial conditions
u0, v0 = fem.Function(V), fem.Function(V)
u0.interpolate(lambda x: np.zeros((domain.topology.dim, x.shape[1]), dtype=PETSc.ScalarType))
v0.interpolate(lambda x: np.zeros((domain.topology.dim, x.shape[1]), dtype=PETSc.ScalarType))
#
T_N.interpolate(T_N_function(0))
###

# Generalized-alpha method parameters
alpha_m = 0.2
alpha_f = 0.4
#gamma   = fem.Constant(domain, PETSc.ScalarType(0.5+alpha_f-alpha_m))
#beta    = fem.Constant(domain, PETSc.ScalarType((gamma+0.5)**2/4.))

#  Variational problem
tStepper = TimeStepper.build(m_, c_, k_, L, dt, V, bcs=[bc_l], scheme='g-a-newmark', alpha_m=alpha_m, alpha_f=alpha_f)
tStepper.initial_condition(u0, v0, t0=0)
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
    T_N.interpolate(T_N_function(t))

def cbck_storeAtPoints(i, tStepper):
    if len(points_output_on_proc)>0: signals_at_points[:,:,i+1] = tStepper.u.eval(points_output_on_proc, cells_output_on_proc)

def cbck_energies(i, tStepper):
    global E_damp
    u_n = tStepper.u
    v_n = tStepper.v
    E_elas = domain.comm.allreduce( fem.assemble_scalar(fem.form( 1/2* k_(u_n, u_n) )) , op=MPI.SUM)
    E_kin  = domain.comm.allreduce( fem.assemble_scalar(fem.form( 1/2* m_(v_n, v_n) )) , op=MPI.SUM)
    E_damp+= dt*domain.comm.allreduce( fem.assemble_scalar(fem.form( c_(v_n, v_n) )) , op=MPI.SUM)
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
plt.plot(dt*np.arange(energies.shape[0]), energies)
plt.legend(("elastic", "kinetic", "damping", "total"))
plt.xlabel("Time")
plt.ylabel("Energies")
plt.show()

