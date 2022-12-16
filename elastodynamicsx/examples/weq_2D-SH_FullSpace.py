"""
Wave equation (time-domain)

Propagation of SH elastic waves in a 2D, homogeneous isotropic solid, and comparison with an analytical solution
"""

import time
from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import pyvista
import matplotlib.pyplot as plt

from elastodynamicsx.timestepper import TimeStepper
from elastodynamicsx.plotting import CustomScalarPlotter
from elastodynamicsx.utils import find_points_and_cells_on_proc
from elastodynamicsx.analyticalsolutions import u_2D_SH_rt, int_Fraunhofer_2D

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
length, height = 10, 10
Nx, Ny = 100, 100
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny])
#
V = fem.FunctionSpace(domain, ("CG", 2))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho     = fem.Constant(domain, PETSc.ScalarType(1))
mu      = fem.Constant(domain, PETSc.ScalarType(1))
#lambda_ = fem.Constant(domain, PETSc.ScalarType(2))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    Source term
# -----------------------------------------------------
### -> Space function
#
X0_src = np.array([length/2,height/2,0]) #center
R0_src = 0.1 #radius
#
### Gaussian function
nrm   = 1/(2*np.pi*R0_src**2) #normalize to int[src_x(x) dx]=1
src_x = lambda x: nrm * np.exp(-1/2*(np.linalg.norm(x-X0_src[:,np.newaxis], axis=0)/R0_src)**2, dtype=PETSc.ScalarType) #source(x)
fn_kdomain_finite_size = int_Fraunhofer_2D['gaussian'](R0_src) #accounts for the size of the source in the analytical formula
#
### -> Time function
#
f0 = 1 #central frequency of the source
T0 = 1/f0 #period
d0 = 2*T0 #duration of source
#
src_t = lambda t: np.sin(2*np.pi*f0 * t) * np.sin(np.pi*t/d0)**2 * (t<d0) * (t>0) #source(t)

### -> Space-Time function
#
F_0 = 1 #amplitude of the source
#
def F_body_function(t): return lambda x: F_0 * src_t(t) * src_x(x) #source(x) at a given time
#
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
tstart = 0 # Start time
tmax   = 4*d0 # Final time
num_steps = 500
dt = (tmax-tstart) / num_steps # time step size
#
# -----------------------------------------------------


###
# Some control numbers...
hx = length/Nx
c_SH = np.sqrt(mu.value/rho.value) #phase velocity
lbda0 = c_SH/f0
C_CFL = dt/hx  * c_SH # Courant number (Courant-Friedrichs-Lewy condition)
print('Number of points per wavelength at central frequency: ', round(lbda0/hx, 2))
print('Number of time steps per period at central frequency: ', round(T0/dt, 2))
print('CFL condition: Courant number = ', round(C_CFL, 2))
###


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
### Body force 'F_body' and normal traction 'T_N'
F_body = fem.Function(V) #body force
T_N    = fem.Constant(domain, PETSc.ScalarType(0)) #normal traction (Neumann boundary condition)

def epsilon(u): return ufl.nabla_grad(u)
def sigma(u): return mu*epsilon(u)

a_tt = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
a_xx = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L    = lambda v  : ufl.dot(F_body, v) * ufl.dx   +   ufl.dot(T_N, v) * ufl.ds
###

###
# Initial conditions
u0, v0 = fem.Function(V), fem.Function(V)
u0.interpolate(lambda x: np.zeros(x.shape[1], dtype=PETSc.ScalarType))
v0.interpolate(lambda x: np.zeros(x.shape[1], dtype=PETSc.ScalarType))
#
F_body.interpolate(F_body_function(tstart))
###

#  Variational problem
tStepper = TimeStepper.build(a_tt, a_xx, L, dt, V, [], scheme='leapfrog')
tStepper.initial_condition(u0, v0, t0=tstart)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    define outputs
# -----------------------------------------------------
### -> Store all time steps ? -> YES if debug & learning // NO if big calc.
storeAllSteps = False
all_u = [fem.Function(V) for i in range(num_steps)] if storeAllSteps else None #all steps are stored here
#
### -> Extract signals at few points
points_output = X0_src[:,np.newaxis] + np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]]).T
points_output_on_proc, cells_output_on_proc = find_points_and_cells_on_proc(points_output, domain)
signals_at_points = np.zeros((len(points_output_on_proc), 1, num_steps)) #<- output stored here
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### define callfirsts and callbacks
def cfst_updateSources(i, tStepper):
    F_body.interpolate(F_body_function(tStepper.t_n))

def cbck_storeFullField(i, tStepper):
    if storeAllSteps: all_u[i].x.array[:] = tStepper.u_n.x.array
def cbck_storeAtPoints(i, tStepper):
    if len(points_output_on_proc)>0: signals_at_points[:,:,i] = tStepper.u_n.eval(points_output_on_proc, cells_output_on_proc)

### enable live plotting
clim = 0.1*F_0*np.array([-1, 1])
tStepper.set_live_plotter(live_plotter_step=10, **{'clim':clim}) #0 to disable
if len(points_output_on_proc)>0:
    tStepper.live_plotter.add_points(points_output_on_proc, render_points_as_spheres=True, point_size=12) #adds points to live_plotter

### Run the big time loop!
tStepper.run(num_steps, callfirsts=[cfst_updateSources], callbacks=[cbck_storeFullField, cbck_storeAtPoints])
### End of big calc.
#
# -----------------------------------------------------


# -----------------------------------------------------
#     Interactive view of all time steps if stored
# -----------------------------------------------------
if storeAllSteps: #plotter with a slider to browse through all time steps
    ### -> Exact solution, Full field
    u_n = tStepper.u_n
    x = u_n.function_space.tabulate_dof_coordinates()
    r = np.linalg.norm(x - X0_src[np.newaxis,:], axis=1)
    all_u_n_exact = u_2D_SH_rt(r, np.roll(src_t(dt*np.arange(num_steps)), -2), rho.value, mu.value, dt, fn_kdomain_finite_size)
    #
    plotter = CustomScalarPlotter(u_n, u_n, u_n, labels=('FE', 'Exact', 'Diff.'), clim=clim)
    def updateTStep(value):
        i = int((value-tstart)/dt)
        plotter.update_scalars(all_u[i].x.array, all_u_n_exact[:,i], all_u[i].x.array-all_u_n_exact[:,i])
    plotter.add_slider_widget(updateTStep, [tstart, tmax-dt])
    plotter.show()
#
# -----------------------------------------------------


# -----------------------------------------------------
#              Plot signals at few points
# -----------------------------------------------------
if len(points_output_on_proc)>0:
    ### -> Exact solution, At few points
    x = points_output_on_proc
    r = np.linalg.norm(x - X0_src[np.newaxis,:], axis=1)
    signals_at_points_exact = u_2D_SH_rt(r, np.roll(src_t(dt*np.arange(num_steps)), -2), rho.value, mu.value, dt, fn_kdomain_finite_size)
    #
    fig, ax = plt.subplots(1,1)
    t = dt*np.arange(num_steps)
    ax.set_title('Signals at few points')
    for i in range(len(signals_at_points)):
        ax.plot(t, signals_at_points[i,0,:],     c='C'+str(i), ls='-') #FEM
        ax.plot(t, signals_at_points_exact[i,:], c='C'+str(i), ls='--') #exact
    ax.set_xlabel('Time')
    plt.show()
#
# -----------------------------------------------------

