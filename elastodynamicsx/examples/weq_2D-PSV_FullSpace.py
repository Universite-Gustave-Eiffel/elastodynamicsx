#documentation: TODO

#TODO: source en Hann plutot que rect
#TODO: tenir compte de la taille de la source dans formule anal

import time
from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import pyvista
import matplotlib.pyplot as plt

from elastodynamicsx.timestepper import TimeStepper
from elastodynamicsx.plotting import CustomVectorPlotter
from elastodynamicsx.utils import find_points_and_cells_on_proc
from elastodynamicsx.analyticalsolutions import u_2D_PSV_rt

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
length, height = 10, 10
Nx, Ny = 100, 100
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny])
#
V_scalar = fem.FunctionSpace(domain, ("CG", 2))
V = fem.VectorFunctionSpace(domain, ("CG", 2))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho     = fem.Constant(domain, PETSc.ScalarType(1))
mu      = fem.Constant(domain, PETSc.ScalarType(1))
lambda_ = fem.Constant(domain, PETSc.ScalarType(2))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    Source term
# -----------------------------------------------------
### -> Space function
#
X0_src = np.array([length/2,height/2,0]) #center
R0_src = 0.1 #radius
nrm   = 1/(np.pi*R0_src**2) #normalize to int[src_x(x) dx]=1
#
src_x = lambda x: nrm * np.array(np.linalg.norm(x-X0_src[:,np.newaxis], axis=0)<=R0_src, dtype=PETSc.ScalarType) #source(x)
if True: #check and correct normalization
    F_ = fem.Function(V_scalar) #body force
    F_.interpolate(src_x)
    nrmFE = domain.comm.allreduce( fem.assemble_scalar(fem.form(F_ * ufl.dx)) , op=MPI.SUM)
    nrm = nrm/nrmFE
    print('norm of FE source term, before correction', nrmFE )

### -> Time function
#
f0 = 1 #central frequency of the source
T0 = 1/f0 #period
d0 = 2*T0 #duration of source
#
src_t = lambda t: np.sin(2*np.pi*f0 * t) * np.sin(np.pi*t/d0)**2 * (t<d0) * (t>0) #source(t)

### -> Space-Time function
#
U0 = np.array([1,0], dtype=PETSc.ScalarType) #amplitude of the source
#
def F_body_function(t): return lambda x: U0[:,np.newaxis] * src_t(t) * src_x(x)[np.newaxis,:] #source(x) at a given time
#
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
tstart = 0 # Start time
tmax   = 4*d0 # Final time
num_steps = 1000
dt = (tmax-tstart) / num_steps # time step size
#
# -----------------------------------------------------


###
# Some control numbers...
hx = length/Nx
c_P = np.sqrt((lambda_.value+2*mu.value)/rho.value) #P-wave velocity
c_S = np.sqrt(mu.value/rho.value) #S-wave velocity
lbda0 = c_S/f0
C_CFL = dt/hx  * c_S # Courant number (Courant-Friedrichs-Lewy condition)
print('Number of points per wavelength at central frequency: ', round(lbda0/hx, 2))
print('Number of time steps per period at central frequency: ', round(T0/dt, 2))
print('CFL condition: Courant number = ', round(C_CFL, 2))
###


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
### Body force 'F_body' and normal traction 'T_N'
F_body = fem.Function(V) #body force
T_N    = fem.Constant(domain, PETSc.ScalarType((0,0))) #normal traction (Neumann boundary condition)

def epsilon(u): return ufl.sym(ufl.grad(u))
def sigma(u): return lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

a_tt = lambda u,v: rho* ufl.dot(u, v) * ufl.dx
a_xx = lambda u,v: ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L    = lambda v  : ufl.dot(F_body, v) * ufl.dx   +   ufl.dot(T_N, v) * ufl.ds
###

###
# Initial conditions
def initial_condition_u(x):  return np.zeros((domain.topology.dim, x.shape[1]), dtype=PETSc.ScalarType)
def initial_condition_du(x): return np.zeros((domain.topology.dim, x.shape[1]), dtype=PETSc.ScalarType)
F_body.interpolate(F_body_function(tstart))
###

#  Variational problem
tStepper = TimeStepper.build(dt, V, a_tt, a_xx, L, scheme='leapfrog')
tStepper.initial_condition(initial_condition_u, initial_condition_du, t0=tstart)
u_n = tStepper.u_n
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
points_output = X0_src[:,np.newaxis] + np.array([[1, 0, 0], [2, 0, 0]]).T
points_output_on_proc, cells_output_on_proc = find_points_and_cells_on_proc(points_output, domain)
signals_at_points = np.zeros((points_output.shape[1], domain.topology.dim, num_steps)) #<- output stored here
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    Exact solution
# -----------------------------------------------------
### -> Full field
x = u_n.function_space.tabulate_dof_coordinates()

#r = np.maximum(1e-8, np.linalg.norm(x - X0_src[np.newaxis,:], axis=1)) #cheat to avoid NaN at r=0
all_u_n_exact = u_2D_PSV_rt(x - X0_src[np.newaxis,:], np.roll(src_t(dt*np.arange(num_steps)), -2), U0, rho.value,lambda_.value, mu.value, dt) if storeAllSteps else np.zeros((x.shape[0],2,num_steps))

### -> At few points
if len(points_output_on_proc)>0:
    x = points_output_on_proc
    #r = np.maximum(1e-8, np.linalg.norm(x - X0_src[np.newaxis,:], axis=1)) #cheat to avoid NaN at r=0
    signals_at_points_exact = u_2D_PSV_rt(x - X0_src[np.newaxis,:], np.roll(src_t(dt*np.arange(num_steps)), -2), U0, rho.value,lambda_.value, mu.value, dt)
#
# -----------------------------------------------------


###
# Plotting
livePlot, i_livePlot = True, 10
clim = 0.1*np.amax(U0)*np.array([0, 1])
plotter = CustomVectorPlotter([u_n, u_n if storeAllSteps else None, u_n], clim=clim, warp_factor=0.5/np.amax(clim))
if len(points_output_on_proc)>0: plotter.add_points(points_output_on_proc, render_points_as_spheres=True, point_size=12)
plotter.show(interactive_update=True) #interactive_update=True pour que l'appel ne soit pas bloquant et que le programme se poursuive
###

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

def cbck_livePlot(i, tStepper):
    # Viewing while calculating: Update plotter
    if livePlot and i%i_livePlot==0:
        plotter.update_vectors((tStepper.u_n.x.array, all_u_n_exact[:,:,i].flatten(), tStepper.u_n.x.array-all_u_n_exact[:,:,i].flatten()))
        time.sleep(0.01)

### Run the big time loop!
tStepper.run(num_steps, callfirsts=[cfst_updateSources], callbacks=[cbck_storeFullField, cbck_storeAtPoints, cbck_livePlot])
### End of big calc.
#
# -----------------------------------------------------


# -----------------------------------------------------
#     Interactive view of all time steps if stored
# -----------------------------------------------------
if storeAllSteps: #add a slider to browse through all time steps
    def updateTStep(value):
        i = int((value-tstart)/dt)
        plotter.update_vectors((all_u[i].x.array, all_u_n_exact[:,:,i].flatten(), all_u[i].x.array-all_u_n_exact[:,:,i].flatten()))
    plotter.add_slider_widget(updateTStep, [tstart, tmax-dt])
    plotter.show(interactive_update=False) #important de desactiver le interactive_update sinon l'appel n'est pas bloquant et le programme se termine
#
# -----------------------------------------------------


# -----------------------------------------------------
#              Plot signals at few points
# -----------------------------------------------------
if len(points_output_on_proc)>0:
    fig, ax = plt.subplots(1,1)
    t = dt*np.arange(num_steps)
    for i in range(len(signals_at_points)):
        ax.plot(t, signals_at_points[i,0,:], c='C'+str(i), ls='-') #FEM
        ax.plot(t, signals_at_points_exact[i,0,:], c='C'+str(i), ls='--') #exact
    ax.set_xlabel('Time')
    plt.show()
#
# -----------------------------------------------------

