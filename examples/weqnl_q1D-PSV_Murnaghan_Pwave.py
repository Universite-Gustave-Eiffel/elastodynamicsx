"""
Wave equation (time-domain)

Propagation of a P wave in a 1D nonlinear (Murnaghan) elastic medium
The 1D medium is implemented as a 2D medium + periodic BCs
"""

from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import matplotlib.pyplot as plt

from elastodynamicsx.pde import material, BodyForce, BoundaryCondition, PDE
from elastodynamicsx.solvers import TimeStepper
from elastodynamicsx.plot import plotter
from elastodynamicsx.utils import find_points_and_cells_on_proc, make_facet_tags

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
degElement = 4

length, height = 20, 2
Nx, Ny = 200//degElement, 20//degElement
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny])

tag_left, tag_top, tag_right, tag_bottom = 1, 2, 3, 4
all_tags = (tag_left, tag_top, tag_right, tag_bottom)
boundaries = [(tag_left  , lambda x: np.isclose(x[0], 0     )),\
              (tag_right , lambda x: np.isclose(x[0], length)),\
              (tag_bottom, lambda x: np.isclose(x[1], 0     )),\
              (tag_top   , lambda x: np.isclose(x[1], height))]
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
T_N      = fem.Constant(domain, PETSc.ScalarType([0,0])) #normal traction (Neumann boundary condition)
Z_N, Z_T = mat.Z_N, mat.Z_T #P and S mechanical impedances
bc_l = BoundaryCondition((V, facet_tags, tag_left  ), 'Neumann', T_N)
bc_r = BoundaryCondition((V, facet_tags, tag_right ), 'Dashpot', (Z_N, Z_T))
#bc_b = BoundaryCondition((V, facet_tags, tag_bottom), 'Periodic', [0, height, 0])
bc_b = BoundaryCondition((V, facet_tags, tag_bottom), 'Free')
bcs = [bc_l, bc_r, bc_b]
#
# -----------------------------------------------------


# -----------------------------------------------------
#               (boundary) Source term
# -----------------------------------------------------
### -> Time function
#
f0 = 1 #central frequency of the source
T0 = 1/f0 #period
d0 = 2*T0 #duration of source
#
src_t = lambda t: np.sin(2*np.pi*f0 * t) * np.sin(np.pi*t/d0)**2 * (t<d0) * (t>0) #source(t)

### -> Space-Time function
#
p0  = PETSc.ScalarType(1.)         #max amplitude
F_0 = p0 * PETSc.ScalarType([1,0]) #source orientation
#
T_N_function = lambda t: src_t(t) * F_0
#
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
tstart = 0 # Start time
tmax   = 4*d0 # Final time
num_steps = 1200
dt = (tmax-tstart) / num_steps # time step size
#
# -----------------------------------------------------


###
# Some control numbers...
hx = length/Nx
c_S = np.sqrt(mu.value/rho.value) #S-wave velocity
lbda0 = c_S/f0
print('Number of points per wavelength at central frequency: ', round(lbda0/hx, 2))
print('Number of time steps per period at central frequency: ', round(T0/dt, 2))
print('CFL condition: Courant number = ', round(TimeStepper.Courant_number(V.mesh, ufl.sqrt((lambda_+2*mu)/rho), dt), 2))
###


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(V, materials=materials, bodyforces=[], bcs=bcs)

#  Time integration
tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, scheme='leapfrog')
tStepper.initial_condition(u0=[0,0], v0=[0,0], t0=tstart)
u_res = tStepper.timescheme.u # The solution
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    define outputs
# -----------------------------------------------------
### -> Extract signals at few points
points_output = np.array([0,height/2,0])[:,np.newaxis] + np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]]).T #shape = (3, nbpts)
points_output_on_proc, cells_output_on_proc = find_points_and_cells_on_proc(points_output, domain)
signals_at_points = np.zeros((points_output.shape[1], domain.topology.dim, num_steps)) #<- output stored here
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### define callfirsts and callbacks
def cfst_updateSources(t, tStepper):
    T_N.value = T_N_function(t)

def cbck_storeAtPoints(i, out):
    if len(points_output_on_proc)>0: signals_at_points[:,:,i+1] = u_res.eval(points_output_on_proc, cells_output_on_proc)

### enable live plotting
clim = 0.1*np.amax(F_0)*np.array([0, 1])
kwplot = { 'clim':clim, 'warp_factor':0.5/np.amax(clim) }
p = plotter(u_res, refresh_step=10, **kwplot) #0 to disable
if len(points_output_on_proc)>0:
    p.add_points(points_output_on_proc, render_points_as_spheres=True, point_size=12) #adds points to live_plotter

### Run the big time loop!
tStepper.run(num_steps-1, callfirsts=[cfst_updateSources], callbacks=[cbck_storeAtPoints], live_plotter=p)
### End of big calc.
#
# -----------------------------------------------------


# -----------------------------------------------------
#              Plot signals at few points
# -----------------------------------------------------
if len(points_output_on_proc)>0:
    t = dt*np.arange(num_steps)
    fig, ax = plt.subplots(1,1)
    ax.set_title('Signals at few points')
    for i in range(len(signals_at_points)):
        ax.plot(t, signals_at_points[i,0,:],       c='C'+str(i), ls='-') #FEM
        #ax.plot(t, signals_at_points_exact[i,0,:], c='C'+str(i), ls='--') #exact
    ax.set_xlabel('Time')
    plt.show()
#
# -----------------------------------------------------

