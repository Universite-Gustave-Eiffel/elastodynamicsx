"""
Wave equation (time-domain)

Propagation of SH elastic waves in a 2D, homogeneous isotropic solid, and comparison with an analytical solution
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
from elastodynamicsx.utils import find_points_and_cells_on_proc, make_facet_tags, make_cell_tags
from analyticalsolutions import u_2D_SH_rt, int_Fraunhofer_2D

# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
degElement = 4

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
V  = fem.FunctionSpace(domain, ("DG", degElement))
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho     = fem.Constant(domain, PETSc.ScalarType(1))
mu      = fem.Constant(domain, PETSc.ScalarType(1))
#lambda_ = fem.Constant(domain, PETSc.ScalarType(2))

mat   = material(V, 'scalar', rho, mu)
materials = [mat]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
Z = mat.Z #mechanical impedance
bc_l = BoundaryCondition((V, facet_tags, tag_left  ), 'Dashpot', Z)
bc_r = BoundaryCondition((V, facet_tags, tag_right ), 'Dashpot', Z)
bc_b = BoundaryCondition((V, facet_tags, tag_bottom), 'Dashpot', Z)
bc_t = BoundaryCondition((V, facet_tags, tag_top   ), 'Dashpot', Z)
bcs = [bc_l, bc_r, bc_b, bc_t]
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

### Body force 'F_body'
F_body = fem.Function(V) #body force
gaussianBF = BodyForce(V, F_body)

bodyforces = [gaussianBF]
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

print('Number of points per wavelength at central frequency: ', round(lbda0/hx, 2))
print('Number of time steps per period at central frequency: ', round(T0/dt, 2))
print('CFL condition: Courant number = ', round(TimeStepper.Courant_number(V.mesh, ufl.sqrt(mu/rho), dt), 2))
###


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(V, materials=materials, bodyforces=bodyforces, bcs=bcs)

#  Time integration
tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, scheme='leapfrog')
tStepper.initial_condition(u0=0, v0=0, t0=tstart)
u_res = tStepper.timescheme.u # The solution
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    define outputs
# -----------------------------------------------------
### -> Store all time steps ? -> YES if debug & learning // NO if big calc.
storeAllSteps = True
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
def cfst_updateSources(t, tStepper):
    F_body.interpolate(F_body_function(t))

def cbck_storeFullField(i, out):
    if storeAllSteps: all_u[i+1].x.array[:] = u_res.x.array
def cbck_storeAtPoints(i, out):
    if len(points_output_on_proc)>0: signals_at_points[:,:,i+1] = u_res.eval(points_output_on_proc, cells_output_on_proc)

### enable live plotting
clim = 0.1*F_0*np.array([-1, 1])
p = plotter(u_res, refresh_step=10, **{'clim':clim}) #0 to disable
if len(points_output_on_proc)>0:
    p.add_points(points_output_on_proc, render_points_as_spheres=True, point_size=12) #adds points to live_plotter

### Run the big time loop!
tStepper.run(num_steps-1, callfirsts=[cfst_updateSources], callbacks=[cbck_storeFullField, cbck_storeAtPoints], live_plotter=p)
### End of big calc.
#
# -----------------------------------------------------


# -----------------------------------------------------
#     Interactive view of all time steps if stored
# -----------------------------------------------------
if storeAllSteps: #plotter with a slider to browse through all time steps
    fn_kdomain_finite_size = int_Fraunhofer_2D['gaussian'](R0_src) #accounts for the size of the source in the analytical formula
    
    ### -> Exact solution, Full field
    x = u_res.function_space.tabulate_dof_coordinates()
    r = np.linalg.norm(x - X0_src[np.newaxis,:], axis=1)
    t = dt*np.arange(num_steps)
    all_u_n_exact = u_2D_SH_rt(r, src_t(t), rho.value, mu.value, dt, fn_kdomain_finite_size)
    
    def update_fields_function(i):
        return (all_u[i].x.array, all_u_n_exact[:,i], all_u[i].x.array-all_u_n_exact[:,i])
    
    #initializes with empty fem.Function(V) to have different valid pointers
    p = plotter(fem.Function(V), fem.Function(V), fem.Function(V), labels=('FE', 'Exact', 'Diff.'), clim=clim)
    p.add_time_browser(update_fields_function, t)
    p.show()
#
# -----------------------------------------------------


# -----------------------------------------------------
#              Plot signals at few points
# -----------------------------------------------------
if len(points_output_on_proc)>0:
    fn_kdomain_finite_size = int_Fraunhofer_2D['gaussian'](R0_src) #accounts for the size of the source in the analytical formula
    
    ### -> Exact solution, At few points
    x = points_output_on_proc
    r = np.linalg.norm(x - X0_src[np.newaxis,:], axis=1)
    t = dt*np.arange(num_steps)
    signals_at_points_exact = u_2D_SH_rt(r, src_t(t), rho.value, mu.value, dt, fn_kdomain_finite_size)
    #
    fig, ax = plt.subplots(1,1)
    ax.set_title('Signals at few points')
    for i in range(len(signals_at_points)):
        ax.plot(t, signals_at_points[i,0,:],     c='C'+str(i), ls='-') #FEM
        ax.plot(t, signals_at_points_exact[i,:], c='C'+str(i), ls='--') #exact
    ax.set_xlabel('Time')
    plt.show()
#
# -----------------------------------------------------

