# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT


"""
Wave equation (time-domain)

Propagation of P and SV elastic waves in a 2D, homogeneous isotropic solid, and comparison with an analytical solution
"""


from dolfinx import mesh, fem, default_scalar_type
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import matplotlib.pyplot as plt

from elastodynamicsx.pde import material, BodyForce, boundarycondition, PDE, PDECONFIG
from elastodynamicsx.solvers import TimeStepper
from elastodynamicsx.plot import plotter
from elastodynamicsx.utils import spectral_element, spectral_quadrature, make_facet_tags, make_cell_tags, ParallelEvaluator
from analyticalsolutions import u_2D_PSV_xt, int_Fraunhofer_2D


# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
# Set up a Spectral Element Method
degElement, nameElement = 4, "GLL"
cell_type = mesh.CellType.quadrilateral
specFE = spectral_element(nameElement, cell_type, degElement, (2,))
PDECONFIG.default_metadata = spectral_quadrature(nameElement, degElement)

length, height = 10, 10
Nx, Ny = 100//degElement, 100//degElement
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(MPI.COMM_WORLD, extent, [Nx, Ny], cell_type)

tag_left, tag_top, tag_right, tag_bottom = 1, 2, 3, 4
all_tags = (tag_left, tag_top, tag_right, tag_bottom)
boundaries = [(tag_left  , lambda x: np.isclose(x[0], 0     )),\
              (tag_right , lambda x: np.isclose(x[0], length)),\
              (tag_bottom, lambda x: np.isclose(x[1], 0     )),\
              (tag_top   , lambda x: np.isclose(x[1], height))]
facet_tags = make_facet_tags(domain, boundaries)
#
V = fem.FunctionSpace(domain, specFE)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho     = fem.Constant(domain, default_scalar_type(1))
mu      = fem.Constant(domain, default_scalar_type(1))
lambda_ = fem.Constant(domain, default_scalar_type(2))

mat   = material(V, 'isotropic', rho, lambda_, mu)
materials = [mat]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
Z_N, Z_T = mat.Z_N, mat.Z_T  # P and S mechanical impedances
bc_l = boundarycondition((V, facet_tags, tag_left  ), 'Dashpot', Z_N, Z_T)
bc_r = boundarycondition((V, facet_tags, tag_right ), 'Dashpot', Z_N, Z_T)
bc_b = boundarycondition((V, facet_tags, tag_bottom), 'Dashpot', Z_N, Z_T)
bc_t = boundarycondition((V, facet_tags, tag_top   ), 'Dashpot', Z_N, Z_T)
bcs = [bc_l, bc_r, bc_b, bc_t]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    Source term
# -----------------------------------------------------
### -> Space function
#
X0_src = np.array([length/2,height/2,0])  # center
R0_src = 0.1  # radius
#
### Gaussian function
nrm   = 1/(2*np.pi*R0_src**2) #normalize to int[src_x(x) dx]=1

def src_x(x):  # source(x): Gaussian
    r = np.linalg.norm(x-X0_src[:,np.newaxis], axis=0)
    return nrm * np.exp(-1/2*(r/R0_src)**2, dtype=default_scalar_type)
#
### -> Time function
#
f0 = 1  # central frequency of the source
T0 = 1/f0  # period
d0 = 2*T0  # duration of source
#
def src_t(t):  # source(t): Sine x Hann window
    window = np.sin(np.pi*t/d0)**2 * (t<d0) * (t>0)  # Hann window
    return np.sin(2*np.pi*f0 * t) * window

### -> Space-Time function
#
F_0 = default_scalar_type([1,0])  # amplitude of the source
#
def F_body_function(t):  # source(x) at a given time
    return lambda x: F_0[:,np.newaxis] * src_t(t) * src_x(x)[np.newaxis,:]

### Body force 'F_body'
F_body = fem.Function(V) #body force
gaussianBF = BodyForce(V, F_body)

bodyforces = [gaussianBF]
# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
tstart = 0  # Start time
tmax   = 4*d0  # Final time
num_steps = 1200
dt = (tmax-tstart) / num_steps  # time step size
#
# -----------------------------------------------------


###
# Some control numbers...
hx = length/Nx
c_S = np.sqrt(mu.value/rho.value) #S-wave velocity
lbda0 = c_S/f0
PETSc.Sys.Print('Number of points per wavelength at central frequency: ', round(lbda0/hx, 2))
PETSc.Sys.Print('Number of time steps per period at central frequency: ', round(T0/dt, 2))
PETSc.Sys.Print('CFL condition: Courant number = ', round(TimeStepper.Courant_number(V.mesh, ufl.sqrt((lambda_+2*mu)/rho), dt), 2))
###


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(V, materials=materials, bodyforces=bodyforces, bcs=bcs)

#  Time integration
tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, scheme='leapfrog', diagonal=True)  # diagonal=True assumes the left hand side operator is indeed diagonal
tStepper.set_initial_condition(u0=[0,0], v0=[0,0], t0=tstart)
u_res = tStepper.timescheme.u # The solution
#
# -----------------------------------------------------


# -----------------------------------------------------
#                    define outputs
# -----------------------------------------------------
### -> Extract signals at few points
# Define points
points_out = X0_src[:,np.newaxis] + np.array([[1, 1, 0], [2, 2, 0], [3, 3, 0]]).T  # shape = (3, nbpts)

# Declare a convenience ParallelEvaluator
paraEval = ParallelEvaluator(domain, points_out)

# Declare data (local)
signals_local = np.zeros((paraEval.nb_points_local,
                          V.num_sub_spaces,
                          num_steps))  # <- output stored here
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### define callfirsts and callbacks
def cfst_updateSources(t):
    F_body.interpolate(F_body_function(t))

def cbck_storeAtPoints(i, out):
    if paraEval.nb_points_local > 0:
        signals_local[:,:,i+1] = u_res.eval(paraEval.points_local, paraEval.cells_local)

### enable live plotting
enable_plot = True
if domain.comm.rank == 0 and enable_plot:
    clim = 0.1*np.amax(F_0)*np.array([0, 1])
    kwplot = { 'clim':clim, 'warp_factor':0.5/np.amax(clim) }
    p = plotter(u_res, refresh_step=10, **kwplot)
    if paraEval.nb_points_local > 0:
        # Add points to live_plotter
        p.add_points(paraEval.points_local, render_points_as_spheres=True, point_size=12)
else:
    p = None

### Run the big time loop!
tStepper.solve(num_steps-1,
               callfirsts=[cfst_updateSources],
               callbacks=[cbck_storeAtPoints],
               live_plotter=p)
### End of big calc.
#
# -----------------------------------------------------


# -----------------------------------------------------
#              Plot signals at few points
# -----------------------------------------------------
all_signals = paraEval.gather(signals_local, root=0)

if domain.comm.rank == 0:
    # Account for the size of the source in the analytical formula
    fn_kdomain_finite_size = int_Fraunhofer_2D['gaussian'](R0_src)
    
    ### -> Exact solution, At few points
    x = points_out.T
    t = dt*np.arange(num_steps)
    signals_exact = u_2D_PSV_xt(x - X0_src[np.newaxis,:], src_t(t), F_0, rho.value,lambda_.value, mu.value, dt, fn_kdomain_finite_size)
    #
    fig, ax = plt.subplots(1,1)
    ax.set_title('Signals at few points')
    for i in range(len(all_signals)):
        ax.plot(t, all_signals[i,0,:],   c='C'+str(i), ls='-')   # FEM
        ax.plot(t, signals_exact[i,0,:], c='C'+str(i), ls='--')  # exact
    ax.set_xlabel('Time')
    ax.legend(['FEM', 'analytical'])
    plt.show()
#
# -----------------------------------------------------

