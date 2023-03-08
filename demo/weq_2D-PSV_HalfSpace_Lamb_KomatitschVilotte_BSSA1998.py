# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT


"""
Wave equation (time-domain)

Lamb's problem, i.e. the response of a half space to a source on its surface.

After [1], Fig. 1:

[1] Komatitsch, D., & Vilotte, J. P. (1998). The spectral element method: an efficient tool to simulate the seismic response of 2D and 3D geological structures. Bulletin of the seismological society of America, 88(2), 368-392.
"""


from dolfinx import mesh, fem
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
import matplotlib.pyplot as plt

from elastodynamicsx.pde     import material, BoundaryCondition, PDE
from elastodynamicsx.solvers import TimeStepper
from elastodynamicsx.plot    import plotter
from elastodynamicsx.utils   import spectral_element, spectral_quadrature, find_points_and_cells_on_proc
from models.model_Lamb_KomatitschVilotte_BSSA1998 import create_model


# -----------------------------------------------------
#                     FE domain
# -----------------------------------------------------
# Set up a Spectral Element Method
degElement, nameElement = 8, "GLL"
cell_type = mesh.CellType.quadrilateral
specFE               = spectral_element(nameElement, cell_type, degElement)
PDE.default_metadata = spectral_quadrature(nameElement, degElement)

# Create a GMSH model
sizefactor = 0.5
tilt = 10  # tilt angle (degrees)
tagBdFree, tagBdInt = 1, 2
model = create_model(sizefactor=sizefactor, tilt=tilt, tagBdFree=tagBdFree, tagBdInt=tagBdInt)

# Convert the GMSH model into a DOLFINx mesh
gmsh_model_rank = 0
comm = MPI.COMM_WORLD
domain, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, gmsh_model_rank, gdim=2)

#
#V = fem.VectorFunctionSpace(domain, specFE, dim=2) #currently does not work
# workaround
import basix.ufl_wrapper
e        = specFE
specFE_v = basix.ufl_wrapper.create_vector_element(e.family(), e.cell_type, e.degree(), e.lagrange_variant, e.dpc_variant, e.discontinuous, dim=2, gdim=domain.geometry.dim)
V = fem.FunctionSpace(domain, specFE_v)
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Material parameters
# -----------------------------------------------------
rho     = 2.2  # density
cP, cS  = 3.2, 1.8475  # P- and S-wave velocities
c11, c44= rho * cP**2, rho * cS**2
rho     = fem.Constant(domain, PETSc.ScalarType(rho))
mu      = fem.Constant(domain, PETSc.ScalarType(c44))
lambda_ = fem.Constant(domain, PETSc.ScalarType(c11-2*c44))

mat   = material(V, 'isotropic', rho, lambda_, mu)
materials = [mat]
#
# -----------------------------------------------------


# -----------------------------------------------------
#                 Boundary conditions
# -----------------------------------------------------
Z_N, Z_T = mat.Z_N, mat.Z_T # P and S mechanical impedances
T_N    = fem.Function(V)      # Normal traction (Neumann boundary condition)
bc_top = BoundaryCondition((V, facet_tags, tagBdFree), 'Neumann', T_N)
bc_int = BoundaryCondition((V, facet_tags, tagBdInt), 'Dashpot', (Z_N, Z_T))  # Absorbing BC on the artificial boundaries
bcs = [bc_int, bc_top]
#
# -----------------------------------------------------


# -----------------------------------------------------
#               (boundary) Source term
# -----------------------------------------------------
### -> Space function
#
length = 4*sizefactor
X0_src = length/2 # Center
W0_src = 0.2*length/50 # Width
#
### Gaussian function
nrm   = 1/np.sqrt(2*np.pi*W0_src**2)  #normalize to int[src_x(x) dx]=1

def src_x(x):
    return nrm * np.exp(-1/2*((x[0]-X0_src)/W0_src)**2, dtype=PETSc.ScalarType) # Source(x)
#
### -> Time function
#
fc  = 14.5 # Central frequency
sig = np.sqrt(2)/(2*np.pi*fc)  # Gaussian standard deviation
t0  = 4*sig

def src_t(t):
    return (1 - ((t-t0)/sig)**2) * np.exp(-0.5*((t-t0)/sig)**2)  # Source(t)
#
### -> Space-Time function
#
p0  = 1.  # Max amplitude
F_0 = p0 * PETSc.ScalarType([np.sin(np.radians(tilt)), -np.cos(np.radians(tilt))])  # Amplitude of the source
#
def T_N_function(t):
    return lambda x: F_0[:,np.newaxis] * src_t(t) * src_x(x)[np.newaxis,:]  #source(x) at a given time

# -----------------------------------------------------


# -----------------------------------------------------
#           Time scheme: Temporal parameters
# -----------------------------------------------------
tstart = 0 # Start time
dt     = 0.25e-3 # Time step
num_steps = int(6000*sizefactor)
#
# -----------------------------------------------------


###
PETSc.Sys.Print('CFL condition: Courant number = ', round(TimeStepper.Courant_number(V.mesh, ufl.sqrt((lambda_+2*mu)/rho), dt), 2))
###


# -----------------------------------------------------
#                        PDE
# -----------------------------------------------------
pde = PDE(V, materials=materials, bodyforces=[], bcs=bcs)

#  Time integration
tStepper = TimeStepper.build(V, pde.m, pde.c, pde.k, pde.L, dt, bcs=bcs, scheme='leapfrog', diagonal=True)
tStepper.initial_condition(u0=[0,0], v0=[0,0], t0=tstart)
u_res = tStepper.timescheme.u # The solution
#
# -----------------------------------------------------


# -----------------------------------------------------
#                       Solve
# -----------------------------------------------------
### define callfirsts and callbacks
def cfst_updateSources(t, tStepper):
    T_N.interpolate(T_N_function(t))

### enable live plotting
clim = 0.015*np.linalg.norm(F_0)*np.array([0, 1])
if domain.comm.rank == 0:
    kwplot = {'refresh_step':30, 'clim':clim, 'show_edges':False, 'warp_factor':0.05/np.amax(clim) }
else:
    kwplot = None

### Run the big time loop!
tStepper.run(num_steps-1, callfirsts=[cfst_updateSources], callbacks=[], live_plotter=kwplot)
### End of big calc.
#
# -----------------------------------------------------

#TODO: outputs, analytical formula, check CFL

