#

import numpy as np
import matplotlib.pyplot as plt
from dolfinx import plot
import pyvista

def plot_mesh(mesh, cell_tags):
    """
    Adapted from: https://jsdokken.com/dolfinx-tutorial/chapter3/em.html
    
    example of use:
    
    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square
    from elastodynamicsx.utils import make_tags
    #
    domain = create_unit_square(MPI.COMM_WORLD, 10, 10)

    Omegas = [(1, lambda x: x[1] <= 0.5),
              (2, lambda x: x[1] >= 0.5)]
    cell_tags = make_tags(domain, Omegas, 'domains')
    
    plot_mesh(domain, cell_tags)
    """
    plotter = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices<num_local_cells]
    grid.set_active_scalars("Marker")
    actor = plotter.add_mesh(grid, show_edges=True)
    #
    if mesh.topology.dim<3:
        plotter.view_xy()
    #
    plotter.show()

class CustomScalarPlotter(pyvista.Plotter):
    
    def __init__(self, *all_scalars, **kwargs):
        self.grids=[]
        dims = []
        for u_ in all_scalars:
            if u_ is None: break
            topology, cell_types, geom = plot.create_vtk_mesh(u_.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            grid.point_data["u"] = u_.x.array
            self.grids.append(grid)
            dims.append(u_.function_space.mesh.topology.dim)
        
        if len(self.grids)==1:
            defaultShape = (1,1)
        else:
            pyvista.global_theme.multi_rendering_splitting_position = 0.75
            defaultShape = '1|'+str(len(self.grids)-1)
        shape = kwargs.pop('shape', defaultShape)
        super().__init__(shape=shape)

        labels = kwargs.pop('labels', ['' for i in range(len(self.grids))])
        show_edges = kwargs.pop('show_edges', True)
        cmap = kwargs.pop('cmap', plt.cm.get_cmap("viridis", 25))
        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black", position_x=0.1, position_y=0.8, width=0.8, height=0.1)
        for i,(grid,dim) in enumerate(zip(self.grids, dims)):
            if len(self.grids)>1: self.subplot(i)
            self.add_text(labels[i])
            self.add_mesh(grid, scalars='u', show_edges=((i==0) and show_edges), lighting=False, scalar_bar_args=sargs, cmap=cmap, **kwargs)
            if dim<3:
                self.view_xy()
                self.camera.zoom(1.3)

        if len(self.grids)>1: self.subplot(0) #resets the focus to first subplot

    def update_scalars(self, *all_scalars, **kwargs):
        for i, (grid, u_) in enumerate(zip(self.grids, all_scalars)):
            super().update_scalars(u_, mesh=grid, render=False)
        if kwargs.get('render', True):
            self.render()

    def add_time_browser(self, update_fields_function, timesteps, **kwargs_slider):
        self._time_browser_cbck  = update_fields_function
        self._time_browser_times = timesteps
        def updateTStep(value):
            i = np.argmin(np.abs(value-self._time_browser_times))
            updated_fields = self._time_browser_cbck(i)
            self.update_scalars(*updated_fields)
        self.add_slider_widget(updateTStep, [timesteps[0], timesteps[-1]], **kwargs_slider)



class CustomVectorPlotter(pyvista.Plotter):
    
    def __init__(self, *all_vectors, **kwargs):
        ###
        self.grids=[]
        
        if 'warp_factor' in kwargs:
            self.warp_factor = kwargs.pop('warp_factor')
        elif 'clim' in kwargs and np.amax(np.abs(kwargs['clim']))>0:
            self.warp_factor = 0.5/np.amax(np.abs(kwargs['clim']))
        else:
            self.warp_factor = 1
        
        for u_ in all_vectors:
            if u_ is None: break
            topology, cell_types, geom = plot.create_vtk_mesh(u_.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            u3D   = get_3D_array_from_FEFunction(u_)
            grid["u"] = u3D
            grid.point_data["u_nrm"] = np.linalg.norm(u3D, axis=1)
            self.grids.append(grid)
        
        nbcomps = max(1, u_.function_space.element.num_sub_elements)
        if len(self.grids)==1:
            defaultShape = (1,1)
        else:
            pyvista.global_theme.multi_rendering_splitting_position = 0.75
            defaultShape = '1|'+str(len(self.grids)-1)
        shape = kwargs.pop('shape', defaultShape)
        super().__init__(shape=shape)

        labels = kwargs.pop('labels', ['' for i in range(len(self.grids))])
        show_edges = kwargs.pop('show_edges', True)
        cmap = kwargs.pop('cmap', plt.cm.get_cmap("viridis", 25))
        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black", position_x=0.1, position_y=0.8, width=0.8, height=0.1)
        for i,grid in enumerate(self.grids):
            if len(self.grids)>1: self.subplot(i)
            self.add_text(labels[i])
            warped = grid.warp_by_vector("u", factor=self.warp_factor)
            grid.warped = warped
            self.add_mesh(warped, scalars="u_nrm", show_edges=((i==0) and show_edges), lighting=False, scalar_bar_args=sargs, cmap=cmap, **kwargs)
            if nbcomps<3:
                self.view_xy()
                self.camera.zoom(1.3)

        if len(self.grids)>1: self.subplot(0) #resets the focus to first subplot

    def update_vectors(self, *all_vectors, render=True):
        for i, (grid, u_) in enumerate(zip(self.grids, all_vectors)):
            nbpts = grid.number_of_points
            u3D   = get_3D_array_from_nparray(u_, nbpts)
            grid["u"] = u3D
            #
            super().update_coordinates(grid.warp_by_vector("u", factor=self.warp_factor).points, mesh=grid.warped, render=False)
            super().update_scalars(np.linalg.norm(u3D, axis=1), mesh=grid.warped, render=False)
        if render:
            self.render()

    def add_time_browser(self, update_fields_function, timesteps, **kwargs_slider):
        self._time_browser_cbck  = update_fields_function
        self._time_browser_times = timesteps
        def updateTStep(value):
            i = np.argmin(np.abs(value-self._time_browser_times))
            updated_fields = self._time_browser_cbck(i)
            self.update_vectors(*updated_fields)
        self.add_slider_widget(updateTStep, [timesteps[0], timesteps[-1]], **kwargs_slider)
        
        

def get_3D_array_from_FEFunction(u_):
    #u_ is a fem.Function
    nbcomps = max(1, u_.function_space.element.num_sub_elements) #number of components
    nbpts   = u_.x.array.size // nbcomps
    if nbcomps < 3:
        z0s = np.zeros((nbpts, 3-nbcomps), dtype=u_.x.array.dtype)
        return np.append(u_.x.array.reshape((nbpts, nbcomps)), z0s, axis=1)
    else:
        return u_.x.array.reshape((nbpts, 3))

def get_3D_array_from_nparray(u_, nbpts):
    #u_ is a np.array
    nbcomps = u_.size//nbpts
    if nbcomps < 3:
        z0s = np.zeros((nbpts, 3-nbcomps), dtype=u_.dtype)
        return np.append(u_.reshape((nbpts, nbcomps)), z0s, axis=1)
    else:
        return u_.reshape((nbpts, 3))

