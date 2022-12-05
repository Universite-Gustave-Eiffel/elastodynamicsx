#

import numpy as np
import matplotlib.pyplot as plt
from dolfinx import plot
import pyvista

class CustomScalarPlotter(pyvista.Plotter):
    
    def __init__(self, all_scalars, **kwargs):
        self.grids=[]
        for u_ in all_scalars:
            if u_ is None: break
            topology, cell_types, geom = plot.create_vtk_mesh(u_.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            grid.point_data["u"] = u_.x.array
            self.grids.append(grid)
        
        if len(self.grids)==1: 
            super().__init__()
        else:
            pyvista.global_theme.multi_rendering_splitting_position = 0.75
            super().__init__(shape='1|'+str(len(self.grids)-1))

        labels = kwargs.pop('labels', ('FE', 'Exact', 'Diff.'))
        show_edges = kwargs.pop('show_edges', True)
        cmap = kwargs.pop('cmap', plt.cm.get_cmap("viridis", 25))
        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black", position_x=0.1, position_y=0.8, width=0.8, height=0.1)
        for i,grid in enumerate(self.grids):
            if len(self.grids)>1: self.subplot(i)
            self.add_text(labels[i])
            self.add_mesh(grid, scalars='u', show_edges=((i==0) and show_edges), lighting=False, scalar_bar_args=sargs, cmap=cmap, **kwargs)
            self.view_xy()
            self.camera.zoom(1.3)

        if len(self.grids)>1: self.subplot(0) #resets the focus to first subplot

    def update_scalars(self, all_scalars, render=True):
        for i, (grid, u_) in enumerate(zip(self.grids, all_scalars)):
            super().update_scalars(u_, mesh=grid, render=False)
        if render:
            self.render()


class CustomVectorPlotter(pyvista.Plotter):
    
    def __init__(self, all_vectors, **kwargs):
        self.grids=[]
        self.warp_factor = kwargs.pop('warp_factor', 1)
        for u_ in all_vectors:
            if u_ is None: break
            topology, cell_types, geom = plot.create_vtk_mesh(u_.function_space)
            z0s = np.zeros((geom.shape[0], 1), dtype=u_.x.array.dtype)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            grid["u"] = np.append(u_.x.array.reshape((geom.shape[0], 2)), z0s, axis=1)
            grid.point_data["u_nrm"] = np.linalg.norm(u_.x.array.reshape((geom.shape[0], 2)), axis=1)
            self.grids.append(grid)
        
        if len(self.grids)==1: 
            super().__init__()
        else:
            pyvista.global_theme.multi_rendering_splitting_position = 0.75
            super().__init__(shape='1|'+str(len(self.grids)-1))

        labels = kwargs.pop('labels', ('FE', 'Exact', 'Diff.'))
        show_edges = kwargs.pop('show_edges', True)
        cmap = kwargs.pop('cmap', plt.cm.get_cmap("viridis", 25))
        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black", position_x=0.1, position_y=0.8, width=0.8, height=0.1)
        for i,grid in enumerate(self.grids):
            if len(self.grids)>1: self.subplot(i)
            self.add_text(labels[i])
            warped = grid.warp_by_vector("u", factor=self.warp_factor)
            grid.warped = warped
            self.add_mesh(warped, scalars="u_nrm", show_edges=((i==0) and show_edges), lighting=False, scalar_bar_args=sargs, cmap=cmap, **kwargs)
            self.view_xy()
            self.camera.zoom(1.3)

        if len(self.grids)>1: self.subplot(0) #resets the focus to first subplot

    def update_vectors(self, all_vectors, render=True):
        for i, (grid, u_) in enumerate(zip(self.grids, all_vectors)):
            nbpts = grid.number_of_points
            z0s = np.zeros((nbpts, 1), dtype=u_.dtype)
            grid["u"] = np.append(u_.reshape((nbpts, 2)), z0s, axis=1)
            #
            super().update_coordinates(grid.warp_by_vector("u", factor=self.warp_factor).points, mesh=grid.warped, render=False)
            super().update_scalars(np.linalg.norm(u_.reshape((nbpts, 2)), axis=1), mesh=grid.warped, render=False)
        if render:
            self.render()


