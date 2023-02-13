import numpy as np
from dolfinx import plot
import pyvista

import elastodynamicsx.plot #ensures automatic configuration of pyvista for jupyter


class ModalBasis():
    """
    Representation of a modal basis, consisting of a set of eigen angular frequencies and modeshapes.
    
    at the moment: is merely a storage + plotter class
    in the future: should be able to perform calculations with source terms, such as modal participation factors, modal summations, ...
    """
    
    def __init__(self, wn, un, **kwargs):
        """
        Args:
            wn: eigen angular frequencies
            un: eigen modeshapes
        """
        self._wn = wn
        self._un = un

    @property
    def fn(self):
        """The eigen frequencies"""
        return self._wn/(2*np.pi)

    @property
    def wn(self):
        """The eigen angular frequencies"""
        return self._wn
    
    @property
    def un(self):
        """The eigen modeshapes"""
        return self._un
    
    def plot(self, function_space, which='all', **kwargs):
        """
        Plots the desired modeshapes
        
        Args:
            which: 'all', or an integer, or a list of integers, or a slice object
            kwargs:
                shape: (default: attempts a square mosaic) shape of the pyvista.Plotter
                factor: (default=1) Scale factor for the deformation
                wireframe: (default=False) Plot the wireframe of the undeformed mesh

        Examples of use:
            plot()                #plots all computed eigenmodes
            plot(3)               #plots mode number 4
            plot([3,5])           #plots modes number 4 and 6
            plot(slice(0,None,2)) #plots even modes
        """
        #inspired from https://docs.pyvista.org/examples/99-advanced/warp-by-vector-eigenmodes.html
        indexes    = _slice_array(np.arange(len(self._wn)), which)
        eigenmodes = _slice_array(self.un, which)
        eigenfreqs = _slice_array(self.fn, which)
        #
        topology, cell_types, geom = plot.create_vtk_mesh(function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        for i, eigM in zip(indexes, eigenmodes):
            nbpts = grid.number_of_points
            grid['eigenmode_'+str(i)] = elastodynamicsx.plot.get_3D_array_from_nparray(np.array(eigM), nbpts)
        #
        nbcols = int(np.ceil(np.sqrt(indexes.size)))
        nbrows = int(np.ceil(indexes.size/nbcols))
        shape  = kwargs.get('shape', (nbrows, nbcols))
        factor = kwargs.get('factor', 1.)
        plotter = pyvista.Plotter(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                plotter.subplot(i,j)
                current_index = i*shape[1] + j
                if current_index >= indexes.size: break
                vector = 'eigenmode_'+str(indexes[current_index])
                plotter.add_text("mode "+str(indexes[current_index])+", freq. "+str(round(eigenfreqs[current_index],2)), font_size=10)
                if kwargs.get('wireframe', False): plotter.add_mesh(grid, style='wireframe', color='black')
                plotter.add_mesh(grid.warp_by_vector(vector, factor=factor), scalars=vector)
        plotter.show()
        
        
def _slice_array(a, which):
    """Not intended to be used externally"""
    if which == 'all'    : which = slice(0,None,None)
    if type(which) is int: which = slice(which, which+1, None)
    return a[which]

