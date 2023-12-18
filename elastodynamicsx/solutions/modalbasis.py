# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

import typing

import numpy as np
from petsc4py import PETSc
from dolfinx import plot, fem
import pyvista

import elastodynamicsx.plot  # ensures automatic configuration of pyvista for jupyter


class ModalBasis():
    """
    Representation of a modal basis, consisting of a set of eigen angular
    frequencies :math:`\omega_n` and modeshapes :math:`\mathbf{u}_n`.

    At the moment: is merely a storage + plotter class
    In the future: should be able to perform calculations with source terms,
    such as modal participation factors, modal summations, ...

    Args:
        wn: eigen angular frequencies
        un: eigen modeshapes
    """

    def __init__(self, wn: np.ndarray, un: typing.List[PETSc.Vec], **kwargs):  # type: ignore[name-defined]
        self._wn = wn
        self._un = un

    @property
    def fn(self) -> np.ndarray:
        """The eigen frequencies :math:`f_n = \omega_n/2\pi`"""
        return self._wn / (2 * np.pi)

    @property
    def wn(self) -> np.ndarray:
        """The eigen angular frequencies :math:`\omega_n`"""
        return self._wn

    @property
    def un(self) -> typing.List[PETSc.Vec]:  # type: ignore[name-defined]
        """The eigen modeshapes :math:`\mathbf{u}_n`"""
        return self._un

    def plot(self, function_space: fem.FunctionSpaceBase, which='all', **kwargs) -> None:
        """
        Plots the desired modeshapes

        Args:
            function_space: The underlying function space
            which: 'all', or an integer, or a list of integers, or a slice object
            kwargs:
                shape: (default: attempts a square mosaic) shape of the pyvista.Plotter
                factor: (default=1) Scale factor for the deformation
                wireframe: (default=False) Plot the wireframe of the undeformed mesh

        Example:
            .. highlight:: python
            .. code-block:: python

              plot(V)                   # plots all computed eigenmodes
              plot(V, 3)                # plots mode number 4
              plot(V, [3,5])            # plots modes number 4 and 6
              plot(V, slice(0,None,2))  # plots even modes
        """
        # inspired from https://docs.pyvista.org/examples/99-advanced/warp-by-vector-eigenmodes.html
        indexes = _slice_array(np.arange(len(self._wn)), which)
        eigenmodes = _slice_array(self.un, which)
        eigenfreqs = _slice_array(self.fn, which)
        #
        topology, cell_types, geom = plot.vtk_mesh(function_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)

        for i, eigM in zip(indexes, eigenmodes):
            nbpts = grid.number_of_points
            with eigM.localForm() as loc_eigM:  # Necessary for correct handling of ghosts in parallel
                grid['eigenmode_' + str(i)] = elastodynamicsx.plot.get_3D_array_from_nparray(loc_eigM.array, nbpts)

        nbcols = int(np.ceil(np.sqrt(indexes.size)))
        nbrows = int(np.ceil(indexes.size / nbcols))
        shape = kwargs.pop('shape', (nbrows, nbcols))
        factor = kwargs.pop('factor', 1.)
        wireframe = kwargs.pop('wireframe', False)
        if (wireframe is True) and not ('opacity' in kwargs.keys()):
            kwargs['opacity'] = 0.8

        plotter = pyvista.Plotter(shape=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                plotter.subplot(i, j)
                current_index = i * shape[1] + j

                if current_index >= indexes.size:
                    break

                vector = 'eigenmode_' + str(indexes[current_index])
                plotter.add_text("mode " + str(indexes[current_index]) + ", freq. "
                                 + str(round(eigenfreqs[current_index], 2)), font_size=10)

                if wireframe:
                    plotter.add_mesh(grid, style='wireframe', color='black')

                plotter.add_mesh(grid.warp_by_vector(vector, factor=factor), scalars=vector, **kwargs)
        plotter.show()


def _slice_array(a, which):
    """Not intended to be used externally"""
    if which == 'all':
        which = slice(0, None, None)

    if type(which) is int:
        which = slice(which, which + 1, None)

    return a[which]
