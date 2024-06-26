# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

from typing import Union, List, Callable

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import pyvista

from petsc4py import PETSc
from dolfinx import plot, fem
from dolfinx.mesh import Mesh, MeshTags

from elastodynamicsx import _DOCS_CFG


# ## ------------------------------------------------------------------------- ## #
# ## --- preliminary: auto-configure pyvista backend for jupyter notebooks --- ## #
# ## ------------------------------------------------------------------------- ## #

pyvista.global_theme.background = pyvista.Color('white')
pyvista.global_theme.font.color = pyvista.Color('grey')


def _is_notebook() -> bool:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/24937408
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if _is_notebook():
    # DEFAULT_JUPYTER_BACKEND = "ipyvtklink"
    # pyvista.set_jupyter_backend(DEFAULT_JUPYTER_BACKEND)
    # pyvista.start_xvfb()  # required by ipyvtklink
    pass


# ## ---------------------------------------- ## #
# ## --- define useful plotting functions --- ## #
# ## ---------------------------------------- ## #

def plot_mesh(mesh: Mesh, cell_tags: Union[MeshTags, None] = None, **kwargs) -> pyvista.Plotter:
    """
    Plot the mesh with colored subdomains

    Args:
        mesh: a dolfinx mesh
        cell_tags: (optional) a dolfinx MeshTag instance

    Returns:
        The pyvista.Plotter

    Adapted from:
        https://jsdokken.com/dolfinx-tutorial/chapter3/em.html

    Example:
        .. highlight:: python
        .. code-block:: python

          from mpi4py import MPI
          from dolfinx.mesh import create_unit_square
          from elastodynamicsx.utils import make_tags

          domain = create_unit_square(MPI.COMM_WORLD, 10, 10)

          Omegas = [(1, lambda x: x[1] <= 0.5),
                    (2, lambda x: x[1] >= 0.5)]
          cell_tags = make_tags(domain, Omegas, 'domains')

          p = plot_mesh(domain, cell_tags=cell_tags)
          p.show()
    """
    p = pyvista.Plotter()
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(mesh, mesh.topology.dim))
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local

    if not (cell_tags is None):
        grid.cell_data["Marker"] = cell_tags.values[cell_tags.indices < num_local_cells]
        grid.set_active_scalars("Marker")

    p.add_mesh(grid, show_edges=True)

    if mesh.topology.dim == 2:
        p.view_xy()

    return p


def live_plotter(u: fem.Function, refresh_step: int = 1, **kwargs) -> pyvista.Plotter:
    """
    Convenience function to initialize a plotter for live-plotting within a
    time- or frequency-domain computation

    Args:
        u: The FEM function to be live-plotted
        refresh_step: Refresh each # step

    Keyword Args:
        kwargs: passed to *plotter*

    Example:
        Time-domain problems:

        .. highlight:: python
        .. code-block:: python

          tStepper = TimeStepper.build(...)
          u_res = tStepper.timescheme.u  # The solution
          p = live_plotter(u_res, refresh_step=10)
          tStepper.solve(..., live_plotter=p)

        Frequency-domain problems:

        .. highlight:: python
        .. code-block:: python

          fdsolver = FrequencyDomainSolver(...)
          u_res = fem.Function(V, name='solution')
          p = live_plotter(u_res)
          fdsolver.solve(omega=omegas,
                         out=u_res.x.petsc_vec,
                         callbacks=[],
                         live_plotter=p)
    """
    kwargs['refresh_step'] = refresh_step
    return plotter(u, **kwargs)


def plotter(*args: Union[List[fem.Function], Mesh], **kwargs) -> pyvista.Plotter:
    """
    A generic function to plot a mesh or one/several fields

    Args:
        *args: FEM functions to be plotted, sharing the same underlying function space

    Keyword Args:
        **kwargs: passed to either CustomScalarPlotter or CustomVectorPlotter

    Example:
        .. highlight:: python
        .. code-block:: python

          u1 = fem.Function(V)
          u2 = fem.Function(V)
          p = plotter(u1, u2, labels=['first', 'second'])
          p.show()
    """
    u1 = args[0]

    if isinstance(u1, Mesh):
        msh = u1
        assert len(args) < 3
        mt = args[1] if len(args) == 2 else kwargs.pop('cell_tags', None)
        assert isinstance(mt, MeshTags) or (mt is None)
        return plot_mesh(msh, mt, **kwargs)

    else:
        assert isinstance(u1, fem.Function)
        # test whether u is scalar or vector and returns the appropriate plotter
        nbcomps = u1.function_space.element.num_sub_elements  # number of components if vector space, 0 if scalar space

        if nbcomps == 0:
            return CustomScalarPlotter(*args, **kwargs)
        else:
            return CustomVectorPlotter(*args, **kwargs)


# ## -------------------------------------- ## #
# ## --- define useful plotting classes --- ## #
# ## -------------------------------------- ## #

class CustomScalarPlotter(pyvista.Plotter):
    """
    Not intended to be instanciated by user; better use the *plotter* function

    Args:
        u1, u2, ...: fem.Function; the underlying function space is a scalar function space

    Keyword Args:
        refresh_step: (default=1) The refresh step, if used for live-plotting
        sleep: (default=0.01) The sleep time after refresh, if used for live-plotting
        complex: (default='real') The way complex functions are plotted.
            Possible options: 'real', 'imag', 'abs', 'angle'
        labels: list of labels
        **kwargs: any valid kwarg for pyvista.Plotter and pyvista.Plotter.add_mesh
    """

    default_cmap = plt.cm.get_cmap("RdBu_r", 25)

    def __init__(self, *all_scalars, **kwargs):
        self.grids: List[pyvista.UnstructuredGrid] = []
        self._refresh_step: int = kwargs.pop('refresh_step', 1)
        self._tsleep: float = kwargs.pop('sleep', 0.01)
        dims = []

        self._trans = lambda x: x
        cmplx = kwargs.pop('complex', 'real')
        if cmplx.lower() == 'real':
            self._trans = np.real
        elif cmplx.lower() == 'imag':
            self._trans = np.imag
        elif cmplx.lower() == 'abs':
            self._trans = np.abs
        elif cmplx.lower() == 'angle':
            self._trans = np.angle

        for u_ in all_scalars:
            if u_ is None:
                break
            topology, cell_types, geom = plot.vtk_mesh(u_.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            grid.point_data["u"] = self._trans(u_.x.array)
            self.grids.append(grid)
            dims.append(u_.function_space.mesh.topology.dim)

        if len(self.grids) == 1:
            defaultShape = (1, 1)
        else:
            pyvista.global_theme.multi_rendering_splitting_position = 0.75
            defaultShape = '1|' + str(len(self.grids) - 1)

        kwinit = {'shape': kwargs.pop('shape', defaultShape)}
        for k in ('off_screen', 'notebook', 'border', 'border_color',
                  'window_size', 'multi_samples', 'line_smoothing',
                  'polygon_smoothing', 'lighting', 'theme'):
            if k in kwargs:
                kwinit[k] = kwargs.pop(k)
        super().__init__(**kwinit)

        show_edges = kwargs.pop('show_edges', 'first')
        if show_edges is False:
            show_edges = 'none'
        elif show_edges is True:
            show_edges = 'all'

        if show_edges == 'none':
            show_edges = [False for i in range(len(self.grids))]
        elif show_edges == 'all':
            show_edges = [True for i in range(len(self.grids))]
        elif show_edges == 'first':
            show_edges = [i == 0 for i in range(len(self.grids))]

        labels = kwargs.pop('labels', ['' for i in range(len(self.grids))])
        cmap = kwargs.pop('cmap', CustomScalarPlotter.default_cmap)
        sargs = dict(color="black", position_y=0.0)

        for i, (grid, dim) in enumerate(zip(self.grids, dims)):
            if len(self.grids) > 1:
                self.subplot(i)
            self.add_text(labels[i])
            self.add_mesh(grid, scalars='u', show_edges=show_edges[i],
                          lighting=False, cmap=cmap, scalar_bar_args=sargs, **kwargs)

            if dim == 2:
                self.view_xy()
                self.camera.zoom(1.2)

        if len(self.grids) > 1:
            # reset the focus to first subplot
            self.subplot(0)

    def update_scalars(self, *all_scalars, **kwargs):
        """Calls pyvista.Plotter.update_scalars for all subplots"""
        for i, (grid, u_) in enumerate(zip(self.grids, all_scalars)):
            grid['u'] = self._trans(u_)
        if kwargs.get('render', True):
            self.render()

    def live_plotter_start(self):
        is_recording = hasattr(self, 'mwriter')
        if is_recording:
            self.notebook = False
        self.show(interactive_update=True)

    def live_plotter_stop(self):
        is_recording = hasattr(self, 'mwriter')
        if is_recording:
            self.close()
            fname = self.mwriter.request.filename
            try:
                import IPython.display
                if self.mwriter.request.extension.lower() == '.gif':
                    im = IPython.display.Image(open(fname, 'rb').read())
                    IPython.display.display(im)  # display .gif in notebook
                elif self.mwriter.request.extension.lower() == '.mp4':
                    vi = IPython.display.Video(fname)
                    if _DOCS_CFG:
                        vi.embed = True
                        vi.html_attributes = "controls loop autoplay"
                    IPython.display.display(vi)  # display .mp4 in notebook
            except ModuleNotFoundError:
                pass

    def live_plotter_update_function(self, i: int, vec: PETSc.Vec) -> None:  # type: ignore[name-defined]
        if not isinstance(vec, PETSc.Vec):  # type: ignore[attr-defined]
            if issubclass(type(vec), fem.function.Function):
                vec = vec.x.petsc_vec
            else:
                try:
                    # assume vec is a TimeStepper instance
                    vec = vec.u.x.petsc_vec
                except:  # noqa
                    raise TypeError
        if (self._refresh_step > 0) and (i % self._refresh_step == 0):
            is_recording = hasattr(self, 'mwriter')
            with vec.localForm() as loc_v:  # Necessary for correct handling of ghosts in parallel
                self.update_scalars(loc_v.array, render=not is_recording)
            if is_recording:
                self.write_frame()  # This triggers a render
            else:
                time.sleep(self._tsleep)

    def add_time_browser(self,
                         update_fields_function: Callable,
                         timesteps: np.ndarray,
                         **kwargs_slider):
        self._time_browser_cbck = update_fields_function
        self._time_browser_times = timesteps

        def updateTStep(value):
            i = np.argmin(np.abs(value - self._time_browser_times))
            updated_fields = self._time_browser_cbck(i)
            self.update_scalars(*updated_fields)

        self.add_slider_widget(updateTStep, [timesteps[0], timesteps[-1]], **kwargs_slider)


class CustomVectorPlotter(pyvista.Plotter):
    """
    Args:
        u1, u2, ...: fem.Function; the underlying function space is a scalar function space

    Keyword Args:
        refresh_step: (default=1) The refresh step, if used for live-plotting
        sleep: (default=0.01) The sleep time after refresh, if used for live-plotting
        complex: (default='real') The way complex functions are plotted.
            Possible options: 'real', 'imag', 'abs', 'angle'
        labels: list of labels
        warp_factor: (default=1 or 0.5/max(clim) if provided) Factor for plotting mesh deformation
        ref_mesh: (default=False) Whether to show the undeformed mesh
        **kwargs: any valid kwarg for pyvista.Plotter and pyvista.Plotter.add_mesh
    """

    default_cmap = plt.cm.get_cmap("viridis")

    def __init__(self, *all_vectors, **kwargs):
        ###
        self.grids: List[pyvista.UnstructuredGrid] = []
        self._refresh_step: int = kwargs.pop('refresh_step', 1)
        self._tsleep: float = kwargs.pop('sleep', 0.01)
        dims = []

        self._trans = lambda x: x
        cmplx = kwargs.pop('complex', 'real')

        if cmplx.lower() == 'real':
            self._trans = np.real
        elif cmplx.lower() == 'imag':
            self._trans = np.imag
        elif cmplx.lower() == 'abs':
            self._trans = np.abs
        elif cmplx.lower() == 'angle':
            self._trans = np.angle

        if 'warp_factor' in kwargs:
            self.warp_factor = kwargs.pop('warp_factor')
        elif 'clim' in kwargs and np.amax(np.abs(kwargs['clim'])) > 0:
            self.warp_factor = 0.5 / np.amax(np.abs(kwargs['clim']))
        else:
            self.warp_factor = 1

        for u_ in all_vectors:
            if u_ is None:
                break
            topology, cell_types, geom = plot.vtk_mesh(u_.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            u3D = _get_3D_array_from_FEFunction(u_)
            grid["u"] = self._trans(u3D)
            grid.point_data["u_nrm"] = np.linalg.norm(u3D, axis=1)
            self.grids.append(grid)
            dims.append(u_.function_space.mesh.topology.dim)

        # nbcomps = max(1, u_.function_space.element.num_sub_elements)
        if len(self.grids) == 1:
            defaultShape = (1, 1)
        else:
            pyvista.global_theme.multi_rendering_splitting_position = 0.75
            defaultShape = '1|' + str(len(self.grids) - 1)

        kwinit = {'shape': kwargs.pop('shape', defaultShape)}
        for k in ('off_screen', 'notebook', 'border', 'border_color',
                  'window_size', 'multi_samples', 'line_smoothing',
                  'polygon_smoothing', 'lighting', 'theme'):
            if k in kwargs:
                kwinit[k] = kwargs.pop(k)

        super().__init__(**kwinit)

        show_edges = kwargs.pop('show_edges', 'first')
        if show_edges is False:
            show_edges = 'none'
        elif show_edges is True:
            show_edges = 'all'

        if show_edges == 'none':
            show_edges = [False for i in range(len(self.grids))]
        elif show_edges == 'all':
            show_edges = [True for i in range(len(self.grids))]
        elif show_edges == 'first':
            show_edges = [i == 0 for i in range(len(self.grids))]

        show_ref_mesh = kwargs.pop('ref_mesh', False)
        if (show_ref_mesh is True) and not ('opacity' in kwargs.keys()):
            kwargs['opacity'] = 0.8

        labels = kwargs.pop('labels', ['' for i in range(len(self.grids))])
        cmap = kwargs.pop('cmap', CustomVectorPlotter.default_cmap)
        sargs = dict(color="black", position_y=0.0)

        for i, grid in enumerate(self.grids):
            if len(self.grids) > 1:
                self.subplot(i)
            if show_ref_mesh:
                self.add_mesh(grid, style='wireframe', color='black')

            self.add_text(labels[i])
            warped = grid.warp_by_vector("u", factor=self.warp_factor)
            grid.warped = warped
            self.add_mesh(warped, scalars="u_nrm", show_edges=show_edges[i],
                          lighting=False, scalar_bar_args=sargs, cmap=cmap, **kwargs)

            if dims[i] == 2:
                self.view_xy()
                self.camera.zoom(1.2)

        if len(self.grids) > 1:
            self.subplot(0)  # resets the focus to first subplot

    def update_vectors(self, *all_vectors, render: bool = True) -> None:
        """
        Calls pyvista.Plotter.update_coordinates and .update_scalars for all subplots

        Args:
            all_vectors: tuple of np.ndarray,
                e.g. all_vectors = (u1.x, u2.x, ...) where u1, u2 are dolfinx.fem.Function
        """
        for i, (grid, u_) in enumerate(zip(self.grids, all_vectors)):
            nbpts = grid.number_of_points
            u3D = _get_3D_array_from_nparray(u_, nbpts)
            grid["u"] = self._trans(u3D)
            grid.warped.points = grid.warp_by_vector("u", factor=self.warp_factor).points
            grid.warped["u_nrm"] = np.linalg.norm(u3D, axis=1)
        if render:
            self.render()

    def _auto_record(self, fname=None, *args):
        is_recording = hasattr(self, 'mwriter')
        if is_recording:
            return
        if fname is None:
            fname = 'tmp.mp4'
        if fname.endswith('.gif'):
            self.open_gif(fname, *args)
        elif fname.endswith('.mp4'):
            self.open_movie(fname, *args)

    def live_plotter_start(self):
        if _DOCS_CFG:
            self._auto_record()
        is_recording = hasattr(self, 'mwriter')
        if is_recording:
            self.notebook = False
        self.show(interactive_update=True)

    def live_plotter_stop(self):
        is_recording = hasattr(self, 'mwriter')
        if is_recording:
            self.close()
            fname = self.mwriter.request.filename
            try:
                import IPython.display
                if self.mwriter.request.extension.lower() == '.gif':
                    im = IPython.display.Image(open(fname, 'rb').read())
                    IPython.display.display(im)  # display .gif in notebook
                elif self.mwriter.request.extension.lower() == '.mp4':
                    vi = IPython.display.Video(fname)
                    if _DOCS_CFG:
                        vi.embed = True
                        vi.html_attributes = "controls loop autoplay"
                    IPython.display.display(vi)  # display .mp4 in notebook
            except ModuleNotFoundError:
                pass

    def live_plotter_update_function(self, i: int, vec: PETSc.Vec) -> None:  # type: ignore[name-defined]
        if not isinstance(vec, PETSc.Vec):  # type: ignore[attr-defined]
            if issubclass(type(vec), fem.function.Function):
                vec = vec.x.petsc_vec
            else:
                try:
                    vec = vec.u.x.petsc_vec  # assume vec is a TimeStepper instance
                except:  # noqa
                    raise TypeError
        if (self._refresh_step > 0) and (i % self._refresh_step == 0):
            is_recording = hasattr(self, 'mwriter')
            with vec.localForm() as loc_v:  # Necessary for correct handling of ghosts in parallel
                self.update_vectors(loc_v.array, render=not is_recording)
            if is_recording:
                self.write_frame()  # This triggers a render
            else:
                time.sleep(self._tsleep)

    def add_time_browser(self,
                         update_fields_function: Callable,
                         timesteps: np.ndarray,
                         **kwargs_slider) -> None:
        self._time_browser_cbck = update_fields_function
        self._time_browser_times = timesteps

        def updateTStep(value):
            i = np.argmin(np.abs(value - self._time_browser_times))
            updated_fields = self._time_browser_cbck(i)
            self.update_vectors(*updated_fields)

        self.add_slider_widget(updateTStep, [timesteps[0], timesteps[-1]], **kwargs_slider)


def spy_petscMatrix(Z: PETSc.Mat, *args, **kwargs) -> AxesImage:  # type: ignore[name-defined]
    """
    matplotlib.pyplot.spy with Z being a petsc4py.PETSc.Mat object

    Args:
        Z:      The array to be plotted, of type petsc4py.PETSc.Mat
        args:   Passed to matplotlib.pyplot.spy (see doc)

    Keyword Args:
        kwargs: Passed to matplotlib.pyplot.spy (see doc)

    Returns:
        See doc of matplotlib.pyplot.spy
    """
    import scipy.sparse  # type: ignore
    kwargs['markersize'] = kwargs.get('markersize', 1)
    # [::-1] because:
    # ai, aj, av    = Z.getValuesCSR()
    # Z_scipysparse = scipy.sparse.csr_matrix((av, aj, ai))
    return plt.spy(scipy.sparse.csr_matrix(Z.getValuesCSR()[::-1]), *args, **kwargs)


# ## ------------------------------------ ## #
# ## --- define useful util functions --- ## #
# ## ------------------------------------ ## #

def _get_3D_array_from_FEFunction(u_):
    """Not intended to be called by user"""
    # u_ is a fem.Function
    nbcomps = max(1, u_.function_space.element.num_sub_elements)  # number of components
    nbpts = u_.x.array.size // nbcomps
    if nbcomps < 3:
        z0s = np.zeros((nbpts, 3 - nbcomps), dtype=u_.x.array.dtype)
        return np.append(u_.x.array.reshape((nbpts, nbcomps)), z0s, axis=1)
    else:
        return u_.x.array.reshape((nbpts, 3))


def _get_3D_array_from_nparray(u_, nbpts):
    """Not intended to be called by user"""
    # u_ is a np.array
    nbcomps = u_.size // nbpts
    if nbcomps < 3:
        z0s = np.zeros((nbpts, 3 - nbcomps), dtype=u_.dtype)
        return np.append(u_.reshape((nbpts, nbcomps)), z0s, axis=1)
    else:
        return u_.reshape((nbpts, 3))
