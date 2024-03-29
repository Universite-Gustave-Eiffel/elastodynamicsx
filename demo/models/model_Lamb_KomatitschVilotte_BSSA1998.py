# Copyright (C) 2023 Pierric Mora
#
# This file is part of ElastodynamiCSx
#
# SPDX-License-Identifier: MIT

### ### ###  ###
# GMSH MODEL ###
### ### ###  ###
import math
import gmsh
import warnings


def create_model(**kwargs):
    """
    kwargs:
        --- geometry ---
        theta: Tilt angle of the top boundary, in degrees
            (default=10)
        sizefactor: Multiply all dimensions and number of elements by this factor
            e.g.: sizefactor=0.5 for a half-scale model
            e.g.: sizefactor=1   for a full-scale model
            (default=1)

        --- tags ---
        tagBdFree: The tag of the free boundary
            (default=1)
        tagBdInt: The tag of the interior (artificial) boundaries
            (default=2)
        tagBulk: The tag of the interior domain
            (default=1)

        --- cell & mesh ---
        structured: True for a structured mesh, False for unstructured
            (default=True)
        cell_type: 'quadrangle' or 'triangle'
            (default='quadrangle')
        degree: The number of nodes per cell and dimension (polynomial order)
            (default=1)

        --- misc ---
        export: Whether to export the mesh to a meshfile
            (default=False)
        show: Plot using the GMSH interface
            (default=False)
    """

    gmsh.initialize()
    gmsh.model.add("Lamb_KomatitschVilotte_BSSA1998")

    # CUSTOM OPTIONS
    meshfile = kwargs.get('export', False)
    show     = kwargs.get('show', False)

    # PHYSICAL SIZE
    sz    = kwargs.get('sizefactor', 1)
    theta = kwargs.get('tilt', 10)  # tilt angle (degrees) of the top boundary
    theta = theta * math.pi/180
    W_ = 4 * sz                     # width
    HL = 2 * sz                     # height (left)
    HR = HL + W_*math.tan(theta)    # height (right)

    # NUMBER OF POINTS
    gdim   = 2  # dimension
    Nx, Ny = int(50 * sz), int(30 * sz)

    # CORNERS
    pt1 = gmsh.model.geo.addPoint(0,  0,  0)
    pt2 = gmsh.model.geo.addPoint(0,  HL, 0)
    pt3 = gmsh.model.geo.addPoint(W_, HR, 0)
    pt4 = gmsh.model.geo.addPoint(W_, 0,  0)

    # BOUNDARIES
    bdl = gmsh.model.geo.addLine(1, 2, pt1)
    bdt = gmsh.model.geo.addLine(2, 3, pt2)
    bdr = gmsh.model.geo.addLine(3, 4, pt3)
    bdb = gmsh.model.geo.addLine(4, 1, pt4)

    # SURFACE
    cl  = gmsh.model.geo.addCurveLoop([bdl, bdt, bdr, bdb])
    srf = gmsh.model.geo.addPlaneSurface([cl], 1)

    # CELLS
    cell_type = kwargs.get('cell_type', 'quadrangle')
    gmsh.model.geo.mesh.setTransfiniteCurve(bdb, Nx)    # prescribe the number of elts
    gmsh.model.geo.mesh.setTransfiniteCurve(bdt, Nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(bdl, Ny)
    gmsh.model.geo.mesh.setTransfiniteCurve(bdr, Ny)
    if kwargs.get('structured', True):
        gmsh.model.geo.mesh.setTransfiniteSurface(srf)  # structured mesh
    if cell_type.lower() == 'quadrangle':
        gmsh.model.geo.mesh.setRecombine(gdim, srf)     # quadrangles instead of triangles
    else:
        assert cell_type.lower() == 'triangle', "'cell_type' should be either 'quadrangle' or 'triangle'"

    # GROUPS
    tagBdFree = kwargs.get('tagBdFree', 1)  # top boundary: stress free
    tagBdInt  = kwargs.get('tagBdInt',  2)  # artificial boundaries
    tagBulk   = kwargs.get('tagBulk',   1)  # solid
    gmsh.model.addPhysicalGroup(gdim-1, [bdt],           tag=tagBdFree)
    gmsh.model.addPhysicalGroup(gdim-1, [bdr, bdb, bdl], tag=tagBdInt)
    gmsh.model.addPhysicalGroup(gdim,   [srf],           tag=tagBulk)

    # MESH
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(gdim)

    # NODES
    # WARNING: dolfinx.gmshio (v0.6.0) works for deg = 1,2,3 but does not support higher order elts
    deg = kwargs.get('degree', 1)
    if deg > 3:
        deg = 3
        warnings.warn("'degree' set to 3: dolfinx.gmshio (v0.6.0) works for deg = 1,2,3 but does not support higher order elts")
    gmsh.model.mesh.setOrder(deg)

    # EXPORT MESHFILE
    if meshfile:
        gmsh.write("Lamb_KomatitschVilotte_BSSA1998.msh")

    # SHOW
    if show:
        gmsh.fltk.run()

    # gmsh.finalize()
    ### ### ### ###
    #  THE END  ###
    ### ### ### ###
    return gmsh.model


if __name__ == "__main__":
    create_model(show=False)

