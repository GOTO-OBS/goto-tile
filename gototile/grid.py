from __future__ import division
try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools as it
import gzip
import os
import tempfile
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units
from spherical_geometry.vector import vector_to_radec
from . import skymaptools as smt
from . import scopetools as sct
from .math import lb2xyz, xyz2lb, intersect
from .math import RAD, PI, PI_2


def get_tile_vertices(ra, dec, delew, delns):
    phiew = delew*RAD
    phins = delns*RAD

    l, b = ra*RAD, dec*RAD
    xyz = lb2xyz(l, b)

    poles = {}
    poles['w'] = l - PI_2, 0 * b
    poles['e'] = l + PI_2, 0 * b
    mask = b < 0
    poles['s'] = l + 0, b-PI_2
    poles['s'][0][mask], poles['s'][1][mask] = l[mask]+PI, -b[mask]-PI_2
    poles['n'] = l+PI, PI_2-b
    poles['n'][0][mask], poles['n'][1][mask] = l[mask], b[mask]+PI_2
    poles['w'] = lb2xyz(*poles['w'])
    poles['e'] = lb2xyz(*poles['e'])
    poles['n'] = lb2xyz(*poles['n'])
    poles['s'] = lb2xyz(*poles['s'])

    edges = {}
    fcos, fsin = np.cos(phiew), np.sin(phiew)
    edges['e'] = xyz * fcos + poles['e'] * fsin
    le, be = xyz2lb(*edges['e'])
    edges['w'] = xyz * fcos + poles['w'] * fsin
    lw, bw = xyz2lb(*edges['w'])
    ls, bs = l, b-phins
    edges['s'] = lb2xyz(ls, bs)
    ln, bn = l, b+phins
    edges['n'] = lb2xyz(ln, bn)

    for key in edges.keys():
        edges[key] = edges[key].T
    for key in poles.keys():
        poles[key] = poles[key].T

    corners = []
    corners.append(intersect(edges['n'], poles['w'], edges['w'], poles['n']))
    corners.append(intersect(edges['n'], poles['e'], edges['e'], poles['n']))
    corners.append(intersect(edges['s'], poles['e'], edges['e'], poles['s']))
    corners.append(intersect(edges['s'], poles['w'], edges['w'], poles['s']))

    corners = np.asarray(corners)
    corners = np.rollaxis(corners, 0, 2)
    return corners


def tileallsky(filename, fov, nside):
    """Create a grid across all sky and store in a file"""
    delra = fov['ra'].decompose(bases=[units.degree]).value / 2
    deldec = fov['dec'].decompose(bases=[units.degree]).value / 2
    north = np.arange(0.0, 90.0, deldec)
    south = -north[:0:-1]
    n2s = np.append(south, north)
    e2w = np.arange(0.0, 360., delra)

    ras, decs = zip(*[(ra, dec) for ra, dec in it.product(e2w, n2s)])
    gridcoords = SkyCoord(np.asarray(ras), np.asarray(decs), unit=units.deg)
    tilelist = [smt.find_tile(ra, dec, deldec, delra)[0]
                for ra, dec in zip(ras, decs)]

    pointlist = [smt.getvectors(tile)[0] for tile in tilelist]
    pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=True)
                        for points in pointlist])

    with gzip.GzipFile(filename, 'w') as fp:
        pickle.dump([tilelist, pixlist, gridcoords], fp, protocol=2)



def tileallsky2(fov, nside, overlap=None, nested=True):
    """Create a grid across all sky and store in a file"""
    if overlap is None:
        overlap = {'ra': 0.5, 'dec': 0.5}
    step = {}
    for key in ('ra', 'dec'):
        overlap[key] = min(max(overlap[key], 0), 0.9)
        step[key] = fov[key] * (1-overlap[key])
    pole = 90 // step['dec'] * step['dec']
    n2s = np.arange(-pole, pole+step['dec']/2, step['dec'])
    e2w = np.arange(0.0, 360., step['ra'])

    ras, decs = zip(*[(ra, dec) for ra, dec in it.product(e2w, n2s)])
    ras, decs = np.asarray(ras), np.asarray(decs)
    gridcoords = SkyCoord(ras, decs, unit=units.deg)
    tilelist = get_tile_vertices(ras, decs, step['ra'], step['dec'])
    pixlist = np.array([hp.query_polygon(nside, vertices, nest=nested)
                        for vertices in tilelist])
    return gridcoords, tilelist, pixlist
