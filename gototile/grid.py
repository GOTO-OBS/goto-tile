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


PI2 = np.pi * 2
PI_2 = np.pi / 2
PI = np.pi
DEG = 180/PI
RAD = PI/180


def xyz2radec(x, y, z):
    l, b = xyz2lb(x, y, z)
    return np.array([l*DEG, b*DEG])


def lb2xyz(l, b):
    x = np.cos(l) * np.cos(b)
    y = np.sin(l) * np.cos(b)
    z = np.sin(b)
    return np.array([x, y, z])


def xyz2lb(x, y, z):
    # Using arctan2 for b instead of b = np.arcsin(z), allows for [x,
    # y, z] to be unnormalized
    b = np.arctan2(z, np.sqrt(y*y+x*x))
    l = np.arctan2(y, x)
    return np.array([l, b])


def cross(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.empty(x.shape)
    z[...,0] = x[...,1] * y[...,2] - x[...,2] * y[...,1]
    z[...,1] = x[...,2] * y[...,0] - x[...,0] * y[...,2]
    z[...,2] = x[...,0] * y[...,1] - x[...,1] * y[...,0]
    return z


def dot(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x[...,0] * y[...,0] + x[...,1] * y[...,1] + x[...,2] * y[...,2]


def intersect(x1, x2, y1, y2):
    """Returns the intersection of the two lines given by x1-x2 and y1-y2.

    All coordinates should be Cartesian coordinates, and be a (N, 3)
    or a (3,) numpy array, or a 3-element list.

    """
    p = cross(x1, x2)
    q = cross(y1, y2)
    t = cross(p, q)
    mask = dot(cross(p, x1), t) < 0
    sign = np.ones(np.asarray(x1).shape)
    sign[mask, ...] = -1
    t = t * sign
    return t


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
