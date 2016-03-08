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


def makegrid(tilesdir, nside, name=None):
    if not os.path.exists(tilesdir):
        os.makedirs(tilesdir)
    if name:
        fp = tempfile.NamedTemporaryFile(prefix='temp__', dir='.', delete=False)
        path = fp.name
        fp.close()
        tileallsky(path, name, nside)
        return path
    for scope in (4, 8):
        scopename = "GOTO{}".format(scope)
        filename = "{}_nside{}_nestTrue.pgz".format(scopename, nside)
        path = os.path.join(tilesdir, filename)
        tileallsky(path, scopename, nside)
    scopename = "SuperWASP-N"
    filename = "{}_nside{}_nested.pgz".format(scopename, nside)
    path = os.path.join(tilesdir, filename)
    tileallsky(path, scopename, nside)

    scopename = "VISTA"
    filename = "{}_nside{}_nested.pgz".format(scopename, nside)
    path = os.path.join(tilesdir, filename)
    tileallsky(path, scopename, nside)
    return path


def tileallsky(filename, scopename, nside):

    delns, delew = sct.getscopeinfo(scopename)[:2]

    north = np.arange(0.0, 90.0, delns)
    south = -1*north
    n2s = np.append(south[::-2], north)
    e2w = np.arange(0.0, 360., delew)

    tilelist = np.array([smt.findFoV(ra, dec, delns, delew)
                         for dec, ra in it.product(n2s, e2w)])

    pointlist = [smt.getvectors(tile)[0] for tile in tilelist]
    pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=True)
                        for points in pointlist])

    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump([tilelist, pixlist], f) #makes gzip compressed pickles


def tileallsky_new(filename, fov, nside):
    """Create a grid across all sky and store in a file"""
    delns = fov['dec'].decompose(bases=[units.degree]).value / 2
    delew = fov['ra'].decompose(bases=[units.degree]).value / 2

    north = np.arange(0.0, 90.0, delns)
    south = -north[:]
    n2s = np.append(south[::-2], north)
    e2w = np.arange(0.0, 360., delew)

    ras, decs = zip(*[(ra, dec) for ra, dec in it.product(e2w, n2s)])
    gridcoords = SkyCoord(np.asarray(ras), np.asarray(decs), unit=units.deg)
    tilelist = [smt.find_tile(ra, dec, delns, delew)[0]
                for ra, dec in zip(ras, decs)]

    pointlist = [smt.getvectors2(tile)[0] for tile in tilelist]
    pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=True)
                        for points in pointlist])
    writetiles_new(filename, tilelist, pixlist, gridcoords)


def readtiles(filename):
    with gzip.open(filename, 'r') as f:
        tilelist, pixlist = pickle.load(f)
    return tilelist, pixlist


def writetiles(filename, tilelist, pixlist):
    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump([tilelist, pixlist], f)


def writetiles_new(filename, tilelist, pixlist, centers):
    with gzip.GzipFile(filename, 'w') as fp:
        pickle.dump([tilelist, pixlist, centers], fp, protocol=2)
