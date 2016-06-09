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
