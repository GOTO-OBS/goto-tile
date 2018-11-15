from __future__ import division
try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools as it
import gzip
import os
import tempfile
import logging
import multiprocessing
import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
from astropy import units as u

from . import skymaptools as smt
from .math import lb2xyz, xyz2lb, intersect
from .math import RAD, PI, PI_2


def get_tile_vertices(coords, delew, delns):
    ra = coords.ra.value
    dec = coords.dec.value

    phiew = delew/2*RAD
    phins = delns/2*RAD

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


def create_allsky_strips(rastep, decstep):
    """Calculate strips along RA and stacked in declination to cover the
    full sky

    The step size in Right Ascension is adjusted with the declination,
    by a factor of 1/cos(declination).

    Parameters
    ----------

    rastep : float
        Step size in Right Ascension, uncorrected for declination.
    decstep : float
        Step size in declination.

    """

    pole = 90 // decstep * decstep
    decs = np.arange(-pole, pole+decstep/2, decstep)
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360., rastep/np.cos(dec*RAD))
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


class PolygonQuery(object):
    def __init__(self, nside, nested):
        self.nside = nside
        self.nested = nested
    def __call__(self, vertices):
        return hp.query_polygon(self.nside, vertices, nest=self.nested)


class SkyGrid(object):
    """An all-sky grid of defined tiles.

    Parameters
    ----------
    fov : list or tuple or dict of int or float or `astropy.units.Quantity`
        The field of view of the tiles in the RA and Dec directions.
        If given as a tuple, the arguments are assumed to be (ra, dec).
        If given as a dict, it should contains the keys 'ra' and 'dec'.
        If not given units the values are assumed to be in degrees.

    overlap : int or float or list or tuple or dict of int or float, optional
        The overlap amount between the tiles in the RA and Dec directions.
        If given a single value, assumed to be the same overlap in both RA and Dec.
        If given as a tuple, the arguments are assumed to be (ra, dec).
        If given as a dict, it should contains the keys 'ra' and 'dec'.
        default is 0.5 in both axes, minimum is 0 and maximum is 0.9

    nside : int, optional
        default is 64

    nested : bool, optional
        default is True

    """

    def __init__(self, fov, overlap=None, nside=64, nested=True):
        # Parse fov
        if isinstance(fov, (list,tuple)):
            fov = {'ra': fov[0], 'dec': fov[1]}
        for key in ('ra', 'dec'):
            # make sure fov is in degrees
            if not isinstance(fov[key], u.Quantity):
                fov[key] *= u.deg
        self.fov = fov

        # Parse overlap
        if overlap is None:
            overlap = {'ra': 0.5, 'dec': 0.5}
        elif isinstance(overlap, (int, float, u.Quantity)):
            overlap = {'ra': overlap, 'dec': overlap}
        elif isinstance(overlap, (list,tuple)):
            overlap = {'ra': overlap[0], 'dec': overlap[1]}
        for key in ('ra', 'dec'):
            # limit overlap to between 0 and 0.9
            overlap[key] = min(max(overlap[key], 0), 0.9)
        self.overlap = overlap

        # Calculate step sizes
        step = {}
        for key in ('ra', 'dec'):
            step[key] = fov[key].value * (1 - overlap[key])
        self.step = step

        # Other params
        self.nside = nside
        self.isnested = nested

        # Create the grid
        ras, decs = create_allsky_strips(self.step['ra'], self.step['dec'])
        self.coords = SkyCoord(ras, decs, unit=u.deg)
        self.ntiles = len(self.coords)

        # Get the tile vertices
        self.vertices = get_tile_vertices(self.coords,
                                          self.fov['ra'].value,
                                          self.fov['dec'].value)

        # Calculate the HEALPix indicies within each tile
        # This is the complicated bit, so it's done over multiple processes
        polygon_query = PolygonQuery(self.nside, self.isnested)
        pool = multiprocessing.Pool()
        pixels = pool.map(polygon_query, self.vertices)
        pool.close()
        pool.join()
        self.pixels = np.array(pixels)

        # Give the tiles unique ids
        self.tilenames = np.arange(self.ntiles) + 1

    def copy(self):
        """Return a new instance containing a copy of the sky grid data."""
        newgrid = SkyGrid(self.fov, self.overlap, self.nside, self.isnested)
        return newgrid

    def regrade(self, nside, nested=True):
        """Up- or downgrade the sky grid HEALPix resolution.

        See the `healpy.pixelfunc.ud_grade()` documentation for the parameters.
        """
        if nside == self.nside and nested == self.isnested:
            return

        polygon_query = PolygonQuery(nside, nested)
        pool = multiprocessing.Pool()
        pixels = pool.map(polygon_query, self.vertices)
        pool.close()
        pool.join()
        self.pixels = np.array(pixels)
        self.nside = nside
        self.isnested = nested

    def apply_skymap(self, skymap):
        """Apply a SkyMap to the grid.

        This means caculate the contained probabiltiy within each tile.

        Parameters
        ----------
        skymap : `gototile.skymap.SkyMap`
            The sky map to map onto this grid.
        """
        if self.nside != skymap.nside:
            # Need to regrade so they match
            # Best option is to match grid precision to the skymap
            self.regrade(skymap.nside)
        self.probs = np.array([skymap.skymap[pix].sum() for pix in self.pixels])
