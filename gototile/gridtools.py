from __future__ import division

try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools as it
import gzip
import os
import math
import tempfile
import logging
import multiprocessing
import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable

from .math import lb2xyz, xyz2lb, intersect
from .math import RAD, PI, PI_2


def create_grid(fov, overlap, kind):
    """Create grid coordinates.

    Calculate strips along RA and stacked in declination to cover the full sky.

    The step size in Right Ascension is adjusted with the declination,
    by a factor of 1/cos(declination).

    Parameters
    ----------
    fov : dict of int or float or `astropy.units.Quantity`
        The field of view of the tiles in the RA and Dec directions.
        It should contains the keys 'ra' and 'dec'.
        If not given units the values are assumed to be in degrees.

    overlap : dict of int or float
        The overlap amount between the tiles in the RA and Dec directions.
        It should contains the keys 'ra' and 'dec'.

    kind : str
        The tiling method to use. Options are:
        - 'cosine':
                Newer algorithm which adjusts RA spacing based on dec.
        - 'cosine_symmetric':
                An alternate version of 'cosine' which rotates each dec stripe
                to be symmetric about the meridian.
        - 'product':
                Old, legacy algorithm.
                This method creates lots of overlap between tiles at high decs,
                which makes it impractical for survey purposes.
    """
    fov = fov.copy()
    overlap = overlap.copy()
    for key in ('ra', 'dec'):
        # Get value of foc
        if isinstance(fov[key], u.Quantity):
            fov[key] = fov[key].to('deg').value

        # Limit overlap to between 0 and 0.9
        overlap[key] = min(max(overlap[key], 0), 0.9)

    if kind == 'cosine':
        return create_grid_cosine(fov, overlap)
    elif kind == 'cosine_symmetric':
        return create_grid_cosine_symmetric(fov, overlap)
    elif kind == 'product':
        return create_grid_product(fov, overlap)
    elif kind == 'minverlap':
        return create_grid_minverlap(fov, overlap)
    elif kind == 'minverlap_enhanced':
        return create_grid_minverlap_enhanced(fov, overlap)
    else:
        raise ValueError('Unknown grid tiling method: "{}"'.format(kind))


def create_grid_product(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method uses the product of RA and Dec to get the RA spacings.
    """
    # Calculate steps
    step_dec = fov['dec'] * (1 - overlap['dec'])
    step_ra = fov['ra'] * (1 - overlap['ra'])

    # Create the dec strips
    pole = 90 // step_dec * step_dec
    decs = np.arange(-pole, pole+step_dec/2, step_dec)

    # Arrange the tiles in RA
    ras = np.arange(0.0, 360., step_ra)
    allras, alldecs = zip(*[(ra, dec) for ra, dec in it.product(ras, decs)])
    allras, alldecs = np.asarray(allras), np.asarray(alldecs)

    return allras, alldecs


def create_grid_cosine(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method adjusts the RA spacings based on the cos of the declination.
    """
    # Calculate steps
    step_dec = fov['dec'] * (1 - overlap['dec'])
    step_ra = fov['ra'] * (1 - overlap['ra'])

    # Create the dec strips
    pole = 90 // step_dec * step_dec
    decs = np.arange(-pole, pole+step_dec/2, step_dec)

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360., step_ra/np.cos(dec*RAD))
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_cosine_symmetric(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method adjusts the RA spacings based on the cos of the declination.

    Compared to `create_grid_cosine` this method rotates the dec strips so
    they are symmetric around the meridian.
    """
    # Calculate steps
    step_dec = fov['dec'] * (1 - overlap['dec'])
    step_ra = fov['ra'] * (1 - overlap['ra'])

    # Create the dec strips
    pole = 90 // step_dec * step_dec
    decs = np.arange(-pole, pole+step_dec/2, step_dec)

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360., step_ra/np.cos(dec*RAD))
        ras += (360-ras[-1])/2  # Rotate the strips so they're symmetric
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_minverlap(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method takes the overlaps given as the minimum rather than fixed,
    and then adjusts the number of tiles in RA and Dec until they overlap
    at least by the amount given.
    """
    # Create the dec strips
    pole = 90
    n_tiles = math.ceil(pole/((1-overlap['dec'])*fov['dec']))
    step_dec = pole/n_tiles
    north_decs = np.arange(pole, 0, step_dec * -1)
    south_decs = north_decs * -1
    decs = np.concatenate([south_decs, np.array([0]), north_decs[::-1]])

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        n_tiles = math.ceil(360/((1-overlap['ra'])*fov['ra']/np.cos(dec*RAD)))
        step_ra = 360/n_tiles
        ras = np.arange(0, 360, step_ra)
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_minverlap_enhanced(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method takes the overlaps given as the minimum rather than fixed,
    and then adjusts the number of tiles in RA and Dec until they overlap
    at least by the amount given.

    This is the second version of the minverlap algorithm.
    In this version the tiles are placed slightly closer to close some of
    the gaps in RA.
    Instead of aligning the tiles based on the declination of the centre
    the declination of the lower (in the north) / upper (in the south) courners
    is used.
    This has the effect of overlapping the courners rather than the centre of
    the sides of adjacent tiles, thereby reducing the gaps between the tiles.
    """
    # Create the dec strips
    pole = 90
    n_tiles = math.ceil(pole/((1-overlap['dec'])*fov['dec'])) + 1  # Bodge
    step_dec = pole/n_tiles
    north_decs = np.arange(pole, 0, step_dec * -1)
    south_decs = north_decs * -1
    decs = np.concatenate([south_decs, np.array([0]), north_decs[::-1]])

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        if 90 > abs(dec) > 0:
            dec2 = abs(dec) - fov['dec']/2
            dec3 = 90 - np.sqrt((90-dec2)**2 + (fov['ra']/2)**2)
        else:
            dec2 = dec
            dec3 = dec
        n_tiles = math.ceil(360/((1-overlap['ra'])*fov['ra']/np.cos(dec3*RAD)))
        step_ra = 360/n_tiles
        ras = np.arange(0, 360, step_ra)
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


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
