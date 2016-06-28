from __future__ import absolute_import, division

import os.path
from collections import defaultdict
import numpy as np
import healpy
from astropy.coordinates import SkyCoord, AltAz
from astropy import units
from astropy.table import Table


def visible_catalog(catalog, sidtimes, telescope):

    mask = np.zeros(len(catalog), dtype=np.bool)
    coords = SkyCoord(ra=catalog['ra']*units.deg, dec=catalog['dec']*units.deg)
    for st in sidtimes:
        frame = AltAz(obstime=st, location=telescope.location)
        altaz = coords.transform_to(frame)
        mask |= altaz.alt > telescope.min_elevation
    return catalog[mask], np.where(mask)[0]


def read_catalog(path, key=None):
    table = Table.read(path)
    if key:
        table['weight'] = table[key]
    else:
        table['weight'] = np.ones(len(table), dtype=np.float)
    return table


def map2catalog(skymap, catalog, key='weight'):
    """Return a copy of the skymap folded with the catalog"""
    weights = np.zeros(len(skymap.skymap))

    phi = np.deg2rad(catalog['ra']%360)
    theta = np.pi/2 - np.deg2rad(catalog['dec'])

    catalogpixels = healpy.ang2pix(skymap.nside, theta, phi, nest=skymap.isnested)
    sources = defaultdict(list)

    for i, weight in enumerate(catalog[key]):
        if weight:
            pixel = catalogpixels[i]
            weights[pixel] += weight
            # Store the catalog source
            sources[pixel].append((i, catalog['ra'][i], catalog['dec'][i]))
    weightmap = skymap.copy()
    weightmap.skymap = weights * weightmap.skymap

    return weightmap, sources
