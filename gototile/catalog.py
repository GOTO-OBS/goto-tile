from __future__ import absolute_import, division

import os.path
from collections import defaultdict
import numpy as np
import healpy
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy.table import Table
import sys
import requests
from urllib.request import urlretrieve
import pkg_resources
import time
import pandas as pd
import healpy as hp

from . import settings
from . import skymaptools
from .skymap import SkyMap


class download():

    @staticmethod
    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    @staticmethod
    def glade(url="http://glade.elte.hu/GLADE_2.3.txt", local_path=None, cutoff_dist=10000):
        print("Downloading GLADE galaxy catalog ...")
        if local_path==None:
            local_path = pkg_resources.resource_filename('gototile', 'data')
            if not os.path.exists(local_path):
                os.makedirs(local_path)

        out_txt = os.path.join(local_path,'GLADE.txt')
        urlretrieve(url, out_txt, download.reporthook)

        print("\nCoverting .txt to .csv ...")
        col = ['PGC','GWGC name','HyperLEDA name',
                '2MASS name','SDSS-DR12 name','flag1',
                'ra','dec','Dist','Dist_err','z','B',
                'B_err','B_Abs','J','J_err','H','H_err',
                'K','K_err','flag2','flag3']

        df = pd.read_csv(out_txt, sep=" ", header=None)
        df.columns = col
        df = df[(df.Dist < cutoff_dist) & (df.flag1=='G')]

        outfile = os.path.join(local_path,'GLADE.csv')
        df.to_csv(outfile, index=False)

        os.remove(out_txt)


def visible_catalog(catalog, sidtimes, telescope):

    mask = np.zeros(len(catalog), dtype=np.bool)
    coords = SkyCoord(ra=catalog['ra']*u.deg, dec=catalog['dec']*u.deg)
    for st in sidtimes:
        frame = AltAz(obstime=st, location=telescope.location)
        altaz = coords.transform_to(frame)
        mask |= altaz.alt > telescope.min_elevation
    return catalog[mask], np.where(mask)[0]


def read_catalog(path):
    """Read a catalog and return a Pandas dataframe."""
    table = pd.read_csv(path)
    return table


def catalog2skymap(name, key='weight', dist_mean=None, dist_err=None,
                   nside=64, nest=True, smooth=True, sigma=15, min_weight=0):
    """Create a skymap of weighted galaxy positions from a given catalog.

    Parameters
    ----------
    name : str
        name of the catalog to use
        options now are 'GWGC' or 'GLADE'
        if 'GLADE' the catalog will need to be downloaded the first time, as it's a big file

    key : str, optional
        table key to use to weight the catalog
        default is 'weight'
        if dist_mean and dist_err are given then the weighting will use the 'Dist' key by default

    dist_mean : float, optional
        mean signal distance, used to weight the catalog sources based on distance
        if given, dist_err should also be given

    dist_err : float, optional
        error on the signal distance, used to weight the catalog sources based on distance
        if given, dist_mean should also be given

    nside : int, optional
        HEALPix Nside parameter for the resulting skymap
        default is 64

    nest : bool, optional
        if True, the resulting skymap will use the HEALPix NESTED order
        otherwise use the RING order
        default is True

    smooth : bool, optional
        if True, smooth the skymap using the `sigma` parameter
        default is True

    sigma : float, optional
        the gaussian sigma used when smoothing the skymap, in arcseconds
        default is 15 arcsec

    min_weight : float, optional
        minimum weight to scale the skymap
        default is 0

    Returns
    -------
    skymap : `gototile.skymap.SkyMap`
        the data in a SkyMap class

    """
    # Find the catalog path
    data_path = pkg_resources.resource_filename('gototile', 'data')
    filename = name + '.csv'
    if os.path.isfile(os.path.join(data_path, filename)):
        # The catalog already exists
        filepath = os.path.join(data_path, filename)
    else:
        if name == 'GLADE':
            # Can download the GLADE catalog if it doesn't exist
            download.glade()
            filepath = os.path.join(data_path, filename)
        else:
            raise ValueError('Catalog name not recognized')

    # Read the catalog
    table = read_catalog(filepath)

    # Calculate the weight for each galaxy based on its reported distance
    if dist_mean and dist_err:
        table['weighted_distance'] = np.exp(-(table['Dist'] - dist_mean)**2/(2*dist_err**2))
        key = 'weighted_distance'

    # Get ra,dec coords of the entries in the table
    ra, dec = table['ra'].values, table['dec'].values
    coord = SkyCoord(ra, dec, unit='deg', frame='fk5')

    # Convert coordinates into HEALPix pixels
    ipix = skymaptools.coord2pix(nside, coord, nest=False)

    # Create skymap weight array by summing weights of each pixel
    # Note there may be multiple galaxies within each HEALPix pixel,
    # np.add.at takes care of that
    npix = hp.nside2npix(nside)
    weights = np.zeros(npix)
    np.add.at(weights, ipix, table[key].values)

    if smooth:
        # Smooth the skymap by the given sigma
        sigma = Angle(sigma, unit=u.arcsec).degree
        weights = hp.smoothing(weights, sigma=np.deg2rad(sigma), verbose=False)

    # Scale weight between `min_weight` and 1
    weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    weights = (1 - min_weight) * weights + min_weight

    # Create a SkyMap class
    skymap = SkyMap.from_data(weights, nested=False, coordsys='C')
    if nest is True:
        skymap.regrade(order='NESTED')

    return skymap


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
