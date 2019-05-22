from __future__ import absolute_import, division

import os.path
from collections import defaultdict
import numpy as np
import healpy
from astropy.coordinates import SkyCoord, AltAz
from astropy import units
from astropy.table import Table
import sys
import requests
from urllib.request import urlretrieve
import pkg_resources
import time
import pandas as pd
from . import settings

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

        print("Coverting .txt to .csv ...")
        col = ['PGC','GWGC name','HyperLEDA name',
                '2MASS name','SDSS-DR12 name','flag1',
                'ra','dec','Dist','Dist_err','z','B',
                'B_err','B_Abs','J','J_err','H','H_err',
                'K','K_err','flag2','flag3']
        outfile = os.path.join(local_path,'GLADE.csv')
        df = pd.DataFrame(np.genfromtxt(out_txt), columns=col)
        df = df[df.dist < cutoff_dist]
        df.to_csv(outfile, index=False)

        os.remove(out_txt)



def visible_catalog(catalog, sidtimes, telescope):

    mask = np.zeros(len(catalog), dtype=np.bool)
    coords = SkyCoord(ra=catalog['ra']*units.deg, dec=catalog['dec']*units.deg)
    for st in sidtimes:
        frame = AltAz(obstime=st, location=telescope.location)
        altaz = coords.transform_to(frame)
        mask |= altaz.alt > telescope.min_elevation
    return catalog[mask], np.where(mask)[0]


def read_catalog(path, GW_dist_info, key=None):
    table = Table.read(path)
    if key:
        table['weight'] = table[key]
    else:
        dist, dist_err = GW_dist_info[0], GW_dist_info[0]
        table['weight'] = np.exp(-(table['Dist'] - dist)**2/(2*dist_err**2))
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
