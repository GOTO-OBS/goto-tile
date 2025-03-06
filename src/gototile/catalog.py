"""Catalog functions for gototile."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import astropy.units as u
import healpy as hp
import numpy as np
import pandas as pd
import pkg_resources
from astropy.coordinates import Angle, SkyCoord

from . import skymaptools
from .skymap import SkyMap


def download_glade(
    url: str = 'http://glade.elte.hu/GLADE_2.3.txt',
    local_path: str | Path | None = None,
    cutoff_dist: int = 10000,
) -> None:
    """Download the GLADE galaxy catalog and convert it to CSV format."""

    def reporthook(count: int, block_size: int, total_size: int) -> None:
        """Report download progress."""
        global start_time  # noqa: PLW0603
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            f'\r...{percent}%, {progress_size / (1024 * 1024):.2f} MB, '
            f'{speed} KB/s, {duration:.2f} seconds passed'
            % (percent, progress_size / (1024 * 1024), speed, duration),
        )
        sys.stdout.flush()

    print('Downloading GLADE galaxy catalog ...')
    if local_path is None:
        local_path = Path(pkg_resources.resource_filename('gototile', 'data'))
        if not local_path.exists():
            local_path.mkdir(parents=True)

    out_txt = local_path / 'GLADE.txt'
    if not url.startswith(('http:', 'https:')):
        raise ValueError("URL must start with 'http:' or 'https:'")
    urlretrieve(url, out_txt, reporthook)  # noqa: S310

    print('\nConverting .txt to .csv ...')
    col = [
        'PGC',
        'GWGC name',
        'HyperLEDA name',
        '2MASS name',
        'SDSS-DR12 name',
        'flag1',
        'ra',
        'dec',
        'Dist',
        'Dist_err',
        'z',
        'B',
        'B_err',
        'B_Abs',
        'J',
        'J_err',
        'H',
        'H_err',
        'K',
        'K_err',
        'flag2',
        'flag3',
    ]

    catalog_df = pd.read_csv(out_txt, sep=' ', header=None)
    catalog_df.columns = col
    catalog_df = catalog_df[(catalog_df.Dist < cutoff_dist) & (catalog_df.flag1 == 'G')]

    outfile = local_path / 'GLADE.csv'
    catalog_df.to_csv(outfile, index=False)

    out_txt.unlink()


def create_catalog_skymap(  # noqa: PLR0913
    name: str,
    dist_mean: float | None = None,
    dist_err: float | None = None,
    key: str = 'weight',
    nside: int = 64,
    nest: bool = True,
    smooth: bool = True,
    sigma: float = 15,
    min_weight: int = 0,
) -> SkyMap:
    """Create a skymap of weighted galaxy positions from a given catalog.

    Parameters
    ----------
    name : str
        name of the catalog to use
        options now are 'GWGC' or 'GLADE'
        if 'GLADE' the catalog will need to be downloaded the first time, as it's a big file

    dist_mean : float, optional
        mean signal distance, used to weight the catalog sources based on distance
        if given, dist_err should also be given

    dist_err : float, optional
        error on the signal distance, used to weight the catalog sources based on distance
        if given, dist_mean should also be given

    key : str, optional
        table key to use to weight the catalog
        default is 'weight'
        if dist_mean and dist_err are given then the weighting will use the 'Dist' key by default

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
    data_path = Path(pkg_resources.resource_filename('gototile', 'data'))
    filename = name + '.csv'
    filepath = data_path / filename
    if not filepath.is_file():
        if name == 'GLADE':
            # Can download the GLADE catalog if it doesn't exist
            download_glade()
        else:
            raise ValueError(f'Catalog name {name} not recognized')

    # Read the catalog
    table = pd.read_csv(filepath)

    # Calculate the weight for each galaxy based on its reported distance
    if dist_mean and dist_err:
        table['weighted_distance'] = np.exp(-((table['Dist'] - dist_mean) ** 2) / (2 * dist_err**2))
        key = 'weighted_distance'

    # Get ra,dec coords of the entries in the table
    ra, dec = table['ra'].to_numpy(), table['dec'].to_numpy()
    coord = SkyCoord(ra, dec, unit='deg', frame='fk5')

    # Convert coordinates into HEALPix pixels
    # NB We need to use nest=False because hp.smoothing expects RING order
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

    # Create a SkyMap class (remember it has order=RING)
    skymap = SkyMap.from_data(weights, order='RING', coordsys='C')

    # If we were asked for a nested skymap then regrade before returning
    if nest is True:
        skymap.regrade(order='NESTED')

    return skymap
