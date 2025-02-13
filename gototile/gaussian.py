import numpy as np
from astropy.table import Table
import healpy as hp
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord

from .skymaptools import pix2coord


def gaussian_prob(grid, peak, radius):
    """Calculate the probability of points gaussian dist.)

    Parameters
    ----------
    grid : `astropy.coordinates.SkyCoord`
        grid coordinates to calculate the probability at
    peak : scalar `astropy.coordinates.SkyCoord`
        central peak of the distribution
    radius : float
        68% containment radius, in degrees
    """
    # for each point on the grid calculate the angular distance to the peak
    dist = peak.separation(grid)
    dist = dist.degree

    # calculate the probability at each point with a 2D gaussian function
    sigma = radius / np.sqrt(2.3)
    prob = np.exp(-dist ** 2 / (2 * sigma ** 2))

    if np.sum(prob) == 0:
        # The radius is probably too small, so even in the peak pixel the gaussian prob is tiny.
        # In this case we just have an empty map and set the prob to 1 at the peak.
        prob = np.zeros_like(prob)
        prob[dist == np.min(dist)] = 1

    # normalise the probability
    prob = prob / np.sum(prob)

    return prob


def create_gaussian_map(peak, radius, nside=64, nest=True):
    """Create a HEALPix map with a Gaussian peak at the given coordinates.

    Parameters
    ----------
    peak : scalar `astropy.coordinates.SkyCoord`
        central peak of the distribution

    radius : float
        68% containment radius, in degrees

    nside : int, default = 64
        HEALPix Nside parameter to use when creating the skymap

    nest : bool, default = True
        if True use HEALPix 'NESTED' ordering, if False use 'RING' ordering

    Returns
    -------
    prob_array : `numpy.array`
        the probability data
    """
    # Get the celestial coordinates of each pixel
    npix = hp.nside2npix(nside)
    ipix = range(npix)
    grid = pix2coord(nside, ipix, nest=nest)

    # Calculate the probability at each pixel
    prob = gaussian_prob(grid, peak, radius)

    return prob


def create_gaussian_skymap(peak, radius, nside=64, nest=True):
    """Create a skymap with a Gaussian peak at the given coordinates.

    Parameters
    ----------
    peak : scalar `astropy.coordinates.SkyCoord`
        central peak of the distribution

    radius : float
        68% containment radius, in degrees

    nside : int, default = 64
        HEALPix Nside parameter to use when creating the skymap

    nest : bool, default = True
        if True use HEALPix 'NESTED' ordering, if False use 'RING' ordering

    Returns
    -------
    skymap : `gototile.skymap.SkyMap`
        the data in a SkyMap class
    """
    # Get the probability data
    prob = create_gaussian_map(peak, radius, nside, nest)

    # Create a SkyMap class
    skymap = SkyMap.from_data(prob, nested=nest)

    return skymap
