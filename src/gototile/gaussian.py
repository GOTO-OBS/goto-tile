"""Functions to create a HEALPix map with a Gaussian peak at a given location."""

import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord

from .grid import SkyGrid
from .skymaptools import pix2coord


def gaussian_prob(grid: SkyGrid, peak: SkyCoord, radius: float) -> np.ndarray:
    """Calculate the probability of points gaussian dist.

    Parameters
    ----------
    grid : `astropy.coordinates.SkyCoord`
        grid coordinates to calculate the probability at
    peak : scalar `astropy.coordinates.SkyCoord`
        central peak of the distribution
    radius : float
        68% containment radius, in degrees

    Returns
    -------
    prob : `numpy.array`
        the probability data

    """
    # for each point on the grid calculate the angular distance to the peak
    dist = peak.separation(grid)
    dist = dist.degree

    # calculate the probability at each point with a 2D gaussian function
    sigma = radius / np.sqrt(2.3)
    prob = np.exp(-(dist**2) / (2 * sigma**2))

    if np.sum(prob) == 0:
        # The radius is probably too small, so even in the peak pixel the gaussian prob is tiny.
        # In this case we just have an empty map and set the prob to 1 at the peak.
        prob = np.zeros_like(prob)
        prob[dist == np.min(dist)] = 1

    # normalise the probability
    return prob / np.sum(prob)


def create_gaussian_map(
    peak: SkyCoord, radius: float, nside: int = 64, nest: bool = True
) -> np.ndarray:
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
    return gaussian_prob(grid, peak, radius)
