import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

    # normalise the probability
    prob = prob / np.sum(prob)

    return prob


def create_gaussian_map(ra, dec, radius, nside=64, nest=True):
    """Create a HEALPix map with a Gaussian peak at the given coordinates.

    Parameters
    ----------
    ra : float
        central ra, in degrees
    dec : float
        central dec, in degrees
    radius : float
        68% containment radius, in degrees
    nside : int, default = 64
        HEALPix Nside parameter to use when creating the skymap
    nest : bool, default = True
        if True use HEALPix 'NESTED' ordering, if False use 'RING' ordering
    """
    # Create an Astropy SkyCoord at the peak
    peak_coord = SkyCoord(ra, dec, unit='deg')

    # Get the celestial coordinates of each pixel
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    grid_coords = pix2coord(nside, ipix, nest=nest)

    # Calculate the probability at each pixel
    prob = gaussian_prob(grid_coords, peak_coord, radius)

    return prob
