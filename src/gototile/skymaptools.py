"""Module containing utility functions for the SkyGrid class."""

from functools import lru_cache

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def get_data_contours(data, min_zero=True):
    """Calculate the minimum contour level of each pixel in a given skymap data array.

    This is done using the cumulative sum method, (vaguely) based on code from
    http://www.virgo-gw.eu/skymap.html or
    ligo.skymap.postprocess.util.find_greedy_credible_levels().

    For example, consider a very small, normalised skymap with the following table:

    ipix | value
       1 |  0.1
       2 |  0.4
       3 |  0.2
       4 |  0.3

    Sort by value, and find the cumulative sum:

    ipix | value | cumsum(value)
       2 |   0.4 |  0.4
       4 |   0.3 |  0.7
       3 |   0.2 |  0.9
       1 |   0.1 |  1.0

    Now shift so the cumsum starts at zero, and that's the minimum contour level that each
    pixel is within.

    ipix | value | contour
       2 |   0.4 |  0.0
       4 |   0.3 |  0.4
       3 |   0.2 |  0.7
       1 |   0.1 |  0.9

    This shift is apparently controversial, since the ligo.skymap function is identical just
    without the shift. The actual effect is very small, but I think it makes more sense to include.

    Consider asking for the minimum number of pixels to cover increasing contour levels:
        -  0%-40%: you only need pixel 2
        - 40%-70%: you need pixels 2 & 3
        - 70%-90%: you need pixels 2, 3 & 4
        - 90%+:    you need all four pixels
    That's why we shift everything up so the first pixel has a contour value of 0%, because you
    should always include at least one pixel to cover the smallest contour levels.

    If we sort back to the original order the minimum confidence region each
    pixel is in is easy to find by seeing if contour(pixel) < percentage:

    ipix | value | contour | in 90%? | in 50%?
       1 |   0.1 |  0.9    | False   | False
       2 |   0.4 |  0.0    | True    | True
       3 |   0.2 |  0.7    | True    | False
       4 |   0.3 |  0.4    | True    | True

    If you select the pixels for which contour(pixel) < percentage you will always cover
    AT LEAST percentage (you may well cover more of course).
    """
    # Get the pixel indices sorted by each pixel's value (but reversed, so highest first)
    # Note 'ipix' are the index numbers of the data array,i.e. the value of pixel X = self.data[X]
    sorted_ipix = np.flipud(np.argsort(data))

    # Sort the data using this mapping
    sorted_data = data[sorted_ipix]

    # Create cumulative sum array of each pixel in the array
    sorted_contours = np.cumsum(sorted_data)

    if min_zero:
        # Shift so we start at 0
        np.roll(sorted_contours, 1)
        sorted_contours[0] = 0

    # "Un-sort" the contour array back to the normal pixel order
    contours = sorted_contours[np.argsort(sorted_ipix)]
    return contours


def coord2pix(nside, coord, nest=False):
    """Convert sky coordinates to pixel indices.

    Parameters
    ----------
    nside : int or array-like
        The healpix nside parameter, must be a power of 2, less than 2**30
    coord : `astropy.coordinates.SkyCoord`
        The coordinates of the point(s) in the sky
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering

    Returns
    -------
    ipix : int or array of int
        The healpix pixel indices. Scalar if all input are scalar, array otherwise.

    See Also
    --------
    pix2coord, `healpy.ang2pix`

    """
    # Convert sky coordinates to angles
    theta = 0.5 * np.pi - coord.dec.rad
    phi = coord.ra.rad

    # Get pixel numbers from healpy
    ipix = hp.ang2pix(nside, theta, phi, nest)

    # Return pixels
    return ipix


def pix2coord(nside, ipix, nest=False):
    """Convert pixel index or indices to sky coordinates.

    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30
    ipix : int or array-like
        Pixel indices
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering

    Returns
    -------
    coord : `astropy.coordinates.SkyCoord`
        The coordinates corresponding to ipix, as an Astropy SkyCoord.
        Scalar if all input are scalar, array otherwise.

    See Also
    --------
    coord2pix, `healpy.pix2ang`

    """
    # Check types
    if isinstance(nside, (list, np.ndarray)):
        nside = tuple(nside)
    if isinstance(ipix, (list, np.ndarray)):
        ipix = tuple(ipix)

    # Now can safely call the cached function
    return _pix2coord_cached(nside, ipix, nest)


@lru_cache(maxsize=128)
def _pix2coord_cached(nside, ipix, nest=False):
    """Convert pixel index or indices to sky coordinates, and cache the results.

    This is the same as pix2coord, but uses the `functools.lru_cache` decorator.

    It's useful as this function is often called with the same arguments
    when creating multiple skymaps with the same resolution.

    Unfortunately that requires hashable inputs, so need to convert lists and arrays to tuples.

    """
    # Get angular coordinates from healpy
    theta, phi = hp.pix2ang(nside, ipix, nest)

    # Convert angles to sky coordinates
    ra = phi
    dec = 0.5 * np.pi - theta

    # Return a SkyCoord object
    return SkyCoord(ra, dec, unit=u.rad)
