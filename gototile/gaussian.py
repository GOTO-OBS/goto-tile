import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import healpy as hp
from astropy import units as u
from astropy.io import fits


def prob(ra_grid,dec_grid,ra,dec,radius):
    """calculate the probability of specific grid (gaussian dist.)

    Parameters
    ----------
    ra_grid : float
        ra coordinate of the grid position, in degrees
    dec_grid : float
        dec coordinate of the grid position, in degrees
    ra : float
        central ra, in degrees
    dec : float
        central dec, in degrees
    radius : float
        68% containment radius, in degrees
    """
    # calculate the angular distance between the reported (RA,Dec) and the grid (RA,Dec)
    a = np.sin(np.abs(dec_grid-dec)/2)**2
    b = np.cos(dec)*np.cos(dec_grid)*np.sin(np.abs(ra_grid-ra)/2)**2
    d = 2*np.arcsin(np.sqrt(a+b))
    ang_dis = np.degrees(d)

    # calculate the probability with 2D gaussian function
    sigma = radius/np.sqrt(2.3)
    f = ang_dis/sigma
    prob = 1/(2*np.pi*sigma**2)*np.e**(-0.5*f**2)

    return prob


def gaussian_skymap(ra, dec, radius, nside=64):
    """Calculate the probability for all skymap grids.

    Parameters
    ----------
    ra : float
        central ra, in degrees
    dec : float
        central dec, in degrees
    radius : float
        68% containment radius, in degrees
    """
    position_ra = np.radians(ra)                                     # convert RA_detect to radian
    position_dec = np.radians(dec)                                   # convert Dec_detect to radian

    npix = hp.nside2npix(nside)
    ipix = range(npix)
    theta, phi = hp.pix2ang(nside, ipix)
    ra = phi
    dec = 0.5 * np.pi - theta

    post = prob(ra, dec, position_ra, position_dec, radius)

    post /= np.sum(post * hp.nside2pixarea(nside))
    postcopy = np.copy(post)
    postcopy *= 4 * np.pi / len(post)

    m = Table([postcopy], names=['PROB'])
    m['PROB'].unit = u.pixel ** -1

    ordering = 'RING'
    extra_header = [
          ('PIXTYPE', 'HEALPIX',
           'HEALPIX pixelisation'),
          ('ORDERING', ordering,
           'Pixel ordering scheme: RING, NESTED, or NUNIQ'),
          ('COORDSYS', 'C',
           'Ecliptic, Galactic or Celestial (equatorial)'),
          ('NSIDE', hp.npix2nside(npix),
           'Resolution parameter of HEALPIX'),
          ('INDXSCHM', 'IMPLICIT',
           'Indexing: IMPLICIT or EXPLICIT')]
    hdu = fits.table_to_hdu(m)
    hdu.header.extend(extra_header)
    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
    return hdulist
