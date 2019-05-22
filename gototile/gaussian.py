import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import healpy as hp
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord 
from astropy.coordinates import Angle

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
    source_coor = SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
    goto_grid = SkyCoord(ra_grid*u.deg, dec_grid*u.deg, frame='fk5')
    ang_dis = source_coor.separation(goto_grid)
    ang_dis = ang_dis.degree

    # calculate the probability with 2D gaussian function
    sigma = radius/np.sqrt(2.3)
    prob = np.exp(-ang_dis**2/(2*sigma**2))
    norm = np.sum(prob)
    prob /= norm

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
    
    npix = hp.nside2npix(nside)
    ipix = range(npix)
    theta, phi = hp.pix2ang(nside, ipix)
    ra_grid = phi
    dec_grid = 0.5 * np.pi - theta
    ra_grid = Angle(ra_grid, u.radian).degree
    dec_grid = Angle(dec_grid, u.radian).degree

    post = prob(ra_grid, dec_grid, ra, dec, radius)

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
