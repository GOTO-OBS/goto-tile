import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import healpy as hp
from astropy import units as u
from astropy.io import fits

from .skymap import SkyMap

def prob(ra_grid,dec_grid,ra,dec,err):
    """calculate the probability of specific grid (gaussian dist.)"""
    sys_err = np.sqrt((3.71**2)*0.9+(14.3**2)*0.1)          # quadrature sum of systematic error
    tot_err = np.sqrt(sys_err**2 + err**2)                  # quadrature sum of total error

    # calculate the angular distance between the reported (RA,Dec) and the grid (RA,Dec)
    a = np.sin(np.abs(dec_grid-dec)/2)**2
    b = np.cos(dec)*np.cos(dec_grid)*np.sin(np.abs(ra_grid-ra)/2)**2
    d = 2*np.arcsin(np.sqrt(a+b))
    ang_dis = np.degrees(d)

    # calculate the probability with 2D gaussian function
    f = ang_dis/tot_err
    prob = 1/(2*np.pi*tot_err**2)*np.e**(-0.5*f**2)

    return prob

def fermi_skymap(gbm_ra, gbm_dec, gbm_err):                         
    """calculate the probability for all skymap grids"""
    gbm_ra = np.radians(gbm_ra)                                     # convert RA_detect to radian
    gbm_dec = np.radians(gbm_dec)                                   # convert Dec_detect to radian

    nside = 64
    npix = hp.nside2npix(nside)
    ipix = range(npix)
    theta, phi = hp.pix2ang(nside, ipix)
    ra = phi
    dec = 0.5 * np.pi - theta
    radec = np.column_stack([ra,dec])

    post = np.zeros(npix)
    for i,coo in enumerate(radec):
        post[i] = prob(coo[0],coo[1],gbm_ra,gbm_dec,gbm_err)
    post = np.asarray(list(post))
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
    gw = hp.read_map(hdulist, verbose=False)

    header_dict = {key.lower(): hdu.header[key] for key in hdu.header}
    skymap = SkyMap(gw, header=header_dict)
    return skymap
