import os
import numpy as np
import astropy
from astropy import units
from astropy.coordinates import SkyCoord
import healpy
try:
    stringtype = basestring  # Python 2
except NameError:
    stringtype = str  # Python 3


class SkyMap(object):
    """A probability skymap

    The SkyMap is a wrapper around the healpy skymap numpy.array,
    returned by healpy.fitsfunc.read_map. The SkyMap class holds track
    of the numpy array, the header information and some options.

    """

    def __init__(self, skymap, header=None, **kwargs):
        if isinstance(skymap, stringtype):
            skymap, header = self._read_file(skymap)
        elif not isinstance(header, dict):
            raise TypeError("header should be a dict")
        self.object = header['object']
        self.order = header['order']
        self.nside = header['nside']
        self.isnested = header['nested']
        self.skymap = skymap
        self.header = header

    def _read_file(self, filename):
        skymap, header = healpy.read_map(filename, h=True,
                                     verbose=False, nest=None)
        header = dict([(key.lower(), value) for key, value in header])
        header['file'] = filename
        if header['ordering'] not in ('NESTED', 'RING'):
            raise ValueError(
                'ORDERING card in header has unknown value: {}'.format(
                    header['ordering']))
        header['order'] = header['ordering']
        header['nested'] = header['order'] == 'NESTED'

        objid = os.path.basename(filename)
        # Twice, in case we use a .fits.gz file
        objid = os.path.splitext(objid)[0]
        objid = os.path.splitext(objid)[0]
        header['objid'] = header.get('object', objid)
        header['url'] = header.get('referenc', '')

        header['mjddet'] = header.get(
            'mjd-obs', astropy.time.Time(header['date']).mjd)
        header['mjd'] = astropy.time.Time.now().mjd
        header['date-det'] = astropy.time.Time(header['mjddet'],
                                                 format='mjd')
        header['date'] = astropy.time.Time(header['mjd'], format='mjd')

        header['nside'] = header.get('nside', healpy.npix2nside(len(skymap)))

        return skymap, header

    def regrade(self, nside=None, order=None, power=None, pess=False,
                dtype=None):
        """Up- or downgrade the skymap resolution.

        See the healpy.pixelfunc.ud_grade documentation about the options.

        """

        self.skymap = healpy.ud_grade(self.skymap, nside_out=nside,
                                      order_in=self.order, order_out=order,
                                      power=power, pess=pess, dtype=dtype)
        self.nside = nside
        self.order = order


    def skycoords(self):
        """Return the sky coordinates (RA, Dec) for the current map.

        The returned value is an astropy.coordinates.SkyCoord object,
        with the number of coordinates equal to the size of the
        skyamp.

        """

        npix = len(self.skymap)
        ipix = np.arange(npix)
        theta, phi = healpy.pix2ang(self.nside, ipix, nest=self.isnested)
        skycoords = SkyCoord(ra=phi*units.rad, dec=(0.5*np.pi - theta)*units.rad)        
        return skycoords
