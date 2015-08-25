#!/usr/bin/env python
#
# Copyright (C) 2013  Leo Singer
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

#
# Modified by Darren J. White for GOTO project, using non-LIGO systems.
#



"""
Reading and writing HEALPix FITS files. An example FITS header looks like this:

$ funhead -a test.fits.gz
SIMPLE  =					T / conforms to FITS standard
BITPIX  =					8 / array data type
NAXIS   =					0 / number of array dimensions
EXTEND  =					T
END
	  Extension: xtension

XTENSION= 'BINTABLE'		   / binary table extension
BITPIX  =					8 / array data type
NAXIS   =					2 / number of array dimensions
NAXIS1  =				 4096 / length of dimension 1
NAXIS2  =				  192 / length of dimension 2
PCOUNT  =					0 / number of group parameters
GCOUNT  =					1 / number of groups
TFIELDS =					1 / number of table fields
TTYPE1  = 'PROB	'
TFORM1  = '1024E   '
TUNIT1  = 'pix-1   '
PIXTYPE = 'HEALPIX '		   / HEALPIX pixelisation
ORDERING= 'RING	'		   / Pixel ordering scheme, either RING or NESTED
COORDSYS= 'C	   '		   / Ecliptic, Galactic or Celestial (equatorial)
EXTNAME = 'xtension'		   / name of this binary table extension
NSIDE   =				  128 / Resolution parameter of HEALPIX
FIRSTPIX=					0 / First pixel # (0 based)
LASTPIX =			   196607 / Last pixel # (0 based)
INDXSCHM= 'IMPLICIT'		   / Indexing: IMPLICIT or EXPLICIT
OBJECT  = 'FOOBAR 12345'	   / Unique identifier for this event
REFERENC= 'http://www.youtube.com/watch?v=0ccKPSVQcFk' / URL of this event
DATE-OBS= '2013-04-08T21:37:32.25' / UTC date of the observation
MJD-OBS =	  56391.151064815 / modified Julian date of the observation
DATE	= '2013-04-08T21:50:32' / UTC date of file creation
CREATOR = 'fits.py '		   / Program that created this file
RUNTIME =				 21.5 / Runtime in seconds of the CREATOR program
END
"""
__author__ = "Leo Singer <leo.singer@ligo.org>"
__all__ = ("read_sky_map", "write_sky_map")


import gzip
import math
import os
import shutil
import tempfile
import healpy as hp
from healpy.fitsfunc import getformat, pixelfunc, standard_column_names, pf, np


#
# Based on https://github.com/healpy/healpy/blob/1.6.1/healpy/fitsfunc.py.
# Reproduced with permission from Andrea Zonca.
#
# Modifications:
#  * Added extra_metadata= argument to inject additional values into header.
#  * Added optional unit= argument to set units of table data.
#  * Support writing to gzip-compressed FITS files.
#
# FIXME: Instead of pyfits, use astropy.io.fits; it supports gzip compression.
#
def write_map(filename,m,nest=False,dtype=np.float32,fits_IDL=True,coord=None,column_names=None,unit=None,extra_metadata=()):
    """Writes an healpix map into an healpix file.

    Parameters
    ----------
    filename : str
      the fits file name
    m : array or sequence of 3 arrays
      the map to write. Possibly a sequence of 3 maps of same size.
      They will be considered as I, Q, U maps.
      Supports masked maps, see the `ma` function.
    nest : bool, optional
      If False, ordering scheme is NESTED, otherwise, it is RING. Default: RING.
    fits_IDL : bool, optional
      If True, reshapes columns in rows of 1024, otherwise all the data will
      go in one column. Default: True
    coord : str
      The coordinate system, typically 'E' for Ecliptic, 'G' for Galactic or 'C' for
      Celestial (equatorial)
    column_names : str or list
      Column name or list of column names, if None we use:
      I_STOKES for 1 component,
      I/Q/U_STOKES for 3 components,
      II, IQ, IU, QQ, QU, UU for 6 components,
      COLUMN_0, COLUMN_1... otherwise
    """
    if not hasattr(m, '__len__'):
        raise TypeError('The map must be a sequence')
    # check the dtype and convert it
    fitsformat = getformat(dtype)

    m = pixelfunc.ma_to_array(m)
    if pixelfunc.maptype(m) == 0: # a single map is converted to a list
        m = [m]

    if column_names is None:
        column_names = standard_column_names.get(len(m), ["COLUMN_%d" % n for n in range(len(m))])
    else:
        assert len(column_names) == len(m), "Length column_names != number of maps"

    # maps must have same length
    assert len(set(map(len, m))) == 1, "Maps must have same length"
    nside = pixelfunc.npix2nside(len(m[0]))

    if nside < 0:
        raise ValueError('Invalid healpix map : wrong number of pixel')

    cols=[]
    for cn, mm in zip(column_names, m):
        if len(mm) > 1024 and fits_IDL:
            # I need an ndarray, for reshape:
            mm2 = np.asarray(mm)
            cols.append(pf.Column(name=cn,
                                   format='1024%s' % fitsformat,
                                   array=mm2.reshape(mm2.size/1024,1024),
                                   unit=unit))
        else:
            cols.append(pf.Column(name=cn,
                                   format='%s' % fitsformat,
                                   array=mm,
                                   unit=unit))

    tbhdu = pf.new_table(cols)
    # add needed keywords
    tbhdu.header.update('PIXTYPE','HEALPIX','HEALPIX pixelisation')
    if nest: ordering = 'NESTED'
    else:    ordering = 'RING'
    tbhdu.header.update('ORDERING',ordering,
                        'Pixel ordering scheme, either RING or NESTED')
    if coord:
        tbhdu.header.update('COORDSYS',coord,
                            'Ecliptic, Galactic or Celestial (equatorial)')
    tbhdu.header.update('EXTNAME','xtension',
                        'name of this binary table extension')
    tbhdu.header.update('NSIDE',nside,'Resolution parameter of HEALPIX')
    tbhdu.header.update('FIRSTPIX', 0, 'First pixel # (0 based)')
    tbhdu.header.update('LASTPIX',pixelfunc.nside2npix(nside)-1,
                        'Last pixel # (0 based)')
    tbhdu.header.update('INDXSCHM','IMPLICIT',
                        'Indexing: IMPLICIT or EXPLICIT')

    for metadata in extra_metadata:
        tbhdu.header.update(*metadata)

    # FIXME: use with-clause, but GzipFile doesn't support it in Python 2.6.
    # We can't even use GzipFile because the ancient version of PyFITS that is
    # in SL6 is too broken, so we have to write the file and then compress it.
    basename, ext = os.path.splitext(filename)
    if ext == '.gz':
        with tempfile.NamedTemporaryFile(suffix='.fits') as tmpfile:
            tbhdu.writeto(tmpfile.name, clobber=True)
            gzfile = gzip.GzipFile(filename, 'wb')
            try:
                try:
                    shutil.copyfileobj(tmpfile, gzfile)
                finally:
                    gzfile.close()
            except:
                os.unlink(gzfile.name)
                raise
    else:
        tbhdu.writeto(filename, clobber=True)


def read_sky_map(filename, nest=False):
    """
    Read a LIGO/Virgo-type sky map and return a tuple of the HEALPix array
    and a dictionary of metadata from the header.

    Parameters
    ----------

    filename: string
        Path to the optionally gzip-compressed FITS file.

    nest: bool, optional
        If omitted or False, then detect the pixel ordering in the FITS file
        and rearrange if necessary to RING indexing before returning.

        If True, then detect the pixel ordering and rearrange if necessary to
        NESTED indexing before returning.

        If None, then preserve the ordering from the FITS file.

        Regardless of the value of this option, the ordering used in the FITS
        file is indicated as the value of the 'nest' key in the metadata
        dictionary.
    """
    prob, header = hp.read_map(filename, h=True, verbose=False, nest=None)
    header = dict(header)

    metadata = {}

    metadata['file']=filename

    ordering = header['ORDERING']
    if ordering == 'RING':
        metadata['nest'] = False
    elif ordering == 'NESTED':
        metadata['nest'] = True
    else:
        raise ValueError(
            'ORDERING card in header has unknown value: {0}'.format(ordering))

    try:
        value = header['OBJECT']
    except KeyError:
        pass
    else:
        metadata['objid'] = value

    try:
        value = header['REFERENC']
    except KeyError:
        pass
    else:
        metadata['url'] = value


    try:
        value = header['MJD-OBS']
    except KeyError:
        pass
    else:
        metadata['mjd'] = value

    try:
        value = header['DATE-OBS']
    except KeyError:
        pass
    else:
        metadata['date'] = value

    try:
        value = header['CREATOR']
    except KeyError:
        pass
    else:
        metadata['creator'] = value

    try:
        value = header['ORIGIN']
    except KeyError:
        pass
    else:
        metadata['origin'] = value

    try:
        value = header['RUNTIME']
    except KeyError:
        pass
    else:
        metadata['runtime'] = value

    try:
        value = header['NSIDE']
    except KeyError:
        print "Getting nside parameter from length of skymap, header missing"
        metadata['nside'] = hp.npix2nside(len(prob))
    else:
        metadata['nside'] = value

    return prob, metadata


if __name__ == '__main__':
    import healpy as hp
    import numpy as np
    nside = 128
    npix = hp.nside2npix(nside)
    prob = np.random.random(npix)
    prob /= sum(prob)

    write_sky_map('test.fits.gz', prob,
        objid='FOOBAR 12345',
        gps_time=1049492268.25,
        creator=os.path.basename(__file__),
        url='http://www.youtube.com/watch?v=0ccKPSVQcFk',
        origin='LIGO Scientific Collaboration',
        runtime=21.5)

    print read_sky_map('test.fits.gz')
