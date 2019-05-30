from __future__ import division

import os
import itertools
import logging
import numpy as np
import astropy
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord, AltAz
from astropy import units as u
from astropy.table import QTable
import healpy
import ephem
from . import settings
from . import skymaptools as smt
from .gaussian import create_gaussian_map
from matplotlib import pyplot as plt
import ligo.skymap.plot

try:
    stringtype = basestring  # Python 2
except NameError:
    stringtype = str  # Python 3


def read_colormaps(name='cylon'):
    """Read special color maps, such as 'cylon'"""
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    filename = os.path.join(settings.DATA_DIR, name + '.csv')
    data = np.loadtxt(filename, delimiter=',')
    cmap = LinearSegmentedColormap.from_list(name, data)
    cm.register_cmap(cmap=cmap)
    cmap = LinearSegmentedColormap.from_list(name+'_r', data[::-1])
    cm.register_cmap(cmap=cmap)


class SkyMap(object):
    """A probability skymap.

    The SkyMap is a wrapper around the healpy skymap numpy.array,
    returned by healpy.fitsfunc.read_map. The SkyMap class holds track
    of the numpy array, the header information and some options.

    SkyMaps should be created using one of the following class methods:
        - SkyMap.from_fits(fits_file)
            Will read in the sky map from a FITS file.
            `fits_file` can be a string (the .FITS file to be loaded) or an
            already-loaded `astropy.io.fits.HDU` or `HDUList`.

        - SkyMap.from_position(ra, dec, radius)
            Will create the SkyMap from the given coordinates.
            The arguments should be in decimal degrees.
            The sky map will be calculated as a 2D Gaussian distribution
            around the given position.
    """

    def __init__(self, skymap, header):
        # Check types
        if not isinstance(skymap, np.ndarray):
            raise TypeError("skymap should be an array, use SkyMap.from_fits()")
        if not isinstance(header, dict):
            raise TypeError("header should be a dict")

        # Convert the skymap to the requested type from settings
        dtype = getattr(settings, 'DTYPE')
        skymap = skymap.astype(dtype)

        # Make sure the header cards are lowercase
        header = {key.lower(): header[key] for key in header}

        # Check the header NSIDE matches the skymap data
        try:
            header_nside = header['nside']
        except KeyError:
            raise ValueError('No NSIDE value in the header')
        skymap_nside = healpy.npix2nside(len(skymap))
        if not header_nside == skymap_nside:
            raise ValueError("NSIDE from header ({:.0f}) doesn't match skymap ({:.0f})".format(
                             header_nside, skymap_nside))

        # Get the data ordering from the header
        try:
            order = header['ordering']
        except KeyError:
            raise ValueError('No ORDERING value in the header')
        if order not in ('NESTED', 'RING'):
            raise ValueError('ORDERING card in header has unknown value: {}'.format(order))

        # Get the coordinate system from the header
        try:
            coordsys = header['coordsys'][0]
        except KeyError:
            raise ValueError('No COORDSYS value in the header')
        if coordsys not in ('G', 'E', 'C'):
            raise ValueError('COORDSYS card in header has unknown value: {}'.format(coordsys))
        self.coordsys = coordsys

        # Parse and store the skymap
        self._save_skymap(skymap, order)

        # Store the header and key infomation as attributes
        self.header = header

        self.filename = self.header.get('filename')

        alt_name = ''
        if self.filename:
            alt_name = os.path.basename(self.filename).split('.')[0]
        else:
            alt_name = 'unknown'
        self.object = self.header.get('object', alt_name)
        if 'coinc_event_id:' in self.object:
            # for test events
            self.object = self.object.split(':')[-1]
        self.objid = self.object

        self.url = self.header.get('referenc', '')

        self.mjd = astropy.time.Time.now().mjd
        self.date = astropy.time.Time(float(self.mjd), format='mjd')
        self.mjd_det = self.header.get('mjd-obs', self.mjd)
        self.date_det = astropy.time.Time(float(self.mjd_det), format='mjd')

    def __eq__(self, other):
        try:
            if len(self.skymap) != len(other.skymap):
                return False
            return np.all(self.skymap == other.skymap) and self.header == other.header
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError('SkyMaps can only be multipled by other SkyMaps')

        result = self.copy()
        other_copy = other.copy()

        if self.nside != other_copy.nside or self.order != other_copy.order:
            other_copy.regrade(self.nside, self.order)
        if self.coordsys != other_copy.coordsys:
            other_copy.rotate(self.coordsys)

        new_skymap = result.skymap * other_copy.skymap
        result._save_skymap(new_skymap, order=self.order)

        return result

    def __repr__(self):
        template = ('SkyMap(objid="{}", date_det="{}", nside={})')
        return template.format(self.objid, self.date_det.iso, self.nside)

    def _pix2coord(self, pix):
        """Convert HEALpy pixel indexes to SkyCoords."""
        return smt.pix2coord(self.nside, pix, nest=self.isnested)

    def _coord2pix(self, coord):
        """Convert SkyCoords to HEALpy pixel indexes."""
        return smt.coord2pix(self.nside, coord, nest=self.isnested)

    def _save_skymap(self, skymap, order):
        """Save the skymap data and add attributes."""
        self.skymap = skymap
        self.npix = len(skymap)
        self.nside = healpy.npix2nside(self.npix)
        self.pixel_area = healpy.nside2pixarea(self.nside, degrees=True)
        self.order = order
        self.isnested = order == 'NESTED'

        # Save the coordinates of each skymap pixel
        all_pixels = range(self.npix)
        self.coords = self._pix2coord(all_pixels)

        # Calculate the probability contours
        self._get_contours()

    def _get_contours(self):
        """Store the contour infomation of each pixel.

        The cumulative probability of the pixels sorted in order corresponds to the
        minimum probability contour that that pixel is within.

        For example, a very small skymap has the following table:

        pix | prob
          1 |  0.1
          2 |  0.4
          3 |  0.2
          4 |  0.3

        Sorted by probability, and finding the cumulative sum:

        pix | prob | cumprob
          2 |  0.4 |  0.4
          4 |  0.3 |  0.7
          3 |  0.2 |  0.9
          1 |  0.1 |  1.0

        So only pixel 2 is within the 50% probability contour (== confidence region),
        while 2, 4 and 3 are within the 90% region.
        Obviously all 4 pixels are within the 100% region.

        This means if we sort back to the origional order the minimum confidence region each
        pixel is in is easy to find by seeing if cumprob(pixel) < percentage:

        pix | prob | cumprob | in 90%? | in 50%?
          1 |  0.1 |  1.0    | False   | False
          2 |  0.4 |  0.4    | True    | True
          3 |  0.2 |  0.9    | True    | False
          4 |  0.3 |  0.7    | True    | False

        See also:
            SkyMap._pixels_within_contour(percentage)
            SkyMap.get_contour(coord)
            SkyMap.within_contour(coord, percentage)

        This is (vaguely) based on code from http://www.virgo-gw.eu/skymap.html
        """
        # Get the indixes sorted by probability (reversed, so highest first)
        # Note what we call the 'pixels' are really just the index numbers of the skymap,
        # i.e. probability of pixel X = self.skymap[X]
        sorted_pixels = self.skymap.argsort()[::-1]

        # Sort the skymap using this mapping
        sorted_skymap = self.skymap[sorted_pixels]

        # Create cumulative sum array of each pixel in the skymap
        cumprob = np.cumsum(sorted_skymap)

        # "Un-sort" the cumulative array back to the normal pixel order
        contours = cumprob[sorted_pixels.argsort()]

        # And save the contours on the SkyMap
        self.contours = contours

    def _pixels_within_contour(self, percentage):
        """Find pixel indices confined in a given percentage contour (range 0-1)."""

        if not 0 <= percentage <= 1:
            raise ValueError('Percentage must be in range 0-1')

        # Return early if the percentage is too low
        # NB for the record min(self.contours) == max(self.skymap)
        if percentage < min(self.contours):
            return []

        # Find the pixels within the given contour
        mask = self.contours < percentage

        return np.arange(self.npix)[mask]

    @classmethod
    def from_fits(cls, fits_file):
        """Initialize a `~gototile.skymap.SkyMap` object from a FITS file.

        Parameters
        ----------
        fits_file : str, `astropy.io.fits.HDU` or `astropy.io.fits.HDUList`
            Path to the FITS file (if str) or FITS HDU,
            to be passed to `healpy.read_map`.

        Returns
        -------
        `~gototile.skymap.SkyMap``
            SkyMap object.
        """
        info = healpy.read_map(fits_file, h=True, field=None,
                               verbose=False, nest=None)
        # `info` will be an array or multiple arrays, with the header appended (because h=True).
        skymap = info[0]
        header = dict(info[-1])

        # Dealing with newer 3D skymaps, the "skymap" will have 4 components
        # (prob, distmu, distsigma, distnorm)
        # We only want the probability map
        if header['TFIELDS'] > 1:
            skymap = skymap[0]

        # Store the file name if the header was from a file
        if isinstance(fits_file, str):
            header['FILENAME'] = fits_file

        return cls(skymap, header)

    @classmethod
    def from_data(cls, data, nested=True, coordsys='C'):
        """Initialize a `~gototile.skymap.SkyMap` object from an array of data.

        Parameters
        ----------
        data : list or `numpy.array`
            an array of data to map onto a HEALPix sphere
            the length of the data must match one of the valid HEALPix resolutions

        nested : bool
            if True the data has order=NESTED, if False then order=RING

        coordsys : str
            The coordinate system the data uses.
            'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

        Returns
        -------
        `~gototile.skymap.SkyMap``
            SkyMap object.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Check the data is a valid length
        try:
            nside = healpy.npix2nside(len(data))
        except ValueError:
            raise ValueError('Length of data is invalid')

        header = {'PIXTYPE': 'HEALPIX',
                  'ordering': 'NESTED' if nested else 'RING',
                  'COORDSYS': coordsys[0],
                  'NSIDE': nside,
                  'INDXSCHM': 'IMPLICIT',
                  }

        return cls(data, header)

    @classmethod
    def from_position(cls, ra, dec, radius, nside=64):
        """Initialize a `~gototile.skymap.SkyMap` object from a sky position and radius.

        Parameters
        ----------
        ra : float
            ra in decimal degrees
        dec : float
            declination in decimal degrees
        radius : float
            68% containment radius in decimal degrees
        nside : int, optional
            healpix nside parameter (must be a power of 2)
            default is 64

        Returns
        -------
        `~gototile.skymap.SkyMap``
            SkyMap object.
        """
        prob_map = create_gaussian_map(ra, dec, radius, nside, nest=True)
        return cls.from_data(prob_map)

    def copy(self):
        """Return a new instance containing a copy of the sky map data."""
        newmap = SkyMap(self.skymap.copy(), self.header.copy())
        return newmap

    def regrade(self, nside=None, order='NESTED',
                power=-2, pess=False, dtype=None):
        """Up- or downgrade the sky map  HEALPix resolution.

        See the `healpy.pixelfunc.ud_grade()` documentation for the parameters.
        """
        if not nside:
            nside = self.nside
        if order not in ['NESTED', 'RING']:
            raise ValueError('Pixel order must be NESTED or RING, not {}'.format(order))
        if nside == self.nside and order == self.order:
            return

        # Regrade the current skymap
        new_skymap = healpy.ud_grade(self.skymap, nside_out=nside,
                                     order_in=self.order, order_out=order,
                                     power=power, pess=pess, dtype=dtype)

        # Save the new skymap
        self._save_skymap(new_skymap, order)

        # Update the header
        self.header['nside'] = nside
        self.header['ordering'] = order

    def rotate(self, coordsys='C'):
        """Convert coordinate systems.

        Parameters
        ------------
        coordsys : str
            First character is the coordinate system to convert to.
            As in HEALPIX, allowed coordinate systems are:
            'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)
        """
        if self.coordsys == coordsys:
            return

        rotator = healpy.Rotator(coord=(self.coordsys, coordsys))

        # NOTE: rotator expectes order=RING in and returns order=RING out
        # If this skymap is NESTED we need to regrade before and after
        if self.order == 'NESTED':
            in_skymap = healpy.ud_grade(self.skymap, nside_out=self.nside,
                                        order_in='NESTED', order_out='RING')
        else:
            in_skymap = self.skymap.copy()

        # Rotate the skymap, now we're sure it's in RING order
        out_skymap = rotator.rotate_map(in_skymap)

        # Convert back to NESTED if needed
        if self.order == 'NESTED':
            out_skymap = healpy.ud_grade(out_skymap, nside_out=self.nside,
                                         order_in='RING', order_out='NESTED')

        # Save the new skymap
        self._save_skymap(out_skymap, self.order)

        # Update the header
        self.header['coordsys'] = coordsys
        self.coordsys = coordsys

    def normalise(self):
        """Normalise the sky map so the probability sums to unity."""
        norm_skymap = self.skymap / self.skymap.sum()

        # Save the new skymap
        self._save_skymap(norm_skymap, self.order)

    def get_probability(self, coord, radius=0):
        """Return the probability at a given sky coordinate.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            The point to find the probability at.
        radius : float, optional
            If given, the radius in degrees of a circle to integrate the probability within.
        """
        # Find distance to points
        sep = np.array(coord.separation(self.coords))

        if radius == 0:
            # Just get the radius of the nearest pixel (not very useful)
            pixel = np.where(sep == (min(sep)))[0][0]
            prob = self.skymap[pixel]
        else:
            # Find all the pixels within the radius and sum them (more useful)
            pixels = np.where(sep < radius)[0]
            prob = self.skymap[pixels].sum()

        return prob

    def get_contour(self, coord):
        """Return the lowest probability contor the given sky coordinate is within.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            The point to find the probability at.
        """
        # Get the pixel that the coordinates are within
        pixel = self._coord2pix(coord)

        return self.contours[pixel]

    def within_contour(self, coord, percentage):
        """Find if the given position is within the given confidence level.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            The point to find the probability at.
        percentage : float
            The confidence level, percentage in the range 0-1.
        """
        if not 0 <= percentage <= 1:
            raise ValueError('Percentage must be in range 0-1')

        contour = self.get_contour(coord)

        return contour < percentage

    def get_table(self):
        """Return an astropy QTable containing infomation on the skymap pixels."""
        col_names = ['pixel', 'ra', 'dec', 'prob']
        col_types = ['U', u.deg, u.deg, 'f8']

        npix = len(self.skymap)
        ipix = np.arange(npix)
        coords = self.coords

        table = QTable([ipix, coords.ra, coords.dec, self.skymap],
                        names=col_names, dtype=col_types)
        return table

    def plot(self, filename=None, dpi=300, coordinates=None, plot_contours=True):
        """Plot the skymap.

        Parameters
        ----------
        filename : str, optional
            filename to save the plot to
            if not given then the plot will be displayed with plt.show()

        dpi : int, optional
            DPI to save the plot at
            default is 300

        coordinates : `astropy.coordinates.SkyCoord`, optional
            any coordinates to also plot on the image

        plot_contours : bool, default = True
            plot the 50% and 90% contour areas

        """
        figure = plt.figure(figsize=(8,6))

        # Can only plot in equatorial coordinates
        # If it's not, temporarily rotate into equatorial and then go back afterwards
        if not self.coordsys == 'C':
            old_coordsys = self.coordsys
            self.rotate('C')
        else:
            old_coordsys = None

        axes = plt.axes(projection='astro hours mollweide')
        axes.grid()
        transform = axes.get_transform('world')

        # Plot the skymap data
        axes.imshow_hpx(self.skymap, cmap='cylon', nested=self.isnested)

        # Plot 50% and 90% contours
        if plot_contours:
            cs = axes.contour_hpx(self.contours , nested=self.isnested,
                                  levels = [0.5 * self.skymap.sum(),
                                            0.9 * self.skymap.sum()],
                                  colors='black', linewidths=0.5, zorder=99,)
        #axes.clabel(cs, inline=False, fontsize=7, fmt='%.0f')

        # Plot coordinates if given
        if coordinates:
            axes.scatter(coordinates.ra.value, coordinates.dec.value,
                         transform=transform,
                         s=99, c='blue', marker='*', zorder=9)
            if coordinates.isscalar:
                coordinates = SkyCoord([coordinates])
            for coord in coordinates:
                axes.text(coord.ra.value, coord.dec.value,
                            coord.to_string('hmsdms').replace(' ','\n')+'\n',
                            transform=transform,
                            ha='center', va='bottom',
                            size='x-small', zorder=12,
                            )

        # Remember to rotate back!
        if old_coordsys:
            self.rotate(old_coordsys)

        # Set title
        title = 'Skymap for trigger {}'.format(self.objid)
        axes.set_title(title, y=1.05)

        # Save or show
        if filename:
            plt.savefig(filename, dpi=dpi)
        else:
            plt.show()
