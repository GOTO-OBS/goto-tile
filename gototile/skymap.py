"""Module containing the SkyMap class."""

import os
import warnings

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits.verify import VerifyWarning
from astropy.time import Time
from astropy.table import QTable

import healpy as hp

import ligo.skymap.plot  # noqa: F401  (for extra projections)

from matplotlib import pyplot as plt
if 'DISPLAY' not in os.environ:
    plt.switch_backend('agg')

import numpy as np

from . import settings
from .gaussian import create_gaussian_map
from .skymaptools import coord2pix, pix2coord


def read_colormaps(name='cylon'):
    """Read special color maps, such as 'cylon'"""
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    filename = os.path.join(settings.DATA_DIR, name + '.csv')
    data = np.loadtxt(filename, delimiter=',')
    cmap = LinearSegmentedColormap.from_list(name, data)
    cm.register_cmap(cmap=cmap)
    cmap = LinearSegmentedColormap.from_list(name + '_r', data[::-1])
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

    def __init__(self, data, order, coordsys='C', header=None):
        # Header is optional
        if header is None:
            header = {}

        # Check types
        if not isinstance(data, np.ndarray):
            raise TypeError('data should be an array, use SkyMap.from_fits()')
        if not isinstance(header, dict):
            raise TypeError('header should be a dict')

        # Parse and store the skymap data
        self._save_data(data, order, coordsys)

        # Make sure the header cards are lowercase, and store
        header = {key.lower(): header[key] for key in header}
        self.header = header

        # Store the filename
        if 'filename' in header:
            self.filename = header['filename']
            if self.filename and self.filename.startswith('http'):
                header['url'] = self.filename
        else:
            self.filename = None

        # Get object name, or create one if it isn't in the header
        if 'object' in header:
            self.object = header['object']
        elif self.filename:
            self.object = os.path.basename(self.filename).split('.')[0]
        else:
            self.object = 'unknown'
        self.objid = self.object

        # Store creation time
        self.date = Time.now()
        self.mjd = self.date.mjd

        # Store event time, if there is one in the header
        if 'date-obs' in header:
            self.date_det = Time(header['date-obs'])
        else:
            self.date_det = self.date
        self.mjd_det = self.date_det.mjd

    def __eq__(self, other):
        try:
            if len(self.data) != len(other.data):
                return False
            return (np.all(self.data == other.data) and
                    self.order == other.order and
                    self.coordsys == other.coordsys)
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

        new_data = result.data * other_copy.data
        result._save_data(new_data)

        return result

    def __repr__(self):
        template = ('SkyMap(nside={}, order={}, coordsys={}, object={})')
        return template.format(self.nside, self.order, self.coordsys, self.object)

    def _pix2coord(self, pix):
        """Convert HEALpy pixel indexes to SkyCoords."""
        return pix2coord(self.nside, pix, nest=self.is_nested)

    def _coord2pix(self, coord):
        """Convert SkyCoords to HEALpy pixel indexes."""
        return coord2pix(self.nside, coord, nest=self.is_nested)

    def _save_data(self, data, order=None, coordsys=None):
        """Save the skymap data and add attributes."""
        self.data = data
        self.skymap = data  # Backwards compatability
        self.npix = len(data)
        self.nside = hp.npix2nside(self.npix)
        self.pixel_area = hp.nside2pixarea(self.nside, degrees=True)
        if order:
            self.order = order
            self.is_nested = order == 'NESTED'
            self.isnested = self.is_nested  # Backwards compatability
        if coordsys:
            self.coordsys = coordsys

        # Save the coordinates of each pixel
        self.ipix = np.arange(self.npix)
        self.coords = self._pix2coord(self.ipix)

        # Calculate the probability contours
        self._get_contours()

    def _get_contours(self):
        """Calculate the minimum contour level of each pixel.

        This is done using the cumulative sum method, (vaguely) based on code from
        http://www.virgo-gw.eu/skymap.html

        For example, consider a very small skymap with the following table:

        ipix | prob
           1 |  0.1
           2 |  0.4
           3 |  0.2
           4 |  0.3

        Sort by probability, and find the cumulative sum:

        ipix | prob | cumsum(prob)
           2 |  0.4 |  0.4
           4 |  0.3 |  0.7
           3 |  0.2 |  0.9
           1 |  0.1 |  1.0

        Now shift so the cumprob starts at zero, and that's the minimum contour level that each
        pixel is within.

        ipix | prob | contour
           2 |  0.4 |  0.0
           4 |  0.3 |  0.4
           3 |  0.2 |  0.7
           1 |  0.1 |  0.9

        Consider asking for the minimum number of pixels to cover increasing contour levels:
            -  0%-40%: you only need pixel 2
            - 40%-70%: you need pixels 2 & 3
            - 70%-90%: you need pixels 2, 3 & 4
            - 90%+:    you need all four pixels
        That's why we shift everything up so the first pixel has a contour value of 0%, because you
        should always include at least one pixel to cover the smallest contour levels.

        If we sort back to the original order the minimum confidence region each
        pixel is in is easy to find by seeing if contour(pixel) < percentage:

        ipix | prob | contour | in 90%? | in 50%?
           1 |  0.1 |  0.9    | False   | False
           2 |  0.4 |  0.0    | True    | True
           3 |  0.2 |  0.7    | True    | False
           4 |  0.3 |  0.4    | True    | True

        If you select the pixels for which contour(pixel) < percentage you will always cover
        AT LEAST percentage (you may well cover more of course).

        See also:
            SkyMap._pixels_within_contour(percentage)
            SkyMap.get_contour(coord)
            SkyMap.within_contour(coord, percentage)
        """
        # Get the indices sorted by probability (reversed, so highest first)
        # Note what we call the 'pixels' are really just the index numbers of the data array (ipix),
        # i.e. probability of pixel X = self.data[X]
        sorted_ipix = self.data.argsort()[::-1]

        # Sort the data using this mapping
        sorted_data = self.data[sorted_ipix]

        # Create cumulative sum array of each pixel in the array
        cumprob = np.cumsum(sorted_data)

        # Shift so we start at 0
        sorted_contours = np.append([0], cumprob[:-1])

        # "Un-sort" the contour array back to the normal pixel order and save
        self.contours = sorted_contours[sorted_ipix.argsort()]

    def _pixels_within_contour(self, percentage):
        """Find pixel indices confined in a given percentage contour (range 0-1)."""

        if not 0 <= percentage <= 1:
            raise ValueError('Percentage must be in range 0-1')

        # Find the pixels within the given contour
        mask = self.contours < percentage

        return self.ipix[mask]

    @classmethod
    def from_fits(cls, fits_file, coordsys='C'):
        """Initialize a `~gototile.skymap.SkyMap` object from a FITS file.

        Parameters
        ----------
        fits_file : str, `astropy.io.fits.HDU` or `astropy.io.fits.HDUList`
            Path to the FITS file (if str) or FITS HDU, passed to `astropy.io.fits.open`.

        coordsys : str, default='C'
            The coordinate system the data uses.
            'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)
            Used as a fallback if 'COORDSYS' is not defined in the FITS header.

        Returns
        -------
        `~gototile.skymap.SkyMap``
            SkyMap object.
        """
        # Load the data and header
        data, header = hp.read_map(fits_file,
                                   h=True,
                                   field=None,
                                   nest=None,
                                   verbose=False,
                                   dtype=None,
                                   )

        # Convert header to dict
        header = dict(header)

        # Some skymaps have multiple components
        # e.g. newer 3D skymaps from LVC (prob, distmu, distsigma, distnorm)
        # We can't deal with them, just take the first map (e.g. probability)
        if header['TFIELDS'] > 1:
            data = data[0]
            # Remove other column info from the header, so we don't get confused
            # The keys follow the pattern T---i, e.g. TFORM1, TTYPE1, TUNIT1
            keys = [k[:-1] for k in header if (k[0] == 'T' and k[-1] == '1')]
            for key in keys:
                for i in range(2, header['TFIELDS'] + 1):
                    del header[key + str(i)]
        del header['TFIELDS']

        # Get primary properties from header
        nside = header['NSIDE']
        if nside != hp.npix2nside(len(data)):
            raise ValueError("NSIDE from header ({}) doesn't match skymap length".format(nside))
        del header['NSIDE']

        order = header['ORDERING'].upper()
        if order not in ('NESTED', 'RING'):
            raise ValueError('ORDERING card in header has unknown value: {}'.format(order))
        del header['ORDERING']

        if 'COORDSYS' in header:
            coordsys = header['COORDSYS'][0].upper()
            del header['COORDSYS']
        if coordsys not in ('G', 'E', 'C'):
            raise ValueError('COORDSYS card in header has unknown value: {}'.format(coordsys))

        # Delete a load more keys that are unnecessary to save
        for key in ['BITPIX', 'EXTNAME', 'FIRSTPIX', 'GCOUNT', 'INDXSCHM',
                    'LASTPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'PCOUNT',
                    'PIXTYPE', 'TFIELDS', 'XTENSION',
                    ]:
            if key in header:
                del header[key]

        # Store the file name if the header was from a file
        if isinstance(fits_file, str):
            header['FILENAME'] = fits_file

        return cls(data, order, coordsys, header)

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
            hp.npix2nside(len(data))
        except ValueError:
            raise ValueError('Length of data is invalid')

        order = 'NESTED' if nested else 'RING'
        coordsys = coordsys[0]

        return cls(data, order, coordsys)

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
        # Create an Astropy SkyCoord at the peak
        peak = SkyCoord(ra, dec, unit='deg')

        # Get the gaussian probability data
        prob_map = create_gaussian_map(peak, radius, nside, nest=True)

        # Create a new SkyMap
        return cls.from_data(prob_map)

    def save(self, filename, overwrite=True):
        """Save the SkyMap as a FITS file.

        Parameters
        ----------
        filename : str
            The file to save the SkyMap as.

        overwrite : bool, optional
            If True, existing file is silently overwritten.
            Otherwise trying to write an existing file raises an OSError.
        """
        warnings.filterwarnings('ignore', category=VerifyWarning)
        hp.write_map(filename, [self.data],
                     nest=self.is_nested,
                     coord=self.coordsys,
                     column_names=[self.header['ttype1']],
                     extra_header=[(k.upper(), self.header[k]) for k in self.header],
                     overwrite=overwrite,
                     )

    def copy(self):
        """Return a new instance containing a copy of the sky map data."""
        newmap = SkyMap(self.data.copy(),
                        self.order,
                        self.coordsys,
                        self.header.copy())
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
        new_data = hp.ud_grade(self.data, nside_out=nside,
                               order_in=self.order, order_out=order,
                               power=power, pess=pess, dtype=dtype)

        # Save the new data
        self._save_data(new_data, order=order)

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

        rotator = hp.Rotator(coord=(self.coordsys, coordsys))

        # NOTE: rotator expects order=RING in and returns order=RING out
        # If this skymap is NESTED we need to regrade before and after
        if self.order == 'NESTED':
            in_data = hp.ud_grade(self.data, nside_out=self.nside,
                                  order_in='NESTED', order_out='RING')
        else:
            in_data = self.data.copy()

        # Rotate the skymap, now we're sure it's in RING order
        out_data = rotator.rotate_map(in_data)

        # Convert back to NESTED if needed
        if self.order == 'NESTED':
            out_data = hp.ud_grade(out_data, nside_out=self.nside,
                                   order_in='RING', order_out='NESTED')

        # Save the new data
        self._save_data(out_data, coordsys=coordsys)

    def normalise(self):
        """Normalise the sky map so the probability sums to unity."""
        norm_data = self.data / self.data.sum()

        # Save the new data
        self._save_data(norm_data)

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
            ipix = np.where(sep == (min(sep)))[0][0]
            prob = self.data[ipix]
        else:
            # Find all the pixels within the radius and sum them (more useful)
            ipix = np.where(sep < radius)[0]
            prob = self.data[ipix].sum()

        return prob

    def get_contour(self, coord):
        """Return the lowest probability contor the given sky coordinate is within.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            The point to find the probability at.
        """
        # Get the pixel that the coordinates are within
        ipix = self._coord2pix(coord)

        return self.contours[ipix]

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

    def get_contour_area(self, percentage):
        """Return the area of a given probability contour area, in square degrees.

        Parameters
        ----------
        percentage : float
            The confidence level, percentage in the range 0-1.
        """
        if not 0 <= percentage <= 1:
            raise ValueError('Percentage must be in range 0-1')

        # Get pixels within that contour
        ipix = self._pixels_within_contour(percentage)

        return len(ipix) * self.pixel_area

    def get_table(self):
        """Return an astropy QTable containing infomation on the skymap pixels."""
        col_names = ['ipix', 'ra', 'dec', 'prob']
        col_types = ['U', u.deg, u.deg, 'f8']

        table = QTable([self.ipix, self.coords.ra, self.coords.dec, self.data],
                       names=col_names, dtype=col_types)
        return table

    def plot(self, title=None, filename=None, dpi=90, figsize=(8, 6),
             plot_type='mollweide', center=(0, 45), radius=10,
             coordinates=None, plot_contours=True):
        """Plot the skymap.

        Parameters
        ----------
        title : str, optional
            title to show above the plot
            if not given a default title will be applied with the name of the grid

        filename : str, optional
            filename to save the plot to
            if not given then the plot will be displayed with plt.show()

        dpi : int, optional
            DPI to display the plot at
            default is 90

        figsize : 2-tuple, optional
            size of the matplotlib figure
            default is (8,6) - matching the GraceDB plots

        plot_type : str, one of 'mollweide', 'globe' or 'zoom', default = 'mollweide'
            type of axes to plot on
            if 'globe' the orthographic plot will be centred on `centre`
            if 'zoom' the plot will be centred on `centre` and have a radius of `radius`

        center : tuple or `astropy.coordinates.SkyCoord`, default (0,45)
            coordinates to center either a globe or zoom plot on
            if given as a tuple units will be considered to be degrees

        radius : float, default 10
            size of the zoomed plot, in degrees
            apparently it can only be a square

        coordinates : `astropy.coordinates.SkyCoord`, optional
            any coordinates to also plot on the image

        plot_contours : bool, default = True
            plot the 50% and 90% contour areas

        """
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Can only plot in equatorial coordinates
        # If it's not, temporarily rotate into equatorial and then go back afterwards
        if not self.coordsys == 'C':
            old_coordsys = self.coordsys
            self.rotate('C')
        else:
            old_coordsys = None

        if isinstance(center, tuple):
            center = SkyCoord(center[0], center[1], unit='deg')
        if isinstance(center, SkyCoord):
            center = center.to_string('hmsdms')

        if plot_type == 'mollweide':
            axes = plt.axes(projection='astro hours mollweide')
        elif plot_type == 'globe':
            axes = plt.axes(projection='astro globe', center=center)
        elif plot_type == 'zoom':
            axes = plt.axes(projection='astro zoom', center=center, radius=radius * u.deg)
        else:
            raise ValueError('"{}" is not a recognised plot type.')

        axes.grid()
        transform = axes.get_transform('world')

        # Plot the skymap data
        axes.imshow_hpx(self.data, cmap='cylon', nested=self.is_nested)

        # Plot 50% and 90% contours
        if plot_contours:
            cs = axes.contour_hpx(self.contours, nested=self.is_nested,
                                  levels=[0.5 * self.data.sum(),
                                          0.9 * self.data.sum()],
                                  colors='black', linewidths=0.5, zorder=99,)
            label_contours = False
            if label_contours:
                axes.clabel(cs, inline=False, fontsize=7, fmt='%.0f')

        # Plot coordinates if given
        if coordinates:
            axes.scatter(coordinates.ra.value, coordinates.dec.value,
                         transform=transform,
                         s=99, c='blue', marker='*', zorder=9)
            if coordinates.isscalar:
                coordinates = SkyCoord([coordinates])
            for coord in coordinates:
                axes.text(coord.ra.value, coord.dec.value,
                          coord.to_string('hmsdms').replace(' ', '\n') + '\n',
                          transform=transform,
                          ha='center', va='bottom',
                          size='x-small', zorder=12,
                          )

        # Remember to rotate back!
        if old_coordsys:
            self.rotate(old_coordsys)

        # Set title
        if title is None:
            title = 'Skymap for trigger {}'.format(self.objid)
        axes.set_title(title, y=1.05)

        # Save or show
        if filename:
            plt.savefig(filename, dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
