"""Module containing the SkyMap class."""

import os
import warnings

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.table import QTable

import healpy as hp

import ligo.skymap.plot  # noqa: F401  (for extra projections)

from matplotlib import pyplot as plt
if 'DISPLAY' not in os.environ:
    plt.switch_backend('agg')

import mhealpy as mhp

import numpy as np

from . import settings
from .gaussian import create_gaussian_map
from .skymaptools import coord2pix, pix2coord, get_data_contours


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

    This class is a wrapper around the `~mhealpy.HealpixMap` class, which supports MOC skymaps.

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
    def __init__(self, data, order, coordsys='C', header=None, uniq=None, density=False):
        # Header is optional
        if header is None:
            header = {}

        # Check types
        if not isinstance(data, np.ndarray):
            raise TypeError('Skymap data should be an array, use SkyMap.from_fits() to load files')
        if order not in ['NESTED', 'RING', 'NUNIQ']:
            raise ValueError(f'Unrecognised HEALPix order: "{order}"')
        if coordsys not in ('G', 'E', 'C'):
            raise ValueError(f'Unrecognised coordinate system: "{coordsys}"')
        if uniq is not None and len(data) != len(uniq):
            raise ValueError(f'UNIQ pixel indices (n={len(uniq)} do not match data (n={len(data)})')
        if not isinstance(header, dict):
            raise TypeError('Header should be a dict')

        # Parse and store the data
        self._save_data(data, order, coordsys, uniq, density)

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
        if self.density != other_copy.density:
            other_copy.density = self.density

        new_data = result.data * other_copy.data
        result._save_data(new_data, order=self.order, coordsys=self.coordsys, density=self.density)

        return result

    def __repr__(self):
        template = ('SkyMap(nside={}, order={}, coordsys={}, density={})')
        return template.format(self.nside, self.order, self.coordsys, self.density)

    def _pix2coord(self, ipix):
        """Convert HEALpy pixel indexes to SkyCoords."""
        if not self.is_moc:
            return pix2coord(self.nside, ipix, nest=self.is_nested)
        else:
            nside, nested_ipix = mhp.uniq2nest(self.uniq[ipix])
            return pix2coord(nside, nested_ipix, nest=True)

    def _coord2pix(self, coord):
        """Convert SkyCoords to HEALpy pixel indexes."""
        if not self.is_moc:
            return coord2pix(self.nside, coord, nest=self.is_nested)
        else:
            nested_ipix = coord2pix(self.nside, coord, nest=True)
            return self.healpix.nest2pix(nested_ipix)

    def _save_data(self, data, order=None, coordsys=None, uniq=None, density=None):
        """Save the skymap data and add attributes."""
        # Create mhealpy HEALPix class (we access most properties from here)
        self.healpix = mhp.HealpixMap(data, uniq,
                                      scheme=order,
                                      density=density,
                                      )

        # Save coordsys (not considered by mhealpy)
        self.coordsys = coordsys

        # Save pixel indices, Nside values and areas
        self.ipix = np.arange(self.npix)
        if self.is_moc:
            self.pix_nside = np.array([2 ** np.floor(np.log2(u / 4) / 2) for u in uniq], dtype=int)
        else:
            self.pix_nside = np.array([self.nside] * self.npix, dtype=int)
        self.pix_area = 4 * np.pi / (12 * np.array(self.pix_nside) ** 2)

        # Find the coordinates of each pixel
        self.coords = self._pix2coord(self.ipix)

        # Calculate the pixel contour levels
        self.contours = get_data_contours(self.data)

    @property
    def data(self):
        return self.healpix.data

    @property
    def skymap(self):
        # backwards-compatable equivalent of SkyMap.data
        return self.healpix.data

    @property
    def nside(self):
        return self.healpix.nside

    @property
    def npix(self):
        return self.healpix.npix

    @property
    def uniq(self):
        return self.healpix.uniq

    @property
    def order(self):
        return self.healpix.scheme

    @property
    def is_nested(self):
        return self.healpix.is_nested

    @property
    def isnested(self):
        # backwards-compatable equivalent of SkyMap.is_nested
        return self.healpix.is_nested

    @property
    def is_moc(self):
        return self.healpix.is_moc

    @property
    def density(self):
        return self.healpix.density()

    @density.setter
    def density(self, to_density):
        """Convert between histogram (counts) and density (per steradian) maps."""
        if not self.density and to_density:
            # Convert the histogram map to density per base pixel
            data = self.data / self.pix_area
        elif self.density and not to_density:
            # Convert the density map (per steradian) to a histogram
            data = self.data * self.pix_area
        # Save the data
        self._save_data(data, self.order, self.coordsys, self.uniq, to_density)

    @classmethod
    def from_fits(cls, fits_file, coordsys='C', hdu=1, data_field=None, density=None):
        """Initialize a `~gototile.skymap.SkyMap` object from a FITS file.

        Parameters
        ----------
        fits_file : str, `astropy.io.fits.HDU` or `astropy.io.fits.HDUList`
            Path to the FITS file (if str) or FITS HDU, passed to `astropy.io.fits.open`.

        coordsys : str, default='C'
            The coordinate system the data uses.
            'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)
            Used as a fallback if 'COORDSYS' is not defined in the FITS header.
        hdu : int, default=1
            HDU number to load data from, if given a file or HDUList with multiple headers.
        data_field : int, default=None
            Field number to read the skymap data from.
            If not given data will be read from field 0 by default, UNLESS the first field
            is labled 'UNIQ' in which case it's a multi-order map and the data should be in
            field 1.
        density : bool or None, default=None
            Is the skymap data given in individual counts per pixel (histogram) or as a density
            (per steradian)?
            If not given the data type will be assumed based on the "unit" label for the data
            column, if any, and otherwise will default to `False`.

        Returns
        -------
        `~gototile.skymap.SkyMap``
            SkyMap object.
        """
        # If a path has been given then open the file
        file_open = False
        if isinstance(fits_file, str):
            hdu_list = fits.open(fits_file)
            hdu = hdu_list[hdu]
            file_open = True
        elif isinstance(fits_file, fits.hdu.HDUList):
            hdu = fits_file[hdu]
        else:
            hdu = fits_file
        header = dict(hdu.header)

        # Store the file name in the header
        if isinstance(fits_file, str):
            header['FILENAME'] = fits_file

        # Check that the file is valid HEALPix
        if ('PIXTYPE' not in header or header['PIXTYPE'].upper() != 'HEALPIX' or
                'ORDERING' not in header):
            raise ValueError('FITS file is not in a valid HEALPix format')
        order = header['ORDERING'].upper()

        # We also need the coordinate system from the header
        if 'COORDSYS' in header:
            coordsys = header['COORDSYS'][0].upper()

        # Load the skymap data
        if data_field is not None:
            data = hdu.data.field[data_field]
            if 'UNIQ' in hdu.data.columns:
                uniq = hdu.data['UNIQ']
            else:
                uniq = None
        elif hdu.data.columns[0].name == 'UNIQ':
            data_field = 1
            data = hdu.data.field(data_field)
            uniq = hdu.data.field(0)
        else:
            data_field = 0
            data = hdu.data.field(0)
            uniq = None

        # Check we could find UNIQ pixel indices
        if order == 'NUNIQ' and uniq is None:
            raise ValueError('Skymap order = "NUNIQ", but could not find UNIQ pixel indices')

        # Attempt to determine if the data is in density units
        if density is None:
            if 'DENSITY' in hdu.data.columns[data_field].name:
                density = True
            elif (hdu.data.columns[data_field].unit is not None and
                  ('/sr' in hdu.data.columns[data_field].unit or
                   'sr-1' in hdu.data.columns[data_field].unit)):
                density = True
            else:
                density = False

        # Remember to close the file
        if file_open:
            hdu_list.close()

        return cls(data, order, coordsys, header, uniq, density)

    @classmethod
    def from_data(cls, data, order, coordsys='C', density=False, uniq=None):
        """Initialize a `~gototile.skymap.SkyMap` object from an array of data.

        Parameters
        ----------
        data : list or `numpy.array`
            an array of data to map onto a HEALPix sphere
            the length of the data must match one of the valid HEALPix resolutions
        order : str
            The HEALPix ordering for the data, either 'RING', 'NESTED' or 'NUNIQ'

        coordsys : str, default=C
            The coordinate system the data uses.
            'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)
        density : bool, default=False
            Is the skymap data given in individual counts per pixel (histogram) or as a density
            (per steradian)?
        uniq : list of int or None, default=None
            UNIQ pixel indices for each pixel in `data`, only required if `order='NUNIQ'`.

        Returns
        -------
        `~gototile.skymap.SkyMap``
            SkyMap object.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # This method doesn't really do anything any more, but I suppose it might still be useful.
        return cls(data, order, coordsys, uniq=uniq, density=density)

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

        # Get the gaussian skymap data
        data = create_gaussian_map(peak, radius, nside, nest=True)

        # Create a new SkyMap
        return cls.from_data(data, order='NESTED')

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
        warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
        if self.header is not None:
            header = [(k.upper(), self.header[k]) for k in self.header]
        else:
            header = None
        self.healpix.write_map(filename,
                               coordsys=self.coordsys,
                               extra_header=header,
                               overwrite=overwrite,
                               )

    def copy(self):
        """Return a new instance containing a copy of the sky map data."""
        newmap = SkyMap(self.data.copy(),
                        self.order,
                        self.coordsys,
                        self.header.copy() if self.header is not None else None,
                        self.uniq,
                        self.density)
        return newmap

    def regrade(self, nside=None, order='NESTED'):
        """Up- or downgrade the sky map HEALPix resolution, or change the pixel ordering.

        Note this function can flatten multi-order skymaps (i.e. convert from NUNIQ to NESTED or
        RING ordering), but not the other way around (i.e. confert flat slymaps to multi-order).
        """
        if nside is None:
            nside = self.nside
        if order == 'NUNIQ':
            raise ValueError('Can not regrade to NUNIQ ordering')
        if order not in ['NESTED', 'RING']:
            raise ValueError(f'Unrecognised HEALPix order: "{order}"')
        if nside == self.nside and order == self.order:
            return

        # Convert the current skymap
        new_skymap = self.healpix.rasterize(nside, order)

        # Save the new data
        self._save_data(new_skymap.data, order=order, coordsys=self.coordsys, density=self.density)

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
        self._save_data(out_data, order=self.order, coordsys=coordsys, density=self.density)

    def normalise(self):
        """Normalise the sky map so the it sums to unity (e.g. for probability maps)."""
        norm_data = self.data / self.data.sum()

        # Save the new data
        self._save_data(norm_data, order=self.order, coordsys=self.coordsys, density=self.density)

    def get_value(self, coord, radius=0):
        """Return the value of the skymap at a given sky coordinate.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            Position coordinates.
        radius : float, optional
            If given, the radius in degrees of a circle to integrate the skymap within.
        """
        # Find distance to points
        sep = np.array(coord.separation(self.coords))

        if radius == 0:
            # Just get the radius of the nearest pixel (not very useful)
            ipix = np.where(sep == (min(sep)))[0][0]
            return self.data[ipix]
        else:
            # Find all the pixels within the radius and sum them (more useful)
            ipix = np.where(sep < radius)[0]
            return self.data[ipix].sum()

    def get_contour(self, coord):
        """Return the lowest contour level the given sky coordinate is within.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            Position coordinates.
        """
        # Get the pixel that the coordinates are within
        ipix = self._coord2pix(coord)

        return self.contours[ipix]

    def within_contour(self, coord, contour_level):
        """Find if the given position is within the area of the given contour level.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            Position coordinates.
        contour_level : float
            The contour level to query.
        """
        contour = self.get_contour(coord)

        return contour < contour_level

    @property
    def pixel_area(self):
        """Return the area of each pixel (only valid for non-NUNIQ skymaps) in square degrees."""
        if self.order != 'NUNIQ':
            return 4 * np.pi / (12 * np.array(self.pix_nside) ** 2) * (180 / np.pi) ** 2
        else:
            raise ValueError('NUNIQ maps have variable pixel areas.')

    def get_pixel_areas(self, ipix):
        """Return the areas covered by each of the given skymap pixels in square degrees.

        This is only really useful for NUNIQ skymaps, where the size of each pixel can vary.
        For non-NUNIQ skymaps the area of every pixel is the same (given by the `SkyMap.pixel_area`)
        property), so all the values returned by this function will be the same.

        See also `SkyMap.get_pixel_area()`

        Parameters
        ----------
        ipix : int or list of int
            Pixel index, or multiple indices.

        Returns
        -------
        areas : int, or array of int
            The areas covered by the given pixel(s).

        """
        return self.pix_area(ipix) * (180 / np.pi) ** 2

    def get_pixel_area(self, ipix):
        """Return the TOTAL area covered by the given skymap pixels in square degrees.

        This is only really useful for NUNIQ skymaps, where the size of each pixel can vary.
        For non-NUNIQ skymaps the area of every pixel is the same (given by the `SkyMap.pixel_area`)
        property), so all this function returns is `SkyMap.pixel_area` * len(ipix).

        See also `SkyMap.get_pixel_areas()`

        Parameters
        ----------
        ipix : int or array of int
            Pixel index, or multiple indices.

        Returns
        -------
        area : int
            The total area covered by the given pixel(s).
        """
        areas = self.get_pixel_areas(ipix)
        if isinstance(areas, int):
            return areas * (180 / np.pi) ** 2
        else:
            return sum(areas) * (180 / np.pi) ** 2

    def get_contour_area(self, contour_level):
        """Return the area of a given contour region, in square degrees.

        Parameters
        ----------
        contour_level : float
            The contour level to query.
        """
        # Get pixels within that contour level
        ipix = self.ipix[self.contours < contour_level]

        # Return the area covered by those pixels
        return self.get_pixel_area(ipix)

    def get_table(self):
        """Return an astropy QTable containing infomation on the skymap pixels."""
        col_names = ['pixel', 'ra', 'dec', 'value', 'area']
        col_data = [self.ipix, self.coords.ra, self.coords.dec, self.data,
                    self.pix_area * 180 / np.pi * u.deg * u.deg]
        if self.is_moc:
            col_names += ['uniq', 'nside']
            col_data += [self.uniq, self.pix_nside]

        return QTable(col_data, names=col_names)

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
        if self.is_moc:
            data = self.healpix.rasterize(self.nside, 'NESTED').data
        else:
            data = self.data
        axes.imshow_hpx(data, cmap='cylon', nested=self.is_nested)

        # Plot 50% and 90% contours
        if plot_contours:
            if self.is_moc:
                contours = get_data_contours(data)
            else:
                contours = self.contours
            cs = axes.contour_hpx(contours, nested=self.is_nested,
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
