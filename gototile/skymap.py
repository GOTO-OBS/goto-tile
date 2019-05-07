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
from .catalog import read_catalog
from .gaussian import gaussian_skymap

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
        hdulist = gaussian_skymap(ra, dec, radius, nside)
        return cls.from_fits(hdulist)

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

    def normalise(self):
        """Normalise the sky map so the probability sums to unity."""
        total = self.skymap.sum()
        self.skymap /= total

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

    def plot(self, date=None, telescopes=None, pointings=None,
             objects=None, catalog=None, catcolor='#999999',
             nightsky=False, geoplot=False, contours=False,
             filename=None, title="", axes=None, dpi=300,
             options=None):
        """Plot the skymap in a Moll-Weide projection.

        Parameters
        ----------
        date : `~astropy.time.Time`, optional
            date to plot the skymap at
            default is the date of the detection from `SkyMap.date_det`

        telescopes : list of `~gototile.telescope.Telescope`, optional
            visible telescopes to plot

        pointings : list of pointings, optional
            tile pointings to plot
            needs `telescopes` to be >= 1

        objects : list, optional
            overplot an object, requires RA, Dec and object name

        catalog : dict, optional
            catalog of objects to overplot

        catcolor : str, optional
            color for catalog objects to be plotted
            default is #999999

        nightsky : bool, optional
            plot the night sky visibility of each telescope in `telescopes`
            only valid if `telescopes` and `catalog` is given
            default is False

        geoplot : bool, optional
            plot in geographic coordinates (lat, lon) instead of (RA, Dec)
            default is False

        contours : bool, optional
            plot 50% and 90% confidence regions
            default is False

        filename : str, optional
            filename to save the plot to
            if not given then the plot will be displayed with plt.show()

        title : str, optional
            title for the plot
            default is created based on object name and time observed

        axes : `matplotlib.pyplot.Axes`, optional
            axes to create the plot on
            default is none, new axes will be created

        dpi : int, optional
            DPI to save the plot at
            default is 300

        options : dict, optional
            various extra plotting options as keys, each with a `True` or `False` value
            all are False by default

            - moon : plot the moon position. The illumination is shown
                  between black (new moon) and white (full moon). Note
                  that the moon outline is always black.

            - sun : plot the sun position

            - coverage : show % probability covered by the thickness
                  of the tile outline. 1 percent is the normal
                  thickness.

            - delay : show the delay time as transparency of the tile.
                  0 hours is fully transparent, 24 hours is fully
                  opaque.

        """

        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as\
            FigureCanvas
        from matplotlib.figure import Figure
        from matplotlib.colors import colorConverter
        import cartopy.crs as ccrs
        import cartopy

        datadir = os.environ.get('CARTOPY_DATADIR')
        if datadir:
            cartopy.config['data_dir'] = datadir

        read_colormaps()

        if telescopes is None:
            telescopes = []
        if date is None:
            date = self.date_det
        if pointings is None:
            pointings = []
        if catalog is None:
            catalog = {'path': None, 'key': None}
        if options is None:
            options = {}
        if not title:
            title = "Skymap"
            if catalog['path']:
                if len(telescopes) > 0:
                    title += ", catalogue {}".format(os.path.basename(catalog['path']))
                else:
                    title += "and catalogue {}".format(os.path.basename(catalog['path']))
            if len(telescopes) > 0:
                telescope_names = ", ".join([telescope.name for telescope in telescopes])
                telescope_names = ' and '.join(telescope_names.rsplit(', ', 1))
                title += " and {} tiling".format(telescope_names)
            title += " for trigger {}".format(self.objid)
            title += "\n{}".format(Time(date).datetime.strftime("%Y-%m-%d %H:%M:%S"))

        sun = get_sun(date) if options.get('sun') else None
        moon = None
        if options.get('moon'):
            moon = ephem.Moon(date.iso)
            phase = moon.phase
            moon = SkyCoord(moon.ra/np.pi*180, moon.dec/np.pi*180, unit=u.deg)
            moon.phase = phase/100

        if axes is None:
            if filename:
                figure = Figure()
            else:
                figure = plt.figure()
            axes = figure.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        geodetic = ccrs.Geodetic()

        if geoplot:
            t = Time(date, location=('0d', '0d'))
            t.delta_ut1_utc = 0
            st = t.sidereal_time('mean')
            dlon = st.radian
            axes.coastlines(linewidth=0.25)
            axes.gridlines(linewidth=0.25, color='grey',
                           linestyle='--')
            axes.set_global()
            #m.nightshade(date=date.datetime, ax=axes)
            if len(telescopes) > 0:
                longs, lats = zip(*[(telescope.location.lon.deg,
                                telescope.location.lat.deg)
                                for telescope in telescopes])
                axes.plot(longs, lats, color='#BBBBBB', marker='8', #markersize=10,
                        linestyle='none', transform=geodetic)
        else:
            axes.set_global()

            dlon = 0  # longitude correction

        axes.scatter(self.coords.ra.value, self.coords.dec.value, s=1, c=self.skymap,
                     cmap='cylon', alpha=0.5, linewidths=0, zorder=1,
                     transform=geodetic)

        if contours:
            axes.tricontour(self.coords.ra.value, self.coords.dec.value, self.contours,
                            levels=[0.5,0.9],
                            colors='black', linewidths=0.5, zorder=99,
                            transform=geodetic)

        # Set up colorscheme for telescopes
        colors = itertools.cycle(
            ['#C44677', '#71CE48', '#7EB4BF', '#4F5C31', '#6670BC',
             '#BE5433', '#513447', '#C9BB46', '#80CD93', '#BC56C8',
             '#C6A586', '#C996BD'])
        colors = dict([(telescope.name, color)
                       for telescope, color in zip(telescopes, colors)])
        # Plot FoVs
        for i, pointing in enumerate(pointings):
            # use pointings.tilelist[i] instead of pointing['tile']
            # the latter has dtype 'object' (containg floats), the
            # former 'float64'
            ra, dec = smt.getshape(pointings.tilelist[i], steps=10)
            ra = ra - np.rad2deg(dlon)
            color = colors[pointing['telescope']]
            alpha = 0
            if options.get('delay'):
                # show delay time as transparency
                alpha = pointing['dt'].jd
                alpha = max(0, min(1, alpha))  # clip to 0 -- 1 range
            acolor = colorConverter.to_rgba(color, alpha=alpha)
            linewidth = 1
            if options.get('coverage'):
                # show % coverage as outline thickness
                linewidth = 100 * pointing['prob']
            ra2 = ra + 180
            # Work around an issue with cartopy-Proj.4, where polygons
            # aren't drawn >= abs(89) latitude. Since the M-W
            # projection is bad near the poles anyway, we can probably
            # safely cheat. See
            # https://github.com/SciTools/cartopy/issues/724
            dec = np.array(dec)
            dec[dec >= 88.99] = 88.99
            dec[dec <= -88.99] = -88.99
            if np.any(ra2 > 0) and np.any(ra2 <= 0):
                mask = ra2 > 0
                axes.fill(ra[mask], dec[mask],
                          fill=True, facecolor=acolor,
                          linewidth=linewidth, linestyle='solid',
                          edgecolor='black', transform=geodetic)
                mask = ~mask
                axes.fill(ra[mask], dec[mask],
                          fill=True, facecolor=acolor,
                          linewidth=linewidth, linestyle='solid',
                          edgecolor='black', transform=geodetic)
            else:
                axes.fill(ra, dec,
                          fill=True, facecolor=acolor,
                          linewidth=linewidth, linestyle='solid',
                          edgecolor='black',
                          transform=geodetic)

        if catalog['path']:
            logging.info("Reading catalog data for plot")
            table = read_catalog(**catalog)

            ras = table['ra']
            decs = table['dec']

            if nightsky:
                sidtimes = []
                for telescope in telescopes:
                    visras, visdecs = [], []
                    sidtimes.append(smt.calc_siderealtimes(
                        date, telescope.location))
                sidtimes = np.hstack(sidtimes)
                radius = 75
                logging.info("Calculating night sky coverage for %d "
                             "points in time", len(sidtimes))
                for st in sidtimes:
                    frame = AltAz(obstime=st, location=telescope.location)
                    radecs = SkyCoord(ras, decs, unit=u.deg)
                    altaz = radecs.transform_to(frame)
                    visras.extend(ras[np.where(altaz.alt.degree > (90-radius))])
                    visdecs.extend(decs[np.where(altaz.alt.degree > (90-radius))])
                xcat, ycat = np.array(visras) - np.rad2deg(dlon), visdecs
            else:
                xcat, ycat = np.array(ras) - np.rad2deg(dlon), decs
            logging.info("Overplotting catalog")
            axes.scatter(xcat, ycat, s=0.5, c=catcolor, alpha=0.5, linewidths=0,
                         transform=geodetic)

        if objects:
            ra = [obj.ra.value - np.rad2deg(dlon) for obj in objects]
            dec = [obj.dec.value for obj in objects]
            axes.plot(ra, dec, linestyle='None', marker='p',
                      color=(0, 1, 1, 0.5), zorder=5,
                      transform=geodetic)
            for obj, xpos, ypos in zip(objects, ra, dec):
                axes.text(xpos, ypos, obj.name, ha='center', va='top',
                          size='x-small', zorder=12,
                          transform=geodetic)
        if sun:
            axes.plot(sun.ra.value-np.rad2deg(dlon), sun.dec.value,
                      color=(1, 1, 0, 0.5), marker='o',
                      markerfacecolor=(1, 1, 0, 0.5), markersize=12,
                      transform=geodetic)
        if moon:
            phase = moon.phase
            axes.plot(moon.ra.value, moon.dec.value,
                      marker='o', markersize=10, markeredgecolor='black',
                      markerfacecolor=(phase, phase, phase, 0.5),
                      transform=geodetic)
        axes.set_title(title, y=1.05)

        if filename:
            canvas = FigureCanvas(figure)
            canvas.print_figure(filename, dpi=dpi)
        else:
            plt.show()
