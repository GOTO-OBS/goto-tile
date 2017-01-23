from __future__ import division

import os
import itertools
import logging
import numpy as np
import astropy
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord, AltAz
from astropy import units
import healpy
import ephem
from . import settings
from . import skymaptools as smt
from .catalog import read_catalog
try:
    stringtype = basestring  # Python 2
except NameError:
    stringtype = str  # Python 3


def read_colormaps(name='cylon'):
    """Read special color maps, such as 'cylon'"""
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap
    filename = os.path.join(os.path.dirname(__file__), name + '.csv')
    data = np.loadtxt(filename, delimiter=',')
    cmap = LinearSegmentedColormap.from_list(name, data)
    cm.register_cmap(cmap=cmap)
    cmap = LinearSegmentedColormap.from_list(name+'_r', data[::-1])
    cm.register_cmap(cmap=cmap)


class SkyMap(object):
    """A probability skymap

    The SkyMap is a wrapper around the healpy skymap numpy.array,
    returned by healpy.fitsfunc.read_map. The SkyMap class holds track
    of the numpy array, the header information and some options.

    """

    def __init__(self, skymap, header=None, **kwargs):
        if isinstance(skymap, stringtype):
            skymap, header = self._read_file(skymap)
        elif header is None:
            header = {}
        elif not isinstance(header, dict):
            raise TypeError("header should be a dict")
        self.object = header.get('object')
        self.order = header.get('order')
        self.nside = header.get('nside')
        self.isnested = header.get('nested')
        if not self.order:
            self.order = 'NESTED' if self.isnested else 'RING'
        dtype = getattr(settings, 'DTYPE')
        self.skymap = skymap.astype(dtype)
        self.objid = header.get('objid')
        self.header = header

    def copy(self):
        newmap = SkyMap(skymap=self.skymap.copy())
        newmap.object = self.object
        newmap.order = self.order
        newmap.isnested = self.isnested
        newmap.nside = self.nside
        newmap.objid = self.objid
        return newmap

    def _read_file(self, filename):
        info = healpy.read_map(filename, h=True, field=None,
                               verbose=False, nest=None)
        try:
            skymap, distmu, distsigma, distnorm, header = info
        except ValueError as exc:
            if "not enough values to unpack" in str(exc):
                # assume an older map without distance information
                skymap, header = info
            else:
                raise
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
        objid = header.get('object', objid)
        header['objid'] = objid.split(':')[-1]
        header['url'] = header.get('referenc', '')

        header['mjd'] = astropy.time.Time.now().mjd
        header['date'] = astropy.time.Time(float(header['mjd']), format='mjd')
        header['mjddet'] = header.get(
            'mjd-obs', astropy.time.Time(header['date']).mjd)
        header['date-det'] = astropy.time.Time(float(header['mjddet']),
                                               format='mjd')

        header['nside'] = header.get('nside', healpy.npix2nside(len(skymap)))
        return skymap, header

    def regrade(self, nside=None, order='NESTED', power=-2, pess=False,
                dtype=None):
        """Up- or downgrade the skymap resolution.

        See the healpy.pixelfunc.ud_grade documentation about the options.

        """

        if nside == self.nside and order == self.order:
            return
        self.skymap = healpy.ud_grade(self.skymap, nside_out=nside,
                                      order_in=self.order, order_out=order,
                                      power=power, pess=pess, dtype=dtype)
        self.nside = nside
        self.order = order
        self.isnested = order == 'NESTED'

    def skycoords(self):
        """Return the sky coordinates (RA, Dec) for the current map.

        The returned value is an astropy.coordinates.SkyCoord object,
        with the number of coordinates equal to the size of the
        skymap.

        """

        npix = len(self.skymap)
        ipix = np.arange(npix)
        theta, phi = healpy.pix2ang(self.nside, ipix, nest=self.isnested)
        skycoords = SkyCoord(ra=phi*units.rad, dec=(0.5*np.pi - theta)*units.rad)
        return skycoords

    def plot(self, filename, telescopes, date, pointings,
             geoplot=False, catalog=None, nightsky=False,
             title="", objects=None,
             catcolor='#999999', dpi=300, options=None,
             axes=None):
        """Plot the skymap in a Moll-Weide projection


        Parameters
        ----------

        - options : dict

            Various extra plotting options.

            ``options`` takes various keys, each with a ``True`` or
            ``False`` value. If a key does not exist in options, it
            equals ``False``.

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

        read_colormaps()
        if catalog is None:
            catalog = {'path': None, 'key': None}
        if options is None:
            options = {}
        if not title:
            formatted_date = Time(date).datetime.strftime("%Y-%m-%d %H:%M:%S")
            telescope_names = ", ".join([telescope.name
                                           for telescope in telescopes])
            telescope_names = ' and '.join(telescope_names.rsplit(', ', 1))
            if catalog['path']:
                title = ("Skymap, catalog {catalog} and {telescope} tiling "
                         "for trigger {trigger}\n{formatted_date}".format(
                             catalog=os.path.basename(catalog['path']),
                             telescope=telescope_names, trigger=self.objid,
                             formatted_date=formatted_date))
            else:
                title = ("Skymap and {telescope} tiling "
                         "for trigger {trigger}\n"
                         "{formatted_date}".format(
                             telescope=telescope_names, trigger=self.objid,
                             formatted_date=formatted_date))
        sun = get_sun(date) if options.get('sun') else None
        moon = None
        if options.get('moon'):
            moon = ephem.Moon(date.iso)
            phase = moon.phase
            moon = SkyCoord(moon.ra/np.pi*180, moon.dec/np.pi*180,
                            unit=units.degree)
            moon.phase = phase/100

        if axes is None:
            figure = Figure()
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
            longs, lats = zip(*[(telescope.location.longitude.deg,
                               telescope.location.latitude.deg)
                              for telescope in telescopes])
            axes.plot(longs, lats, color='#BBBBBB', marker='8', markersize=10,
                      linestyle='none', transform=geodetic)
        else:
            axes.set_global()

            dlon = 0  # longitude correction

        npix = healpy.nside2npix(self.nside)
        ipix = np.arange(npix)
        thetas, phis = healpy.pix2ang(self.nside, ipix, nest=self.isnested)
        ras = np.rad2deg(phis-dlon)%360
        decs = np.rad2deg(np.pi/2 - thetas%np.pi)
        axes.scatter(ras, decs, s=1, c=self.skymap,
                     cmap='cylon', alpha=0.5, linewidths=0, zorder=1,
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
                    radecs = SkyCoord(ra=ras*units.deg, dec=decs*units.deg)
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
