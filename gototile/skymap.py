from __future__ import division

import os
import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import colorConverter
from mpl_toolkits.basemap import Basemap
import astropy
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord, AltAz
from astropy import units
import healpy
import ephem
from . import galtools as gt
from . import skymaptools as smt
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
        self.objid = header['objid']

    def copy(self):
        return SkyMap(skymap=self.skymap.copy(),
                      header=self.header.copy())

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
        objid = header.get('object', objid)
        header['objid'] = objid.split(':')[-1]
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
        skymap.

        """

        npix = len(self.skymap)
        ipix = np.arange(npix)
        theta, phi = healpy.pix2ang(self.nside, ipix, nest=self.isnested)
        skycoords = SkyCoord(ra=phi*units.rad, dec=(0.5*np.pi - theta)*units.rad)
        return skycoords

    def plot(self, filename, telescopes, date, pointings,
             geoplot=False, galaxies=False, nightsky=False,
             title="", objects=None, sun=False, moon=False,
             galcolor='#999999', dpi=300):
        """Plot the skymap in a Moll-Weide projection"""

        if not title:
            formatted_date = Time(date).datetime.strftime("%Y-%m-%d %H:%M:%S")
            telescope_names = ", ".join([telescope.name
                                           for telescope in telescopes])
            telescope_names = ' and '.join(telescope_names.rsplit(', ', 1))
            if galaxies:
                title = ("Skymap, GWGC galaxies and {0} tiling for trigger {1}\n"
                         "{2}".format(telescope_names,
                                      self.objid, formatted_date))
            else:
                title = "Skymap and {0} tiling for trigger {1}\n{2}".format(
                    telescope_names, self.objid, formatted_date)
        if sun:
            sun = get_sun(date)
        if moon:
            moon = ephem.Moon(date.iso)
            phase = moon.phase
            moon = SkyCoord(moon.ra/np.pi*180, moon.dec/np.pi*180,
                            unit=units.degree)
            moon.phase = phase/100

        figure = Figure()
        axes = figure.add_subplot(1, 1, 1)
        m = Basemap(projection='moll', resolution='c', lon_0=0.0, ax=axes)
        m.drawmeridians(np.arange(0, 360, 30), linewidth=0.25)
        m.drawparallels(np.arange(-90, 90, 30), linewidth=0.25, labels=[1,0,0,0])
        m.drawmapboundary(color='k', linewidth=0.5)

        if geoplot:
            t = Time(date, location=('0d', '0d'))
            t.delta_ut1_utc = 0
            st = t.sidereal_time('mean')
            dlon = st.radian
            m.drawcoastlines(linewidth=0.25)
            m.nightshade(date=date.datetime)
            longs, lats = zip(*[(telescope.location.longitude.deg,
                               telescope.location.latitude.deg)
                              for telescope in telescopes])
            x, y = m(longs, lats)
            m.plot(x, y, color='#BBBBBB', marker='8', markersize=10,
                   linestyle='none')
        else:
            dlon = 0  # longitude correction

        npix = healpy.nside2npix(self.nside)
        ipix = np.arange(npix)
        thetas, phis = healpy.pix2ang(self.nside, ipix, nest=self.isnested)
        ras, decs = smt.sph2cel(thetas, phis-dlon)
        xmap, ymap = m(ras, decs)
        m.scatter(xmap, ymap, s=1, c=self.skymap,
                  cmap='cylon', alpha=0.5, linewidths=0, zorder=1)

        # Set up colorscheme for telescopes
        colors = itertools.cycle(
            ['#C44677', '#71CE48', '#7EB4BF', '#4F5C31', '#6670BC',
             '#BE5433', '#513447', '#C9BB46', '#80CD93', '#BC56C8',
             '#C6A586', '#C996BD'])
        colors = dict([(telescope.name, color)
                       for telescope, color in zip(telescopes, colors)])
        # Plot FoVs
        for pointing in pointings:
            ra, dec = smt.getshape(pointing['tile'])
            ra2 = ra - dlon / np.pi * 180
            x, y = m(ra2, dec)
            color = colors[pointing['telescope']]
            alpha = 1 - pointing['dt'].jd / 0.5
            alpha = max(0, min(1, alpha))
            acolor = colorConverter.to_rgba(color, alpha=alpha)
            if np.any(ra2 >= 180) and np.any(ra2 <= 180):
                mask = ra2 > 180
                axes.fill(x[mask], y[mask],
                          fill=True, facecolor=acolor,
                          linewidth=100 * pointing['prob'], linestyle='solid',
                          edgecolor='black')
                mask = ra2 <= 180
                axes.fill(x[mask], y[mask],
                          fill=True, facecolor=acolor,
                          linewidth=100 * pointing['prob'], linestyle='solid',
                          edgecolor='black')
            else:
                axes.fill(x, y,
                          fill=True, facecolor=acolor,
                          linewidth=100 * pointing['prob'], linestyle='solid',
                          edgecolor='black')
            #m.plot(x, y, color=color, linewidth=100 * pointing['prob'])

        if galaxies:
            gals = gt.readgals_new()

            ras = gals['ra']
            decs = gals['dec']

            if nightsky:
                visras, visdecs = [], []
                sidtimes = smt.calc_siderealtimes(date, telescope.location)
                radius = 75
                for st in sidtimes:
                    frame = AltAz(obstime=st, location=telescope.location)
                    radecs = SkyCoord(ra=ras*units.deg, dec=decs*units.deg)
                    altaz = radecs.transform_to(frame)
                    visras.extend(ras[np.where(altaz.alt.degree > (90-radius))])
                    visdecs.extend(decs[np.where(altaz.alt.degree > (90-radius))])
                xgal, ygal = m(np.array(visras) - (dlon/np.pi*180.0), visdecs)

            else:
                xgal, ygal = m(np.array(ras) - (dlon/np.pi*180.0), decs)
            m.scatter(xgal, ygal, s=0.5, c=galcolor, alpha=0.5, linewidths=0,
                      zorder=2)

        if objects:
            ra = [obj.ra.value for obj in objects]
            dec = [obj.dec.value for obj in objects]
            x, y = m(np.array(ra) - (dlon / np.pi*180), np.array(dec))
            m.plot(x, y, linestyle='None', marker='p', color=(0, 1, 1, 0.5),
                    zorder=5)
            for obj, xpos, ypos in zip(objects, x, y):
                plt.annotate(obj.name, xy=(xpos, ypos), xytext=(xpos, ypos),
                             ha='center', va='top', size='x-small',
                             zorder=12)
        if sun:
            x, y = m(np.array(sun.ra.value) - (dlon / np.pi*180), sun.dec.value)
            m.plot(x, y, color=(1, 1, 0, 0.5), marker='o',
                   markerfacecolor=(1, 1, 0, 0.5), markersize=12)
        if moon:
            phase = moon.phase
            x, y = m(np.array(moon.ra.value) - (dlon / np.pi*180), moon.dec.value)
            m.plot(x, y, marker='o', markersize=10, markeredgecolor='black',
                   markerfacecolor=(phase, phase, phase, 0.5))
        axes.set_title(title)

        canvas = FigureCanvas(figure)
        canvas.print_figure(filename, dpi=dpi)
