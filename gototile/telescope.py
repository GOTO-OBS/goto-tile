"""Module to handle telescopes and instrumentation

"""
from __future__ import division
import os
import gzip
import pickle
import logging
import yaml
import numpy as np
from astropy.coordinates import EarthLocation, Latitude, Longitude, get_sun
from astropy.coordinates import AltAz, SkyCoord
from astropy.units import Quantity, UnitConversionError
from astropy import units
from astropy.table import Table
import healpy
import ephem
from .catalog import visible_catalog, read_catalog, map2catalog
from .settings import NSIDE, TILESDIR, SUNALTITUDE, COVERAGE
from .grid import tileallsky
from .skymaptools import (calc_siderealtimes, get_visiblemap,
                          filltiles, ordertiles, getvectors, sph2cel)

try:  # Python 3
    FileNotFoundError
except NameError:  # Python 2
    from .utils import FileExistsError, FileNotFoundError


class Telescope(object):
    """Telescope and instrument class

    The telescope class keeps track of the position, field of view,
    name and minimum elevation of the telescope + instrument
    combination.

    Position is longitude Eastward, latitude and height above sea
    level.

    Minimum elevation is measured in degrees from the horizon.

    """
    def __init__(self, location, fov=None, name="", min_elevation=15):
        # Try to be flexible in what is accepted; this creates a
        # relatively long initialization
        self.name = ""
        self.min_elevation = min_elevation
        if not isinstance(self.min_elevation, Quantity):
            self.min_elevation *= units.degree
        if isinstance(location, Telescope):
            self.location = location.location.copy()
            self.name = location.name
            self.fov = location.fov.copy()
            self.min_elevation = location.min_elevation
            if fov:
                self.fov = self._parse_fov(fov)
        else:
            self.location = self._parse_location(location)
            self.fov = self._parse_fov(fov)
        if name:
            self.name = name

    def __str__(self):
        location = "{:s}, {:s}, {:.1f}".format(self.location.longitude,
                                               self.location.latitude,
                                               self.location.height)
        return "{:s} @ {:s} - [{:s}, {:s}]".format(
            self.name, location, self.fov['ra'], self.fov['dec'])

    def _parse_location(self, location):
        """Parse a location.

        Arguments
        =========

        - location: EarthLocation, 3-tuple or dict

            If location is an EarthLocation, it is used as is.

            If location is a 3-tuple, the argument should be
            (longitude, latitude, height).

            If location is a dict, keys should be 'longitude',
            'latitude' and 'height'. Alternative keys 'lon', 'lat' and
            'elevation' are allowed.

            If not supplied, units are assumed to be degree for
            longitude and latitude, and meter for height. Longitude
            and latitude are converted to Longitude and Latitude
            objects is not already.

        Returns
        =======

        - an EarthLocation instance

        Raises
        ======

        - TypeError, if location is not a valid input type

        - ValueError, if location is a tuple, but not of length 3.

        """

        if isinstance(location, EarthLocation):
            return location.copy()
        elif isinstance(location, (tuple, list, dict)):
            if isinstance(location, (tuple, list)):
                if len(location) != 3:
                    raise ValueError("supply location as a 3-tuple "
                                     "of (longitude, latitude, height)")
                lon = location[0]
                lat = location[1]
                height = location[2]
            else:  # We try alternative keys as well
                try:
                    lon = location['lon']
                except KeyError:
                    lon = location['longitude']
                try:
                    lat = location['lat']
                except KeyError:
                    lat = location['latitude']
                try:
                    height = location['elevation']
                except KeyError:
                    height = location['height']

            if not isinstance(lon, Longitude):
                lon *= 1 if isinstance(lon, Quantity) else units.degree
                lon = Longitude(lon)
            if not isinstance(lat, Latitude):
                lat *= 1 if isinstance(lat, Quantity) else units.degree
                lat = Latitude(lat)
            height *= 1 if isinstance(height, Quantity) else units.meter
            return EarthLocation(lon, lat, height)
        else:
            raise TypeError("location is not an EarthLocation, "
                            "3-tuple or dict")

    def _parse_fov(self, fov):
        """Parse the field of view

        Arguments
        =========

        - fov: 2-tuple or dict

            If fov is a tuple, the arguments are assumed to be the RA
            and dec field of view.

            If fov is a dict, it should contains the keys 'ra' and
            'dec'.

            The units of RA and dec are assumed to be degree, if not
            supplied.

        Returns
        =======

        fov: the field of view dict

        Raises
        ======

        - TypeError: if fov is not a 2-tuple or dict

        - ValueError, if fov is a tuple, but not of length 2

        """

        if isinstance(fov, (tuple, list)):
            if len(fov) != 2:
                raise ValueError("fov is not a 2-tuple or dict")
            fovra = fov[0]
            fovdec = fov[1]
        elif isinstance(fov, dict):
            fovra = fov['ra']
            fovdec = fov['dec']
        else:
            raise TypeError("fov is not a 2-tuple or dict")
        fovra *= 1 if isinstance(fovra, Quantity) else units.degree
        fovdec *= 1 if isinstance(fovdec, Quantity) else units.degree
        return {'ra': fovra, 'dec': fovdec}


    def altitude(self, time, coords):
        frame = AltAz(obstime=time, location=self.location)
        obscoords = coords.transform_to(frame)
        return obscoords.alt


    def is_visible(self, time, coords):
        """Return whether the skycoord are above the minimum altitude.

        Ignores night or day time.

        Parameters
        ----------

        - time: astropy.time.Time instance

        - coords: astropy.coordinates.SkyCoord instance

        Returns
        -------

        - bool

        """

        return self.altitude(time, coords) >= self.min_elevation

    def is_night(self, time, suncoords=False):
        """Return whether it is (astronomical) night

        Parameters
        ----------

        - time: astropy.time.Time instance

        - suncoords: astropy.coordinates.SkyCoord, or False

        Returns
        -------

        - bool


        The suncoords argument is optional. If False (the default),
        the Sun position will be obtained in this method, otherwise
        the supplied coordinates will be used.

        """

        if not suncoords:
            suncoords = get_sun(time)
        frame = AltAz(obstime=time, location=self.location)
        obscoords = suncoords.transform_to(frame)

        return obscoords.alt < SUNALTITUDE


    def calculate_tiling(self, skymap, date=None,
                         coverage=None, maxtiles=100, within=None,
                         nightsky=False, catalog=None,
                         tilespath=None, njobs=1, tileduration=None):

        """Calculate the best tiling coverage of the probability skymap for a
        given date.

        Arguments

        - skymap: skymap.SkyMap

            A probability skymap instance.

        - date: str, float or astropy.time.Time

            The date for which to calculate the tiling. If a str, it
            should be understood by astropy.time.Time. If a float, it
            is interpreted as a Julian Day value.

            If date is not given or None, the observation date from
            the skymap or metadata is used instead.

        - coverage: dict

            Minimum and maximum fraction to cover. Keys are 'min' and
            'max', values are between 0 and 1.

        - maxtiles: int

            Maximum number of tiles to calculate. Less tiles may be
            calculated if the full coverage is reached before maxtiles
            is reached.

        - nightsky: bool, or 'all'

            Take into account the current or next (upcoming) night.
            Only calculate tiles that are inside the Earth's shadow.
            Night here is astronomical night, but the Sun's minimum
            altitude is configurable in the settings module.

            If nigthsky equals 'all', all of the night sky is taken
            into account, including tiles that have already set for
            the current night.

        - catalog: None or dict

            Fold the probability sky map with a catalog. Specify a
            dict containing 'path' and 'key' keys; the 'key' indicates
            the column to weigh with. If set to `None`, use no
            weighting (equivalent to folding with just the galaxy
            density).

        """

        if coverage is None:
            coverage = COVERAGE
        date = skymap.header['date-det'] if date is None else date
        pointings, tilelist, pixlist, tiledmap, allskymap = self.findtiles(
            skymap, date, catalog=catalog, nightsky=nightsky,
            coverage=coverage, maxtiles=maxtiles, within=within,
            tilespath=tilespath, tileduration=tileduration,
            njobs=njobs)
        if not pointings:
            pointings = None
        pointings = Table(rows=pointings,
                          names=('ra', 'dec', 'prob', 'cumprob',
                                 'relprob', 'cumrelprob'))
        pointings['telescope'] = self.name

        self.results_ = (pointings, tilelist, pixlist, tiledmap, allskymap)

    def findtiles(self, skymap, date, catalog=None, nightsky=False,
                  coverage=(0.05, 0.95), maxtiles=100, within=None,
                  tilespath=None, tileduration=None, njobs=1):
        if catalog is None:
            catalog = {'path': None, 'key': None}
        tiles, pixlist, _ = self.readtiles(tilespath)
        allskymap = skymap.copy()
        allnight = True if nightsky == 'all' else False
        sidtimes = calc_siderealtimes(date, self.location, within=within,
                                      allnight=allnight)
        if len(sidtimes) == 0:
            return [], [], [], np.array([]), allskymap.skymap
        if tileduration:
            maxtiles = int((max(sidtimes) - min(sidtimes)) / tileduration)
        if catalog['path']:
            table = read_catalog(**catalog)
            if nightsky:
                _, mask = visible_catalog(table , sidtimes, self)
                newskymap, _ = map2catalog(allskymap, table[mask])
            allskymap, _ = map2catalog(allskymap, table)
            if not nightsky:
                newskymap = allskymap.copy()
        elif nightsky:
            newskymap, _ = get_visiblemap(skymap, sidtimes, self, njobs=njobs)
        else:
            newskymap = skymap.copy()

        # get fractional percentage covered per pix
        newskymap.skymap /= allskymap.skymap.sum()
        # normalise so allskymap.sum() == 1
        allskymap.skymap /= allskymap.skymap.sum()
        GWtot = newskymap.skymap.sum()
        if GWtot < 1e-8:
            return [], [], [], np.array([]), allskymap.skymap
        tileprobs = filltiles(newskymap.skymap, tiles, pixlist)

        nside = skymap.nside
        pointings = []
        obstilelist = []
        obspixlist = []
        seenpix = []
        usedmap = newskymap.copy()
        GWobs = 0.0
        otiles, opixs, oprobs, itiles = ordertiles(tiles, pixlist, tileprobs)
        l = 0
        while GWobs <= coverage['max']*GWtot and len(pointings) < maxtiles:
            # first tile will be brightest, so blank out pixels of usedmap
            usedmap.skymap[opixs[0]] = 0.0
            seenpix.extend(opixs[0])
            GWobs += oprobs[0]
            _,center = getvectors(otiles[0])
            sphpoints = healpy.vec2ang(center)
            cra, cdec = sph2cel(sphpoints[0], sphpoints[1])

            pointings.append([cra[0], cdec[0], oprobs[0], GWobs, oprobs[0]/GWtot,
                              GWobs/GWtot])
            obstilelist.append(otiles[0])
            obspixlist.append(opixs[0])
            oprobs = filltiles(usedmap.skymap, otiles, opixs)
            otiles, opixs, oprobs, itiles = ordertiles(otiles, opixs, oprobs)
        return pointings, obstilelist, obspixlist, newskymap.skymap, allskymap.skymap

    def gettilespath(self, tilespath=None):
        if tilespath is None:
            tilespath = TILESDIR
        if os.path.isdir(tilespath):
            filename = "{}_nside{}.pgz".format(self.name, NSIDE)
            tilespath = os.path.join(tilespath, filename)
        elif not os.path.isfile(tilespath):
            # Assume tilespath is a 'base' path
            dirname, filename = os.path.split(tilespath)
            filename = "{}_nside{}_{}".format(self.name, NSIDE, filename)
            tilespath = os.path.join(dirname, filename)
        return tilespath

    def readtiles(self, tilespath=None):
        tilespath = self.gettilespath(tilespath)
        if not os.path.isfile(tilespath):
            raise FileNotFoundError("no pre-made tiled grid file found")
        with gzip.GzipFile(tilespath, 'r') as infile:
            try:
                data = pickle.load(infile, encoding='latin1')  # Python 3
            except TypeError:
                data = pickle.load(infile)  # Python 2
        # Allow for multiple 'nside' grids inside the file
        if isinstance(data, dict):
            tilelist, pixlist, centers = data[NSIDE]
        else:
            tilelist, pixlist, centers = data
        logging.debug("Read %s: %d tiles, %d pixels", tilespath,
                      len(tilelist), len(pixlist))
        self.tilelist = tilelist
        self.pixlist = pixlist
        self.gridcoords = centers

        return tilelist, pixlist, centers

    def makegrid(self, tilespath=None):
        tilespath = self.gettilespath(tilespath)
        if os.path.isdir(tilespath):
            filename = "{}_nside{}_nested.pgz".format(self.name, NSIDE)
            tilespath = os.path.join(tilespath, filename)
        if os.path.exists(tilespath):
            raise FileExistsError("tile file {} already exists".format(tilespath))
        logging.info("Creating tiling database map %s", tilespath)
        tileallsky(tilespath, self.fov, NSIDE)


def build_scope(config):
    """Create a telescope instance from a configuration dict"""
    location = EarthLocation(lon=config['longitude'], lat=config['latitude'],
                             height=config['height'])
    fov = {'ra': config['fov-ra'] * units.deg,
           'dec': config['fov-dec'] * units.deg}
    telescope = Telescope(location=location, fov=fov, name=config['short'],
                          min_elevation=config.get('min_elevation', 15))
    return telescope


def read_config_file(filename):
    with open(filename) as infile:
        config = yaml.safe_load(infile)
    return config


# # # Pre-defined telescopes # # #

class GOTON4(Telescope):
    def __init__(self,
                 location=EarthLocation(lon=-17.8793802, lat=28.7598742,
                                        height=2396),  # La Palma
                 fov={'ra': 4.2 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-N-4', min_elevation=15, **kwargs):
        super(GOTON4, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class GOTON8(Telescope):
    def __init__(self,
                 location=EarthLocation(lon=-17.8793802, lat=28.7598742,
                                        height=2396),  # La Palma
                 fov={'ra': 8.4 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-N-8', min_elevation=15, **kwargs):
        super(GOTON8, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class SuperWASPN(Telescope):
    def __init__(self, location=EarthLocation(lon=-17.8793802, lat=28.7598742,
                                              height=2396),
                 fov={'ra': 15 * units.deg,
                      'dec': 30 * units.deg},
                 name='SuperWASP-N', min_elevation=15, **kwargs):
        super(SuperWASPN, self).__init__(location=location, fov=fov, name=name,
                                         min_elevation=min_elevation, **kwargs)


class GOTOS4(Telescope):
    def __init__(self,
                 location=EarthLocation(lon=118.144, lat=-22.608,
                                        height=1200),  # Mt Bruce, WA
                 fov={'ra': 4.2 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-S-4', min_elevation=15, **kwargs):
        super(GOTOS4, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class GOTOS8(Telescope):
    def __init__(self,
                 location=EarthLocation(lon=118.144, lat=-22.608,
                                        height=1200),  # Mt Bruce, WA
                 fov={'ra': 8.4 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-S-8', min_elevation=15, **kwargs):
        super(GOTOS8, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class GOTOLS4(Telescope):
    def __init__(self,
                 location=EarthLocation(lon=-70.7313, lat=-29.2612,
                                        height=2400),  # La Silla
                 fov={'ra': 4.2 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-LS-4', min_elevation=15, **kwargs):
        super(GOTOLS4, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class GOTOLS8(Telescope):
    def __init__(self,
                 location=EarthLocation(lon=-70.7313, lat=-29.2612,
                                        height=2400),  # La Silla
                 fov={'ra': 8.4 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-LS-8', min_elevation=15, **kwargs):
        super(GOTOLS8, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class VISTA(Telescope):
    def __init__(self, location=EarthLocation(lon=-70.3975, lat=-24.6158,
                                              height=2518),
                 fov={'ra': 2.95 * units.deg,
                      'dec': 2.034 * units.deg},
                 name='VISTA', min_elevation=15, **kwargs):
        super(VISTA, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)
