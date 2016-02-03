"""Module to handle telescopes and instrumentation

"""
from __future__ import division
import os
import gzip
import pickle
import logging
import numpy as np
from astropy.coordinates import EarthLocation, Latitude, Longitude
from astropy.coordinates import AltAz, SkyCoord
from astropy.units import Quantity, UnitConversionError
from astropy import units
from astropy.table import Table
import healpy
import ephem
from .settings import NSIDE, TILESDIR, TIMESTEP, SUNALTITUDE, COVERAGE
from .grid import readtiles, tileallsky_new
from .skymaptools import (findtiles, calc_siderealtimes, get_visiblemap,
                          filltiles, ordertiles, getvectors, sph2cel)

try:  # Python 2
    stringtype = basestring
    class FileNotFoundError(IOError):
        pass
    class FileExistsError(IOError):
        pass
except NameError:  # Python 3
    stringtype = str


class Telescope(object):
    """Telescope and instrument class

    The telescope class keeps track of the position, field of view,
    name and minimum elevation of the telescope + instrument
    combination.

    Position is longitude Eastward, latitude and height above sea
    level.

    Minimum elevation is measured in degrees from the horizon.

    """
    def __init__(self, location, fov=None, name=None, min_elevation=0):
        # Try to be flexible in what is accepted; this creates a
        # relatively long initialization
        self.name = None
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
        if name is not None:
            self.name = name

    @property
    def logger(self):
        return logging.getLogger(__name__)

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

    def __str__(self):
        location = "{:s}, {:s}, {:.1f}".format(self.location.longitude,
                                             self.location.latitude,
                                             self.location.height)
        return "{:s} @ {:s} - [{:s}, {:s}]".format(
            self.name, location, self.fov['ra'], self.fov['dec'])

    def calculate_tiling(self, skymap, date=None,
                         coverage=None, maxtiles=100, within=None,
                         nightsky=False, galaxies=False,
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

        - galaxies: bool

            Fold the probability sky map with the Gravitational Wave
            Galaxy Catalog (White et al 2011), for optimal tiling.

        """

        if coverage is None:
            coverage = COVERAGE
        date = skymap.header['date-det'] if date is None else date
        pointings, tilelist, pixlist, tiledmap, allskymap = self.findtiles(
            skymap, date, usegals=galaxies, nightsky=nightsky,
            coverage=coverage, maxtiles=maxtiles, within=within,
            tilespath=tilespath, tileduration=tileduration,
            sim=False, injgal=False, simpath='.', njobs=njobs)
        pointings = Table(rows=pointings, names=('ra', 'dec', 'obs_sky_frac',
                                                 'cum_obs_sky_frac',
                                                 'tileprob', 'cum_prob'))
        self.results_ = (pointings, tilelist, pixlist, tiledmap, allskymap)

    def findtiles(self, skymap, date, usegals=False, nightsky=False,
                  coverage=(0.05, 0.95), maxtiles=100, within=None,
                  tilespath=None, tileduration=None,
                  sim=False, injgal=False, simpath='.', njobs=1):
        tiles, pixlist = self.readtiles(tilespath)
        allskymap = skymap.skymap.copy()
        allnight = True if nightsky == 'all' else False
        sidtimes = calc_siderealtimes(date, self.location, within=within,
                                      allnight=allnight)
        if len(sidtimes) == 0:
            return [], [], [], np.array([]), allskymap
        if tileduration:
            maxtiles = int((max(sidtimes) - min(sidtimes)) / tileduration)
        if usegals:
            allgals = gt.readgals(skymap.object, injgal, simpath)
            if nightsky:
                gals = gt.visiblegals(allgals, sidtimes, self.location,
                                      self.min_elevation.value)
                newskymap = gt.map2gals(allskymap, gals, metadata)
            allskymap = gt.map2gals(allskymap, allgals, metadata)
            if not nightsky:
                newskymap = allskymap.copy()
        elif nightsky:
            newskymap = get_visiblemap(skymap, sidtimes, self.location,
                                       self.min_elevation.value, njobs=njobs)
        else:
            newskymap = skymap.skymap

        newskymap = newskymap/allskymap.sum() # gets fractional percentage covered per pix
        allskymap = allskymap/allskymap.sum() # normalised so allskymap.sum() == 1
        GWtot = newskymap.sum()
        if GWtot < 1e-8:
            return [], [], [], np.array([]), allskymap
        tileprobs = filltiles(newskymap, tiles, pixlist)

        nside = skymap.nside
        pointings = []
        obstilelist = []
        obspixlist = []
        seenpix = []
        usedmap = newskymap.copy()
        GWobs = 0.0
        otiles, opixs, oprobs = ordertiles(tiles, pixlist, tileprobs)
        l=0
        while GWobs <= coverage['max']*GWtot and len(pointings) < maxtiles:
            # first tile will be brightest, so blank out pixels of usedmap
            usedmap[opixs[0]] = 0.0
            seenpix.extend(opixs[0])
            GWobs += oprobs[0]
            _,center = getvectors(otiles[0])
            sphpoints = healpy.vec2ang(center)
            cra, cdec = sph2cel(sphpoints[0], sphpoints[1])

            pointings.append([cra, cdec, oprobs[0], GWobs, oprobs[0]/GWtot,
                            GWobs/GWtot])
            obstilelist.append(otiles[0])
            obspixlist.append(opixs[0])
            oprobs = filltiles(usedmap, otiles, opixs)
            otiles, opixs, oprobs = ordertiles(otiles, opixs, oprobs)
        return pointings, obstilelist, obspixlist, newskymap, allskymap

    def gettilespath(self, tilespath=None):
        if tilespath is None:
            tilespath = TILESDIR
        if os.path.isdir(tilespath):
            filename = "{}_nside{}_nested.pgz".format(self.name, NSIDE)
            tilespath = os.path.join(tilespath, filename)
        return tilespath

    def readtiles(self, tilespath=None):
        tilespath = self.gettilespath(tilespath)
        if not os.path.isfile(tilespath):
            raise FileNotFoundError("no pre-made tiled grid file found")
        with gzip.GzipFile(tilespath, 'r') as infile:
            tilelist, pixlist = pickle.load(infile)
        self.logger.debug("Read %s: %d tiles, %d pixels", tilespath,
                      len(tilelist), len(pixlist))
        return tilelist, pixlist

    def makegrid(self, tilespath=None):
        tilespath = self.gettilespath(tilespath)
        if os.path.isdir(tilespath):
            filename = "{}_nside{}_nested.pgz".format(self.name, NSIDE)
            tilespath = os.path.join(tilespath, filename)
        if os.path.exists(tilespath):
            raise FileExistsError("tile file {} already exists".format(tilespath))
        self.logger.info("Creating tiling database map %s", tilespath)
        tileallsky_new(tilespath, self.fov, NSIDE)



# # # Pre-defined telescopes # # #

class GOTON4(Telescope):
    def __init__(self, location=EarthLocation(-17.8793802, 28.7598742, 2396),
                 fov={'ra': 4.2 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-N-4', min_elevation=15, **kwargs):
        super(GOTON4, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class GOTON8(Telescope):
    def __init__(self, location=EarthLocation(-17.8793802, 28.7598742, 2396),
                 fov={'ra': 8.4 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-N-8', min_elevation=15, **kwargs):
        super(GOTON8, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class SuperWASPN(Telescope):
    def __init__(self, location=EarthLocation(-17.8793802, 28.7598742, 2396),
                 fov={'ra': 15 * units.deg,
                      'dec': 30 * units.deg},
                 name='SuperWASP-N', min_elevation=15, **kwargs):
        super(SuperWASPN, self).__init__(location=location, fov=fov, name=name,
                                         min_elevation=min_elevation, **kwargs)


class GOTOS4(Telescope):
    def __init__(self, location=EarthLocation(118.144, -22.608, 1200),
                 fov={'ra': 4.2 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-S-4', min_elevation=15, **kwargs):
        super(GOTOS4, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class GOTOS8(Telescope):
    def __init__(self, location=EarthLocation(118.144, -22.608, 1200),
                 fov={'ra': 8.4 * units.deg,
                      'dec': 4.2 * units.deg},
                 name='GOTO-S-8', min_elevation=15, **kwargs):
        super(GOTOS8, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)


class VISTA(Telescope):
    def __init__(self, location=EarthLocation(-70.3975, -24.6158, 2518),
                 fov={'ra': 2.95 * units.deg,
                      'dec': 2.034 * units.deg},
                 name='VISTA', min_elevation=15, **kwargs):
        super(VISTA, self).__init__(location=location, fov=fov, name=name,
                                     min_elevation=min_elevation, **kwargs)
