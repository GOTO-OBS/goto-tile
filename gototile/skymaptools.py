from __future__ import absolute_import, division, print_function
import os
import logging
import astropy.coordinates as acoord
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time, TimeDelta
from astropy.table import QTable
from astropy import units
from astropy.utils import iers
import ephem
import sys
import multiprocessing
import numpy as np
from . import settings
from .catalog import visible_catalog, read_catalog, map2catalog
from . import math
from . import utils


def getshape(tile, steps=50):
    """Interpolate a tile with corners to a full shape to be drawn.

    tile is a two dimensional array of [ra-array, dec-array] in degrees.

    """
    # Code adopted from spherical_geometry.great_circle_arc

    x, y, z = math.radec2xyz(tile[0], tile[1])
    xyz = np.array(math.radec2xyz(tile[0], tile[1])).T
    points = []
    for corner1, corner2 in zip(xyz, np.roll(xyz, 1, axis=0)):
        # Could retrieve the lengths from the telescope info
        length = np.arccos(np.clip(math.dot(corner1, corner2), -1, 1))
        offsets = np.linspace(0, 1, steps, endpoint=True).reshape((steps, 1))
        if length > 0:
            offsets = np.sin(offsets * length) / np.sin(length)
        point = offsets[::-1] * corner1 + offsets * corner2
        points.extend(offsets[::-1] * corner2 + offsets * corner1)
    ra, dec = math.xyz2radec(*np.asarray(points).T)
    return ra, dec


def getvectors(tile):

    points = np.array(list(tile.points)[0])
    center = list(tile.inside)[0]

    return points, center


def calc_siderealtimes(date, location, within=None, allnight=False):
    """Calculate the sidereal times from sunset to sunrise for `date`.

    This uses either the current night, or, if daytime, the upcoming
    night. If the within argument is given, sideral times are only
    calculated within the given amount of days from date.

    Arguments
    ---------

    date :

    location :

    within :


    Remarks
    -------

    Sunset and sunrise are defined by the Sun's altitude being
    below -18. (This can be changed in the settings module.)

    The list of sidereal times is stepped in intervals of 300
    seconds (this can be changed in the settings module).


    """
    logging.info("Calculating sideral times for {}".format(location))
    obs = ephem.Observer()
    obs.pressure = 0
    obs.horizon = str(getattr(settings, 'SUNALTITUDE').value)
    obs.lon = str(location.lon.value)
    obs.lat = str(location.lat.value)
    obs.elevation = location.height.value
    obs.date = ephem.Date(date.iso)
    sun = ephem.Sun(obs)

    # daytime so want to know time of next setting and rising of sun
    if sun.alt > ephem.degrees(obs.horizon):
        start = Time(
            obs.next_setting(sun).datetime(), format='datetime')
        stop = Time(
            obs.next_rising(sun).datetime(), format='datetime')
    elif allnight: # currently night, but take into account the part
                   # of sky that has already set.
        start = Time(
            obs.previous_setting(sun).datetime(), format='datetime')
        stop = Time(
            obs.next_rising(sun).datetime(), format='datetime')
    else: # currently night; only calculate from now until Sun rise
        start = date
        stop = Time(
            obs.next_rising(sun).datetime(), format='datetime')

    if within:
        stop_ = date + within
        if stop_ <= start:  # Stop before Sun set
            return []
        if stop_ < stop:  # Stop before Sun rise
            stop = stop_
        else:  # check we're not in the next night
            nextset = Time(
                obs.next_setting(sun).datetime(), format='datetime')
            if stop_ >= nextset:
                if within > 1 * units.year:  # limit to 1 year
                    within = 1 * units.year
                start = [start]
                stop = [stop]
                t = nextset
                while t < stop_:
                    obs.date = ephem.Date((t + 1*units.min).iso )
                    rise = Time(
                        obs.next_rising(sun).datetime(), format='datetime')
                    if stop_ < rise:
                        rise = stop_
                    start.append(t)
                    stop.append(rise)
                    t = Time(
                        obs.next_setting(sun).datetime(), format='datetime')

    if not isinstance(start, list):
        start = [start]
    if not isinstance(stop, list):
        stop = [stop]
    delta = getattr(settings, 'TIMESTEP')
    times = []

    for start_, stop_ in zip(start, stop):
        diff = stop_ - start_
        steps = int(np.round(diff / delta))
        times_ = np.linspace(start_.mjd, stop_.mjd, steps)
        times.append(Time(times_, format='mjd').mjd)
    times = Time(np.hstack(times), format='mjd')

    return times


# We can't pass extra arguments to a simple pool.map call, nor can we
# pickle a closure (pickle used for multiprocessing), thus we use a
# class, where extra arguments are passed as instance attributes, and
# __call__ is used to mimic a function
class VisibleMap(object):
    def __init__(self, telescope, skycoords, ipix, iers_url=None):
        self.telescope = telescope
        self.skycoords = skycoords
        self.ipix = ipix
        self.iers_url = iers_url

    def __call__(self, sidtime):
        # Since this call is multiprocessed and on an independent
        # Python process, we need to (re)set the IERS URL as necessary
        # for each process
        if self.iers_url is not None:
            iers.conf.iers_auto_url = self.iers_url
            if self.iers_url == '':
                # These two settings don't really help; perhaps in the
                # future
                iers.conf.auto_max_age = None
                iers.conf.auto_download = False

        frame = AltAz(obstime=sidtime,
                      location=self.telescope.location)
        obscoords = self.skycoords.transform_to(frame)
        seenpix = self.ipix[np.where(obscoords.alt >
                                     self.telescope.min_elevation)]
        return seenpix



def get_visiblemap(skymap, sidtimes, telescope, njobs=1):
    logging.info("Calculating the visible (night-sky) map")
    maskedmap = skymap.copy()
    maskedmap.skymap[:] = 0.0
    if njobs == -1:
        njobs = None
    if not sidtimes:
        return maskedmap, np.array([], dtype=np.int)
    skycoords = skymap.coords.copy()
    dtype = getattr(settings, 'IDTYPE')
    ipix = np.arange(len(skymap.skymap), dtype=dtype)
    func = VisibleMap(telescope, skycoords, ipix,
                      iers_url=iers.conf.iers_auto_url)

    if njobs is None or njobs > 1:
        pool = multiprocessing.Pool(njobs)
        seen = pool.map(func, sidtimes)
        # Close and free up the memory
        pool.close()
        pool.join()
    else:
        seen = []
        for sidtime in sidtimes:
            seen.append(func(sidtime))
    indices = np.unique(np.hstack(seen))
    logging.info("{:d} pixels out of {:d} visible".format
                 (len(indices), len(skymap.skymap)))
    maskedmap.skymap[indices] = skymap.skymap[indices]

    return maskedmap, indices

# For further speed-up, one can use the class and function below
# instead of the above.. It is not used, since it gains
# relatively little (20%-40%), at the cost of being more complex to
# read.
class VisibleMapBitFaster(object):
    def __init__(self, location, skycoords, ipix, min_elevation=15,
                 unseen=None):
        self.location = location
        self.skycoords = skycoords
        self.ipix = ipix
        self.min_elevation = min_elevation
        self.unseen = [] if unseen is None else unseen

    def __call__(self, sidtime):
        frame = AltAz(obstime=sidtime, location=self.location)
        obscoords = self.skycoords[self.unseen].transform_to(frame)
        indices = np.where(obscoords.alt.degree > self.min_elevation)
        seenpix = self.ipix[self.unseen][indices]
        return seenpix


def getbatch(data, size=1):
    for i in range(0, len(data), size):
        yield data[i:i+size]


def get_visiblemap_bit_faster(skymap, sidtimes, location, min_elevation,
                              njobs=1):
    if njobs == -1 or njobs is None:
        njobs = os.cpu_count()
    skycoords = skymap.coords.copy()
    ipix = np.arange(len(skymap.skymap))
    seenlist = []
    pool = multiprocessing.Pool(njobs)
    unseen = np.ones(len(skymap.skymap), dtype=np.bool)
    for times_batch in getbatch(sidtimes, size=njobs):
        func = VisibleMapBitFaster(location, skycoords, ipix, min_elevation,
                                   unseen)
        seen = pool.map(func, times_batch)
        if seen:
            seenlist.extend(seen)
        unseen[np.hstack(seenlist)] = False
    # Close and free up the memory
    pool.close()
    pool.join()
    seen = np.unique(np.hstack(seenlist))

    maskedmap = skymap.copy()
    maskedmap.skymap[:] = 0.0
    maskedmap.skymap[seen] = skymap.skymap[seen]

    return maskedmap


def calculate_tiling(skymap, telescopes, date=None,
                     coverage=None, maxtiles=100, within=None,
                     nightsky=False, catalog=None,
                     tilespath=None, njobs=1):
    if coverage is None:
        coverage = getattr(settings, 'COVERAGE')
    if catalog is None:
        catalog = {'path': None, 'key': None}
    date = skymap.date_det if date is None else date

    utils.test_iers()

    allskymap = skymap.copy()
    tiles, pixlist, sidtimes = {}, {}, {}
    for telescope in telescopes:
        telescope.indices = {}
        telescope.get_grid()
        telescope.sidtimes = calc_siderealtimes(
            date, telescope.location, within=within,
            allnight=(nightsky == 'all'))
    timespan = within
    if timespan is None:
        timespan = max([telescope.sidtimes[-1] for telescope in telescopes])
        timespan = timespan - date
    if catalog['path']:
        cattable = read_catalog(**catalog)
        allskymap, catsources = map2catalog(allskymap, cattable)
        if nightsky:
            indiceslist = []
            for telescope in telescopes:
                _, indices = visible_catalog(
                    cattable, telescope.sidtimes,
                    telescope)
                telescope.indices['catalog'] = indices
                telescope.skymap, telescope.catsources = map2catalog(
                    skymap, cattable[indices])
                indiceslist.append(indices)
            indices = np.unique(np.hstack(indiceslist))
            cattable = cattable[indices]
            newskymap, catsources = map2catalog(skymap, cattable)
        else:
            newskymap = allskymap.copy()
            for telescope in telescopes:
                telescope.skymap = allskymap.copy()
    elif nightsky:
        indiceslist = []
        for telescope in telescopes:
            vismap, indices = get_visiblemap(
                skymap, telescope.sidtimes, telescope, njobs=njobs)
            telescope.indices['vis'] = indices
            telescope.skymap = vismap.copy()
            indiceslist.append(indices)
        indices = np.unique(np.hstack(indiceslist))
        newskymap = skymap.copy()
        newskymap.skymap[:] = 0
        newskymap.skymap[indices] = skymap.skymap[indices]
    else:
        newskymap = skymap.copy()
        for telescope in telescopes:
            telescope.skymap = skymap.copy()

    # get fractional percentage covered per pix
    total = allskymap.skymap.sum()
    newskymap.skymap /= total
    # normalise so allskymap.sum() == 1
    allskymap.skymap /= total
    for telescope in telescopes:
        telescope.skymap.skymap /= total

    GWtot = newskymap.skymap.sum()
    if GWtot < 1e-8:
        pointings = QTable(names=['center', 'prob', 'cumprob', 'relprob',
                                  'cumrelprob', 'telescope', 'time', 'dt',
                                  'tile', 'sources'],
                           dtype=[SkyCoord, 'f8', 'f8', 'f8', 'f8', 'U20',
                                  Time, TimeDelta, np.ndarray, np.ndarray])
        pointings.tilelist = np.array([], dtype=np.float)
        pointings.pixellist = np.array([], dtype=np.ndarray)

        return pointings, np.array([]), allskymap.skymap

    nside = skymap.nside
    pointings = []
    obstilelist = []
    obspixlist = []
    usedmap = newskymap.copy()
    GWobs = 0.0
    nscopes = len(telescopes)
    time = date
    dt = getattr(settings, 'TIMESTEP')
    endtime = date + timespan
    base_indices = np.arange(len(telescopes))
    ntiles = 0
    nobs = 0
    logging.info("Calculating timing between %s and %s", time, endtime)
    while (GWobs <= coverage['max'] * GWtot and
           ntiles < maxtiles and
           time < endtime):
        logging.debug("Calculating tiling for %s", time)
        # Filter out telescopes in daytime
        indices = np.array([i for i in base_indices[:]
                            if telescopes[i].is_night(time)])
        # We rerun the tiling with a subset of telescopes until all
        # telescopes have calculated their optimal tiling
        while len(indices):
            # Select subset of relevant telescopes
            seltelescopes = [telescopes[i] for i in indices]

            # Run best tile calculating on this subset
            for telescope in seltelescopes:
                #filltiles
                telescope.grid.apply_skymap(telescope.skymap)
                telescope.tileprobs = telescope.grid.probs
                if time:
                    telescope.vismask = telescope.is_visible(time,
                        telescope.grid.coords)

                #ordertiles
                vismask = telescope.vismask
                if not len(telescope.tileprobs[vismask]):
                    telescope.topprob = 0
                    telescope.toptile = None
                    telescope.toppixlist = []
                    telescope.topcenter = None
                    telescope.topname = ''
                    continue
                ins = np.argsort(telescope.tileprobs[vismask])[::-1]
                itop = np.where(vismask)[0][ins[0]]
                itops = np.where(vismask)[0][ins[:10]]
                telescope.topprob = telescope.tileprobs[itop]
                telescope.toptile = telescope.grid.vertices[itop]
                telescope.toppixlist = telescope.grid.pixels[itop]
                telescope.topcenter = telescope.grid.coords[itop]
                telescope.topname = telescope.grid.tilenames[itop]
                telescope.topsources = []
                if hasattr(telescope, 'catsources'):
                    for pixel in telescope.pixlist[itop]:
                        sources = telescope.catsources[pixel]
                        telescope.topsources.extend(sources)
                telescope.topsources = np.asarray(telescope.topsources)

            # Compare first (brightest) tiles of telescopes
            sortindices = np.argsort([telescope.topprob
                                      for telescope in seltelescopes])[::-1]
            index = indices[sortindices][0]
            indices = indices[sortindices][1:]
            telescope = telescopes[index]
            tile = math.xyz2radec(*telescope.toptile.T)
            prob = telescope.topprob
            sources = telescope.topsources
            GWobs += prob
            center = telescope.topcenter
            name = telescope.topname
            if prob >= getattr(settings, 'MINPROB'):
                logging.debug("Tile with prob. %.4f for %s",
                              prob, telescope.name)
                pointings.append([str(name), center, prob, GWobs, prob/GWtot,
                                  GWobs/GWtot, telescope.name,
                                  time, time-date, tile, sources])
                obstilelist.append(tile)
                obspixlist.append(telescope.toppixlist)
                logging.debug("Total probability: %.4f", GWobs)
            # Blank out used pixels in all telescope skymaps
            pixlist = telescope.toppixlist
            for telescope in telescopes:
                telescope.skymap.skymap[pixlist] = 0
        if len(obstilelist) > nobs:
            nobs = len(obstilelist)
            ntiles += 1
        time += dt

    # The `rows` parameter in the `QTable` initializer can't properly
    # handle equal-sized numpy arrays that should be treated as a
    # single object ('tile' and 'source'): it will attempt to expand
    # the array. The list(zip(*pointings)) works around that.
    names = ['fieldname', 'center', 'prob', 'cumprob', 'relprob',
             'cumrelprob', 'telescope', 'time', 'dt', 'tile', 'sources']
    dtype = ['U20', SkyCoord, 'f8', 'f8', 'f8', 'f8', 'U20',
             Time, TimeDelta, np.ndarray, np.ndarray]
    if len(pointings) == 0:
        pointings = QTable(names=names, dtype=dtype)
    else:
        pointings = QTable(list(zip(*pointings)), names=names, dtype=dtype)
    # Add the obstilelist and obspixelist separately
    # Note that individual tiles inside the table are preserved as
    # type 'object', not as type 'float64'
    pointings.tilelist = obstilelist
    pointings.pixellist = obspixlist

    return pointings, newskymap.skymap, allskymap.skymap


def tile_skymap(skymap, grid, observed=None):
    '''Return the tile probabilities for a given skymap.

    Unlike calculate_tiling this function doesn't consider time and
    visibility, only attributes of the telescope and tiles.

    Parameters
    ----------
    skymap : `gototile.skymap.SkyMap`
        The sky map to apply

    grid : `gototile.skymap.SkyGrid`
        The all-sky grid to apply the map to

    observed : list of int or str, optional
        A list containing tile indicies or names that have already been observed (if any).
        Any tiles in this list will have their probability of any HEALPix pixels
        within set to zero, thereby reducing the contained probability of any
        overlapping tiles.
    '''
    utils.test_iers()

    # Get the pixels within each grid tile
    pixlist = grid.get_pixels(skymap.nside, skymap.isnested)

    # Get all the pixels within observed tiles
    bad_pix = set()
    for tile in observed:
        if isinstance(tile, str):
            tile = grid.tilenames.index(tile)
        for pix in pixlist[tile]:
            bad_pix.add(pix)
    bad_pix = np.array(list(bad_pix))

    # Make sure the skymap is normalised
    skymap.normalise()

    # Reduce the bad pixels to zero probability
    if len(bad_pix) > 0:
        skymap.skymap[bad_pix] = 0

    # and apply the skymap
    grid.apply_skymap(skymap)

    return grid.get_table()
