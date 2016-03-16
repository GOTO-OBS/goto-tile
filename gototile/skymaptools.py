from __future__ import absolute_import, division, print_function
import os
import spherical_geometry as sg
import spherical_geometry.great_circle_arc as sggc
import spherical_geometry.polygon as sgp
import astropy.coordinates as acoord
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time, TimeDelta
from astropy.table import QTable
from astropy import units
import ephem
import sys
import math
import multiprocessing
import numpy as np
import healpy as hp
from .settings import SUNALTITUDE, TIMESTEP, ARC_PRECISION, COVERAGE, MINPROB
from .catalog import visible_catalog, read_catalog, map2catalog


PI_2 = np.pi / 2
PI2 = 2 * np.pi


def _convc2s_v(ra, dec):
    p = ra * np.pi / 180
    t = -dec * np.pi / 180 + PI_2
    mask = p < 0
    p[mask] = p[mask] + PI2
    mask1 = t > np.pi
    mask2 = t < 0
    t[mask1] = PI2 - t[mask1]
    t[mask2] = -t[mask2]
    mask = mask1 | mask2
    p[mask] = (p[mask] + np.pi) % PI2

    return t, p


def _convc2s(r,d):
    p = r*np.pi/180
    t = (-1*d*np.pi/180)+(np.pi/2.0)

    pchange=False

    if p<0.0:p+=2*np.pi

    if t>np.pi:
        t = np.pi-(t-np.pi)
        pchange = True
    elif t<0.0:
        t = -1*t
        pchange = True

    if pchange == True: p=(p+np.pi)%(2*np.pi)

    return t,p

def _convs2c(t,p):
    r = (p*(180/np.pi))
    d = (t-np.pi/2.0)*-1*(180/np.pi) #spherical coords theta=0 at
                                     #dec=90, and theta=180 at dec=-90

    rachange=False

    if r<0.0:r+=360.0

    if d>90.0:
        d = d%90.0
        rachange = True
    elif d<-90.0:
        d  = -1*(d%90)
        rachange = True

    if rachange == True: r = (r+180.0)%360

    return r,d

def cel2sph_v(rs, ds):
    if isinstance(rs,list) or isinstance(rs,np.ndarray):
        if len(rs)!=len(ds):
            raise ValueError("RA and Dec arrays must be same lengths")
        ts, ps = _convc2s_v(np.asarray(rs), np.asarray(ds))
    else:
        ts, ps=_convc2s(rs, ds)

    return ts,ps


def sph2cel_v(rs, ds):
    if isinstance(rs, list) or isinstance(rs, np.ndarray):
        if len(rs) != len(ds):
            raise ValueError("RA and Dec arrays must be same lengths")
        ts, ps = _convs2c_v(np.asarray(rs), np.asarray(ds))
    else:
        ts, ps=_convs2c(rs, ds)

    return ts,ps


def cel2sph(rs,ds):
    if isinstance(rs,list) or isinstance(rs,np.ndarray):
        if len(rs)!=len(ds):
            sys.exit("RA and Dec arrays must be same lengths")
        elif len(rs)>1:
            ts,ps=[],[]
            for r,d in zip(rs,ds):
                t,p=_convc2s(r,d)
                ts.append(t)
                ps.append(p)
            ts=np.array(ts)
            ps=np.array(ps)
        else:ts,ps=_convs2c(rs,ds)
    else: ts,ps=_convc2s(rs,ds)

    return ts,ps


def sph2cel(ts,ps):
    if isinstance(ts,list) or isinstance(ts,np.ndarray):
        if len(ts)!=len(ps):
            sys.exit("Theta and phi arrays must be same lengths")
        elif len(ts)>1:
            rs,ds=[],[]
            for t,p in zip(ts,ps):
                r,d=_convs2c(t,p)
                rs.append(r)
                ds.append(d)
            rs=np.array(rs)
            ds=np.array(ds)
        else:rs,ds=_convs2c(ts,ps)
    else:
        rs,ds=_convs2c(ts,ps)

    return rs,ds


def findedge(GC, delta):
    for i, step in enumerate(GC):
        if sggc.length(GC[0], step, degrees=True)>delta:
            edge=step
            break
    return edge


def getpoints(FoV): #Get lra/dec vertices for shape on sky

    points = vars(vars(FoV)['_polygons'][0])['_points']
    sphpoints = hp.vec2ang(points)


    ra,dec = sph2cel(sphpoints[0],sphpoints[1])

    return np.array(ra),np.array(dec)


def find_tile(ra, dec, delra, deldec):

    tc, pc = cel2sph(ra, dec)
    # find vertices needed for drawing along great circles.
    te, pe = cel2sph(ra+90.0, 0.0)
    tw, pw = cel2sph(ra-90.0, 0.0)
    tn, pn = cel2sph(ra, dec+90.0)
    ts, ps = cel2sph(ra, dec-90.0)

    center = hp.ang2vec(tc, pc)
    npole = hp.ang2vec(tn, pn) # "poles" of GC from center (ie +/- 90
                               # degrees at right angles)
    spole = hp.ang2vec(ts, ps)
    epole = hp.ang2vec(te, pe)
    wpole = hp.ang2vec(tw, pw)

    eastGC = sggc.interpolate(center, epole, steps=ARC_PRECISION)
    westGC = sggc.interpolate(center, wpole, steps=ARC_PRECISION)

    e = findedge(eastGC, delra)
    w = findedge(westGC, delra)

    # don't need to interpolate for stepping along RA great circle, so
    # just do +/- step
    dmin = dec - deldec
    dmax = dec + deldec

    tmax, pmax = cel2sph(ra, dmax)
    tmin, pmin = cel2sph(ra, dmin)

    n = hp.ang2vec(tmax, pmax)
    s = hp.ang2vec(tmin, pmin)

    nw = sggc.intersection(npole, w, wpole, n)
    ne = sggc.intersection(npole, e, epole, n)
    sw = sggc.intersection(spole, w, wpole, s)
    se = sggc.intersection(spole, e, epole, s)
    fov = sgp.SphericalPolygon([nw,ne,se,sw,nw], inside=center)

    return fov, center
    #return Tile(fov, center)


def getshape(FoV): #Get points that allow shape to be drawn on sky
                   #using plot/scatter points

    points = vars(vars(FoV)['_polygons'][0])['_points']
    ras, decs = [],[]
    for i,A in enumerate(points[:-1]):
        B=points[i+1]

        ipoints = sggc.interpolate(A,B,steps=100)

        sphpoints = hp.vec2ang(ipoints)

        ra,dec = sph2cel(sphpoints[0],sphpoints[1])

        ras.extend(ra)
        decs.extend(dec)

    return np.array(ras),np.array(decs)


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
    obs = ephem.Observer()
    obs.pressure = 0
    obs.horizon = str(SUNALTITUDE.value)
    obs.lon = str(location.longitude.value)
    obs.lat = str(location.latitude.value)
    obs.elevation = location.height.value
    obs.date = ephem.Date(date.iso)
    sun = ephem.Sun(obs)

    # daytime so want to know time of next setting and rising of sun
    if sun.alt > ephem.degrees(obs.horizon):
        start = Time(
            obs.next_setting(ephem.Sun()).datetime(), format='datetime')
        stop = Time(
            obs.next_rising(ephem.Sun()).datetime(), format='datetime')
    elif allnight: # currently night, but take into account the part
                   # of sky that has already set.
        start = Time(
            obs.previous_setting(ephem.Sun()).datetime(), format='datetime')
        stop = Time(
            obs.next_rising(ephem.Sun()).datetime(), format='datetime')
    else: # currently night; only calculate from now until Sun rise
        start = date
        stop = Time(
            obs.next_rising(ephem.Sun()).datetime(), format='datetime')

    if within:
        stop_ = date + within
        if stop_ <= start:  # Stop before Sun set
            return []
        if stop_ < stop:  # Stop before Sun rise
            stop = stop_
    delta = TIMESTEP
    diff = stop - start
    steps = int(np.round(diff / delta))
    times = np.linspace(start.mjd, stop.mjd, steps)
    times = Time(times, format='mjd')

    return times


# We can't pass extra arguments to a simple pool.map call, nor can we
# pickle a closure (pickle used for multiprocessing), thus we use a
# class, where extra arguments are passed as instance attributes, and
# __call__ is used to mimic a function
class VisibleMap(object):
    def __init__(self, telescope, skycoords, ipix):
        self.telescope = telescope
        self.skycoords = skycoords
        self.ipix = ipix
        #self.min_elevation = min_elevation

    def __call__(self, sidtime):
        frame = AltAz(obstime=sidtime,
                      location=self.telescope.location)
        obscoords = self.skycoords.transform_to(frame)
        seenpix = self.ipix[np.where(obscoords.alt > self.telescope.min_elevation)]
        return seenpix



def get_visiblemap(skymap, sidtimes, telescope, njobs=1):
    maskedmap = skymap.copy()
    maskedmap.skymap[:] = 0.0
    if njobs == -1:
        njobs = None
    if not sidtimes:
        return maskedmap, np.array([], dtype=np.int)
    skycoords = skymap.skycoords()
    ipix = np.arange(len(skymap.skymap))
    pool = multiprocessing.Pool(njobs)
    func = VisibleMap(telescope, skycoords, ipix)
    seen = pool.map(func, sidtimes)
    # Close and free up the memory
    pool.close()
    pool.join()
    indices = np.unique(np.hstack(seen))

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
    skycoords = skymap.skycoords()
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


def calc_tilecenter(tile):
    """Return the tile center

    Returns

    - astropy.coordinates.SkyCoord

    """

    _, center = getvectors(tile)
    sphpoints = hp.vec2ang(center)
    cra, cdec = sph2cel(sphpoints[0], sphpoints[1])
    return SkyCoord(cra[0] * units.deg, cdec[0] * units.deg)


def calculate_tiling(skymap, telescopes, date=None,
                     coverage=None, maxtiles=100, within=None,
                     nightsky=False, catalog=None,
                     tilespath=None, njobs=1, tileduration=None):
    if coverage is None:
        coverage = COVERAGE
    if catalog is None:
        catalog = {'path': None, 'key': None}
    date = skymap.header['date-det'] if date is None else date

    allskymap = skymap.copy()
    tiles, pixlist, sidtimes = {}, {}, {}
    for telescope in telescopes:
        telescope.indices = {}
        telescope.tiles, telescope.pixlist, telescope.tilecenters = telescope.readtiles(tilespath)
        telescope.sidtimes = calc_siderealtimes(date, telescope.location,
                                                within=within,
                                                allnight=(nightsky == 'all'))
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
                                  'tile'],
                           dtype=[SkyCoord, 'f8', 'f8', 'f8', 'f8', 'S20',
                                  Time, TimeDelta, sgp.SphericalPolygon])
        return pointings, [], [], np.array([]), allskymap.skymap

    nside = skymap.nside
    pointings = []
    obstilelist = []
    obspixlist = []
    usedmap = newskymap.copy()
    GWobs = 0.0
    nscopes = len(telescopes)
    time = date
    dt = TIMESTEP
    endtime = date + within if within else date + units.year
    base_indices = np.arange(len(telescopes))
    ntiles = 0
    while (GWobs <= coverage['max'] * GWtot and
           ntiles < maxtiles and
           time < endtime):
        # Filter out telescopes in daytime
        indices = np.array([i for i in base_indices[:]
                            if telescopes[i].is_night(time)])
        if len(indices):
            ntiles += 1
        # We rerun the tiling with a subset of telescopes until all
        # telescopes have calculated their optimal tiling
        while len(indices):
            # Select subset of relevant telescopes
            seltelescopes = [telescopes[i] for i in indices]
            # Run best tile calculating on this subset
            filltiles(seltelescopes, time)
            ordertiles(seltelescopes)
            # Compare first (brightest) tiles of telescopes
            sortindices = np.argsort([telescope.topprob
                                      for telescope in seltelescopes])[::-1]
            index = indices[sortindices][0]
            indices = indices[sortindices][1:]
            telescope = telescopes[index]
            tile = telescope.toptile
            prob = telescope.topprob
            sources = telescope.topsources
            GWobs += prob
            center = telescope.topcenter
            if prob >= MINPROB:
                pointings.append([center, prob, GWobs, prob/GWtot,
                                  GWobs/GWtot, telescope.name,
                                  time, time-date, tile, sources])
                obstilelist.append(tile)
                obspixlist.append(telescope.toppixlist)
            # Blank out used pixels in all telescope skymaps
            pixlist = telescope.toppixlist
            for telescope in telescopes:
                telescope.skymap.skymap[pixlist] = 0
        time += dt

    pointings = QTable(rows=pointings,
                      names=['center', 'prob', 'cumprob', 'relprob',
                             'cumrelprob', 'telescope', 'time', 'dt',
                             'tile', 'sources'])
    return pointings, obstilelist, obspixlist, newskymap.skymap, allskymap.skymap


# NB: the telescope instances are modified in-place, so we don't
# return from this function
def filltiles(telescopes, time):
    tileprobs = []
    for telescope in telescopes:
        telescope.tileprobs = np.array(
            [telescope.skymap.skymap[pixels].sum()
             for i, pixels in enumerate(telescope.pixlist)])
        telescope.vismask = telescope.is_visible(time, telescope.tilecenters)


# NB: the telescope instances are modified in-place, so we don't
# return from this function
def ordertiles(telescopes):
    for telescope in telescopes:
        vismask = telescope.vismask
        if not len(telescope.tileprobs[vismask]):
            telescope.topprob = 0
            telescope.toptile = None
            telescope.toppixlist = []
            telescope.topcenter = None
            continue
        indices = np.argsort(telescope.tileprobs[vismask])[::-1]
        itop = np.where(vismask)[0][indices[0]]
        itops = np.where(vismask)[0][indices[:10]]
        telescope.topprob = telescope.tileprobs[itop]
        telescope.toptile = telescope.tiles[itop]
        telescope.toppixlist = telescope.pixlist[itop]
        telescope.topcenter = telescope.tilecenters[itop]
        telescope.topsources = None
        if hasattr(telescope, 'catsources'):
            telescope.topsources = []
            for pixel in telescope.pixlist[itop]:
                sources = telescope.catsources[pixel]
                telescope.topsources.extend(sources)
