from __future__ import absolute_import, division
import os
import spherical_geometry as sg
import spherical_geometry.great_circle_arc as sggc
import spherical_geometry.polygon as sgp
import astropy.coordinates as acoord
from astropy.time import Time, TimeDelta
import astropy.units as u
import ephem
import sys
import math
import multiprocessing
import numpy as np
import healpy as hp
from . import galtools as gt
from .settings import SUNALTITUDE, TIMESTEP, ARC_PRECISION


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

def findedge(GC,delta):
    for i,step in enumerate(GC):
        if sggc.length(GC[0],step,degrees=True)>delta:

            edge=step
            break
    return edge

def findFoV(RA,dec, delns, delew):

    tc,pc = cel2sph(RA,dec)
    te,pe = cel2sph(RA+90.0,0.0) #find vertices needed for drawing
                                 #along great circles.
    tw,pw = cel2sph(RA-90.0,0.0)
    tn,pn = cel2sph(RA,dec+90.0)
    ts,ps = cel2sph(RA,dec-90.0)

    center = hp.ang2vec(tc,pc)
    npole = hp.ang2vec(tn,pn) #"poles" of GC from center (ie +/- 90
                              #degrees at right angles)
    spole = hp.ang2vec(ts,ps)
    epole = hp.ang2vec(te,pe)
    wpole = hp.ang2vec(tw,pw)

    eastGC = sggc.interpolate(center,epole,steps=ARC_PRECISION)
    westGC = sggc.interpolate(center,wpole,steps=ARC_PRECISION)

    e = findedge(eastGC,delew)
    w = findedge(westGC,delew)

    # don't need to interpolate for stepping along RA great circle, so
    # just do +/- step
    dmin = dec-delns
    dmax = dec+delns

    tmax,pmax = cel2sph(RA,dmax)
    tmin,pmin = cel2sph(RA,dmin)

    n = hp.ang2vec(tmax,pmax)
    s = hp.ang2vec(tmin,pmin)

    nw = sggc.intersection(npole,w,wpole,n)
    ne = sggc.intersection(npole,e,epole,n)
    sw = sggc.intersection(spole,w,wpole,s)
    se = sggc.intersection(spole,e,epole,s)

    FoV = sgp.SphericalPolygon([nw,ne,se,sw,nw], inside = center)

    return FoV #returns sgp polygon of FoV

def getpoints(FoV): #Get lra/dec vertices for shape on sky

    points = vars(vars(FoV)['_polygons'][0])['_points']
    sphpoints = hp.vec2ang(points)


    ra,dec = sph2cel(sphpoints[0],sphpoints[1])

    return np.array(ra),np.array(dec)

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

def getgrid(FoV,steps = 100): #Get regular grid of points for field to
                              #find pixels using hp.ang2pix

    points = vars(vars(FoV)['_polygons'][0])['_points']
    edgeras, edgedecs = [],[]
    for i,A in enumerate(points[:-1]):
        B=points[i+1]

        ipoints = sggc.interpolate(A,B,steps=steps)

        sphpoints = hp.vec2ang(ipoints)

        ra,dec = sph2cel(sphpoints[0],sphpoints[1])

        edgeras.append(ra)
        edgedecs.append(dec)


    tt,tp = cel2sph(edgeras[0],edgedecs[0])
    bt,bp = cel2sph(edgeras[2],edgedecs[2])

    top = hp.ang2vec(tt,tp)
    bottom = hp.ang2vec(bt[::-1],bp[::-1])

    gridras,griddecs=[],[]
    for A,B in zip(top,bottom):
        gpoints = sggc.interpolate(A,B,steps=steps)

        gsphpoints = hp.vec2ang(gpoints)

        gra,gdec = sph2cel(gsphpoints[0],gsphpoints[1])

        gridras.extend(gra)
        gridds.extend(gdec)
    return np.array(gridras),np.array(griddecs)

def getvectors(FoV):

    points = vars(vars(FoV)['_polygons'][0])['_points']
    center = vars(vars(FoV)['_polygons'][0])['_inside']

    return points,center

# return the pixel list in order of loudest to quietest, with
# corresponding original indices
def orderedpixels(skymap):
    pixels = np.sort(skymap)
    indices = np.argsort(skymap)
    return np.array(pixels[::-1]),np.array(indices[::-1])


# return the pixel list in order of loudest to quietest, with
# corresponding original indices
def ordertiles(tiles,pixlist,tileprobs):
    oprobs = np.sort(tileprobs)
    otiles = tiles[np.argsort(tileprobs)]
    opixs = pixlist[np.argsort(tileprobs)]
    return np.array(otiles[::-1]),np.array(opixs[::-1]),np.array(oprobs[::-1])


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
    obs.horizon = str(SUNALTITUDE)
    obs.lon = str(location.longitude.value)
    obs.lat = str(location.latitude.value)
    obs.elevation = location.height.value
    obs.date = ephem.Date(date.iso)
    sun = ephem.Sun(obs)

    # daytime so want to know time of next setting and rising of sun
    if sun.alt > ephem.degrees(str(SUNALTITUDE)):
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
    delta = TimeDelta(TIMESTEP, format='sec')
    diff = stop - start
    steps = int(np.round(diff / delta))
    times = np.linspace(start.mjd, stop.mjd, steps)
    times = Time(times, format='mjd')

    return times


def siderealtimes(lat, lon, height, date, allnight=False):
    #t = Time(mjd, format='mjd', scale='utc')

    obs = ephem.Observer()
    obs.pressure=0
    obs.horizon='-18:00'
    obs.lat,obs.lon = str(lat),str(lon)
    obs.date = ephem.Date(date.iso)
    sun = ephem.Sun(obs)
    # daytime so want to know time of next setting and rising of sun
    if sun.alt > ephem.degrees(str(-18.)):
        ATstart = Time(
            obs.next_setting(ephem.Sun()).datetime(), format='datetime')
        ATend = Time(
            obs.next_rising(ephem.Sun()).datetime(), format='datetime')
    elif allnight: # currently night, but take into account the part
                   # of sky that has already set.
        ATstart = Time(
            obs.previous_setting(ephem.Sun()).datetime(), format='datetime')
        ATend = Time(
            obs.next_rising(ephem.Sun()).datetime(), format='datetime')
    else: # currently night; only calculate from now until Sun rise
        ATstart = date
        ATend = Time(
            obs.next_rising(ephem.Sun()).datetime(), format='datetime')


    delt = TimeDelta(300., format='sec')

    diff = ATend-ATstart
    steps = diff/delt

    times = np.linspace(ATstart.mjd,ATend.mjd,steps+1)

    timeobjs = Time(times, format='mjd')

    return timeobjs


def visiblemap(skymap, sidtimes, lat, lon, height, radius, metadata):


    observatory = acoord.EarthLocation(lat=lat*u.deg, lon=lon*u.deg,
                                       height=height*u.m)
    seen = []
    for st in sidtimes:
        frame = acoord.AltAz(obstime=st, location=observatory)
        npix = len(skymap)
        ipix = np.arange(npix)
        theta, phi = hp.pix2ang(metadata['nside'], ipix, nest=metadata['nest'])
        radecs = acoord.SkyCoord(ra=phi*u.rad, dec=(0.5*np.pi - theta)*u.rad)
        altaz = radecs.transform_to(frame)
        seenpix = ipix[np.where(altaz.alt.degree>(90-radius))]
        seen.extend(list(seenpix))
        seen = list(np.unique(seen))

    maskedmap = skymap.copy()
    maskedmap[:] = 0.0
    maskedmap[seen] = skymap[seen]

    return maskedmap


def visiblemap_new(skymap, sidtimes, lat, lon, height, radius, metadata):


    observatory = acoord.EarthLocation(lat=lat*u.deg, lon=lon*u.deg,
                                       height=height*u.m)
    seen = []
    npix = len(skymap)
    ipix = np.arange(npix)
    theta, phi = hp.pix2ang(metadata['nside'], ipix, nest=metadata['nest'])
    radecs = acoord.SkyCoord(ra=phi*u.rad, dec=(0.5*np.pi - theta)*u.rad)
    unseen = np.ones(npix, dtype=np.bool)
    for st in sidtimes:
        # Since we need the ipix indices over the full range, we can't
        # simply shorten radec using radec = radec[unseen], but we
        # need the boolean indices.
        frame = acoord.AltAz(obstime=st, location=observatory)
        altaz = radecs[unseen].transform_to(frame)
        seenpix = ipix[unseen][np.where(altaz.alt.degree>(90-radius))]
        unseen[seenpix] = False
        seen.extend(list(seenpix))
        seen = list(np.unique(seen))

    maskedmap = skymap.copy()
    maskedmap[:] = 0.0
    maskedmap[seen] = skymap[seen]

    return maskedmap


class VisibleMap(object):
    def __init__(self, location, skycoords, ipix, min_elevation=15):
        self.location = location
        self.skycoords = skycoords
        self.ipix = ipix
        self.min_elevation = min_elevation

    def __call__(self, sidtime):
        frame = acoord.AltAz(obstime=sidtime, location=self.location)
        obscoords = self.skycoords.transform_to(frame)
        seenpix = self.ipix[np.where(obscoords.alt.degree > self.min_elevation)]
        return seenpix



def getbatch(data, size=1):
    for i in range(0, len(data), size):
        yield data[i:i+size]


def get_visiblemap(skymap, sidtimes, location, min_elevation, njobs=1):
    if njobs == -1:
        njobs = None
    skycoords = skymap.skycoords()
    ipix = np.arange(len(skymap.skymap))
    pool = multiprocessing.Pool(njobs)
    func = VisibleMap(location, skycoords, ipix, min_elevation)
    seen = pool.map(func, sidtimes)
    # Close and free up the memory
    pool.close()
    pool.join()
    seen = np.unique(np.hstack(seen))

    maskedmap = skymap.skymap.copy()
    maskedmap[:] = 0.0
    maskedmap[seen] = skymap.skymap[seen]

    return maskedmap

# For further speed-up, one can use the class and function below
# instead of the above.. It is not implemented, since it gains
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
        frame = acoord.AltAz(obstime=sidtime, location=self.location)
        obscoords = self.skycoords[self.unseen].transform_to(frame)
        indices = np.where(obscoords.alt.degree > self.min_elevation)
        seenpix = self.ipix[self.unseen][indices]
        return seenpix


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

    maskedmap = skymap.skymap.copy()
    maskedmap[:] = 0.0
    maskedmap[seen] = skymap.skymap[seen]

    return maskedmap


def filltiles(skymap, tiles, pixlist):

    tileprobs = np.array([skymap[pixels].sum() for pixels in pixlist])

    #~ lenlist=[len(pixels) for pixels in pixlist]
    return tileprobs

# return pixel to center on for next tile
def findtiles(skymap, date, delns, delew, metadata, usegals, nightsky, minf,
              maxf, maxt, sim, injgal, simpath, lat, lon, height, tiles,
              pixlist, min_elevation=15):
    allskymap = skymap.copy()
    sidtimes = siderealtimes(lat, lon, height, date)

    if usegals:
        allgals = gt.readgals(metadata,injgal,simpath)

        if nightsky:
            gals = gt.visiblegals(allgals, sidtimes, lat, lon, height, 75.)
            skymap = gt.map2gals(allskymap,gals,metadata)

        allskymap = gt.map2gals(allskymap,allgals,metadata)

        if not nightsky:
            skymap = allskymap.copy()

    elif nightsky:
        skymap = visiblemap(skymap, sidtimes, lat, lon, height, 75., metadata)

    skymap = skymap/allskymap.sum() #gets fractional percentage covered per pix
    allskymap = allskymap/allskymap.sum() #normalised so allskymap.sum()===1
    GWtot = skymap.sum()
    if GWtot < minf:
        sys.exit("Less than 5% of the skymap probability is visible, "
                 "ignoring...")
    tileprobs = filltiles(skymap, tiles, pixlist)

    nside = metadata['nside']
    npix = len(skymap)
    ipix = np.arange(npix)


    pointings = []
    obstilelist = []
    obspixlist = []
    seenpix = []
    usedmap = skymap.copy()
    GWobs = 0.0

    otiles,opixs,oprobs = ordertiles(tiles,pixlist,tileprobs)
    l=0
    while GWobs <= maxf*GWtot and len(pointings) < maxt:

        # first tile will be brightest, so blank out pixels of usedmap
        usedmap[opixs[0]]=0.0
        seenpix.extend(opixs[0])
        GWobs+=oprobs[0]

        _,center = getvectors(otiles[0])
        sphpoints = hp.vec2ang(center)
        cra,cdec = sph2cel(sphpoints[0],sphpoints[1])

        pointings.append([cra,cdec,oprobs[0],GWobs,oprobs[0]/GWtot,
                        GWobs/GWtot])
        obstilelist.append(otiles[0])
        obspixlist.append(opixs[0])
        #oprobs[0] = 0.0 #seen so set tile prob to zero
        oprobs = filltiles(usedmap, otiles, opixs)
        otiles,opixs,oprobs = ordertiles(otiles,opixs,oprobs)
        #~ l+=1
        #~ ro=0
        #~ while len(list(set(opixs[0]).intersection(seenpix)))>0:
            #~ #does new top tile overlap with seen?
            #~ ro+=1
            #~ oprobs[0] = usedmap[opixs[0]].sum() #recalc prob from seen map
            #~ otiles,opixs,oprobs = ordertiles(otiles,opixs,oprobs) #reorder

    return pointings, obstilelist, obspixlist, skymap, allskymap
