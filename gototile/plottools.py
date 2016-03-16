from __future__ import absolute_import, division
from . import skymaptools as smt
from . import scopetools as sct
from .catalog import read_catalog
from . import cmap

import numpy as np
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt

from math import sin,cos,atan2,sqrt,pi
from mpl_toolkits.basemap import Basemap
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
from astropy import units
import ephem


def plotskymapsmoll(skymap, pointings, tilelist, metadata, geoplot, catalog,
                    nightsky, scopename, trigger, date,
                    plotfilename, title=None, sun=False, moon=False,
                    objects=None, dpi=300, catcolor='#999999'):
    if title is None:
        formatted_date = Time(date).datetime.strftime("%Y-%m-%d %H:%M:%S")
        if catalog:
            title = ("Skymap, {catalog} and {scopename} tiling "
                     "for trigger {trigger}\n{formatted_data}".format(
                         scopename, trigger, formatted_date))
        else:
            title = ("Skymap and {scopename} tiling "
                     "for trigger {trigger}\n{formatted_data}".format(
                         scopename, trigger, formatted_date))
    if sun:
        sun = get_sun(date)
    if moon:
        moon = ephem.Moon(date.iso)
        phase = moon.phase
        moon = SkyCoord(moon.ra/np.pi*180, moon.dec/np.pi*180,
                        unit=units.degree)
        moon.phase = phase/100
    fig = plt.figure()
    fig.clf()

    m = Basemap(projection='moll',resolution='c',lon_0=0.0)

    m.drawmeridians(np.arange(0,360,30),linewidth=0.25)
    m.drawparallels(np.arange(-90,90,30),linewidth=0.25,labels=[1,0,0,0])
    m.drawmapboundary(color='k', linewidth=0.5)

    if geoplot:
        t = Time(date, location=('0d', '0d'))
        t.delta_ut1_utc = 0
        st = t.sidereal_time('mean')
        dlon = st.radian

        m.drawcoastlines(linewidth=0.25)

    else: dlon = 0 # longitude correction

    ####################
    # Plot Skymap
    ###################
    nside = metadata['nside']
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    thetas,phis = hp.pix2ang(nside,ipix,nest=metadata['nest'])

    ras,decs = smt.sph2cel(thetas,phis-dlon)
    xmap,ymap=m(ras,decs)
    m.scatter(xmap, ymap, s=1, c=skymap, cmap='cylon', alpha=0.5, linewidths=0,
                zorder=1)

    #################
    # Plot FoVs
    #################
    for plotFoV in tilelist:

        FoVra,FoVdec = smt.getshape(plotFoV)
        FoVx,FoVy = m(FoVra-(dlon/np.pi*180.0),FoVdec)
        m.plot(FoVx,FoVy,marker='.',markersize=1,linestyle='none')

    if catalog:
        cattable = read_catalog()

        ras = cattable['ra']
        decs = cattable['dec']

        if nightsky:
            visras, visdecs = [],[]
            import astropy.coordinates as acoord
            import astropy.units as u
            delns, delew, lat, lon, height = sct.getscopeinfo(scopename)
            sidtimes = smt.siderealtimes(lat, lon, height, date)

            observatory = acoord.EarthLocation(lat=lat*u.deg, lon=lon*u.deg,
                                           height=height*u.m)
            radius = 75.
            for st in sidtimes:
                frame = acoord.AltAz(obstime=st, location=observatory)
                radecs = acoord.SkyCoord(ra=ras*u.deg, dec=decs*u.deg)
                altaz = radecs.transform_to(frame)
                visras.extend(ras[np.where(altaz.alt.degree>(90-radius))])
                visdecs.extend(decs[np.where(altaz.alt.degree>(90-radius))])
            xcat, ycat = m(np.array(visras)-(dlon/np.pi*180.0), visdecs)

        else:
            xcat, ycat = m(np.array(ras)-(dlon/np.pi*180.0),decs)
        m.scatter(xcat, ycat, s=0.5, c=catcolor, alpha=0.5, linewidths=0,
                  zorder=2)

        plt.title(title)
    else:
        plt.title(title)
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

    plt.savefig(plotfilename, dpi=dpi)
    plt.close()
    return
