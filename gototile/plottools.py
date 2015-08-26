from __future__ import absolute_import, print_function, division
from . import skymaptools as smt
from . import galtools as gt
from . import cmap

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt

from math import sin,cos,atan2,sqrt,pi
from mpl_toolkits.basemap import Basemap
from astropy.time import Time

def plotskymapsnsper(skymap, pointings, metadata, geoplot, usegals, 
                     output, path, scopename):
    fig = plt.figure()
    fig.clf()

    if geoplot:
        # longitude correction
        t = Time(metadata['mjd'], format='mjd',location=('0d', '0d'))
        st = t.sidereal_time('mean')
        dlon = st.radian
    else: dlon = 0

    nside = metadata['nside']
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    thetas,phis = hp.pix2ang(nside,ipix,nest=metadata['nest'])

    p1 = pointings[0]
    lonmax,latmax = p1[0]-(dlon/np.pi*180.0),p1[1]

    h = 3000.
    m = Basemap(projection='nsper',lon_0=lonmax,lat_0=latmax, 
                satellite_height=h*10000.,resolution='l')#

    m.drawmeridians(np.arange(0,360,30),linewidth=0.25)
    m.drawparallels(np.arange(-90,90,30),linewidth=0.25)
    m.drawmapboundary(color='k', linewidth=0.5)

    if geoplot:

        m.drawcoastlines(linewidth=0.25)
        #m.drawcountries(linewidth=0.25)
        #m.fillcontinents(color='coral',lake_color='aqua')
        #m.drawmapboundary(fill_color='aqua')


    ####################
    # Plot Skymap
    ###################
    longs, lats = smt.sph2cel(thetas,phis-dlon)
    xmap,ymap=m(longs,lats)
    m.scatter(xmap, ymap, s=1, c=skymap, cmap='cylon', alpha=0.5, linewidths=0)

    #################
    # Plot FoVs
    #################
    for tileinfo in pointings:

        plotFoV = tileinfo[2]

        FoVlon,FoVlat = smt.getshape(plotFoV)
        FoVx,FoVy = m(FoVlon-(dlon/np.pi*180.0),FoVlat)
        m.plot(FoVx,FoVy,marker='.',markersize=1,linestyle='none')

    if usegals:
        gals = gt.readgals(metadata)
        galras = gals['ra']*15.
        galdecs = gals['dec']
        masslist = [getmass(gal) for gal in gals]
        galmassnorm = np.array(masslist)/max(masslist)
        #ts,ps = cel2sph(ras,decs)
        xgal,ygal=m(galras,galdecs)
        m.scatter(xgal, ygal, s=0.5, c='k', cmap='cylon', alpha=0.5, 
                  linewidths=0)

        plt.title(
                "Skymap, GWGC galaxies and {0} tiling for trigger {1}".format(
                        output, scopename))
    else:
        plt.title("Skymap and {0} tiling for trigger {1}".format(
                scopename, output))

    plt.savefig('{0}/{1}nsper{2}.png'.format(path, output, scopename), 
                dpi=300)
    plt.close()
    return


def plotskymapsmoll(skymap, pointings, metadata, geoplot, usegals, 
                    output, path, scopename):
    fig = plt.figure()#
    fig.clf()

    m = Basemap(projection='moll',resolution='c',lon_0=0.0)

    m.drawmeridians(np.arange(0,360,30),linewidth=0.25)
    m.drawparallels(np.arange(-90,90,30),linewidth=0.25,labels=[1,0,0,0])
    m.drawmapboundary(color='k', linewidth=0.5)

    if geoplot:
        t = Time(metadata['mjd'], format='mjd',location=('0d', '0d'))
        st = t.sidereal_time('mean')
        dlon = st.radian

        m.drawcoastlines(linewidth=0.25)
        #m.drawcountries(linewidth=0.25)
        #m.fillcontinents(color='coral',lake_color='aqua')
        #m.drawmapboundary(fill_color='aqua')


        # longitude correction

    else: dlon = 0

    ####################
    # Plot Skymap
    ###################
    nside = metadata['nside']
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    thetas,phis = hp.pix2ang(nside,ipix,nest=metadata['nest'])

    longs, lats = smt.sph2cel(thetas,phis-dlon)
    xmap,ymap=m(longs,lats)
    m.scatter(xmap, ymap, s=1, c=skymap, cmap='cylon', alpha=0.5, linewidths=0)

    #################
    # Plot FoVs
    #################
    for tileinfo in pointings:

        plotFoV = tileinfo[2]

        FoVlon,FoVlat = smt.getshape(plotFoV)
        FoVx,FoVy = m(FoVlon-(dlon/np.pi*180.0),FoVlat)
        m.plot(FoVx,FoVy,marker='.',markersize=1,linestyle='none')

    if usegals:
        gals = gt.readgals(metadata)
        galras = gals['ra']*15.
        galdecs = gals['dec']
        masslist = [getmass(gal) for gal in gals]
        galmassnorm = np.array(masslist)/max(masslist)
        #ts,ps = celi2sph(ras,decs)
        xgal,ygal=m(galras,galdecs)
        m.scatter(xgal, ygal, s=0.5, c='k', cmap='cylon', alpha=0.5, 
                  linewidths=0)

        plt.title(
                "Skymap, GWGC galaxies and {0} tiling for trigger {1}".format(
                        output, scopename))
    else:
        plt.title("Skymap and {0} tiling for trigger {1}".format(
                scopename, output))

    plt.savefig('{0}/{1}moll{2}.png'.format(path, output, scopename), 
                dpi=300)
    plt.close()
    return
