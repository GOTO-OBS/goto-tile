from __future__ import absolute_import, division

import os.path
import numpy as np
import healpy as hp
import astropy.coordinates as acoord
import astropy.units as u

def visiblegals(gals, sidtimes, lat, lon, height, radius):
    observatory = acoord.EarthLocation(lat=lat*u.deg, lon=lon*u.deg, 
                                       height=height*u.m)

    visras,visdecs = [],[]
    ras, decs = [],[]

    ras = gals['ra']
    decs = gals['dec']

    for st in sidtimes:
        frame = acoord.AltAz(obstime=st, location=observatory)
        radecs = acoord.SkyCoord(ra=ras*u.deg, dec=decs*u.deg)
        altaz = radecs.transform_to(frame)
        visras.extend(ras[np.where(altaz.alt.degree>(90.-radius))])
        visdecs.extend(decs[np.where(altaz.alt.degree>(90.-radius))])
    
    galradecs = {tuple(row):gals[i] for i,row in enumerate(zip(ras,decs))}
    galradecseen = {tuple(row) for row in zip(visras,visdecs)}
    
    visgal = []
    for row in galradecseen:
        visgal.append(galradecs[row])

    return np.array(visgal)


def visiblegals_new(gals, sidtimes, telescope):

    mask = np.zeros(len(gals), dtype=np.bool)
    radecs = acoord.SkyCoord(ra=gals['ra']*u.deg, dec=gals['dec']*u.deg)
    for st in sidtimes:
        frame = acoord.AltAz(obstime=st, location=telescope.location)
        altaz = radecs.transform_to(frame)
        mask |= altaz.alt > telescope.min_elevation
    return gals[mask], np.where(mask)[0]


def readgals(metadata,injgal,simpath):
    path = os.path.join(os.path.dirname(__file__), 'GWGCCatalog_I.txt')
    gals = np.genfromtxt(path, skip_header=1, delimiter='|',
                         dtype=None, usecols=(0, 1, 2, 3, 7, 11, 20, 21),
                         missing_values='~', filling_values=99.9,
                         names=('PGC', 'Name', 'ra', 'dec', 'B', 'I', 'dist',
                                'e_dist'))
    
    if injgal:
        from . import simtools as simt
        gals = simt.addnewgal(metadata,gals,simpath)
        
    gals['ra'] = gals['ra']*15.
        
    return gals


def readgals_new(objid='', simpath='.'):
    path = os.path.join(os.path.dirname(__file__), 'GWGCCatalog_I.txt')
    gals = np.genfromtxt(path, skip_header=1, delimiter='|',
                         dtype=None, usecols=(0, 1, 2, 3, 7, 11, 20, 21),
                         missing_values='~', filling_values=99.9,
                         names=('PGC', 'Name', 'ra', 'dec', 'B', 'I', 'dist',
                                 'e_dist'))
    
    if objid:
        from . import simtools as simt
        gals = simt.addnewgal_new(objid, gals=gals, simpath=simpath)
        
    gals['ra'] = 15 * gals['ra']
    
    return gals


def Blum(Bmag):
    return pow(10.,(5.48-Bmag)/2.5)


def Ilum(Imag):
    return pow(10.,(4.04-Imag)/2.5)


def getmass(gal):
    return Ilum(gal['I'])*pow(10.,-0.88+0.6*(gal['B']-gal['I']))


def map2gals(skymap,gals,metadata):
    from . import skymaptools as smt
    masses = np.zeros(len(skymap))
    ras = gals['ra']
    decs = gals['dec']

    ts,ps = smt.cel2sph(ras,decs)
    galpix = hp.ang2pix(metadata['nside'],ts,ps,nest=metadata['nest'])

    for i,gal in enumerate(gals):
        if gal['B']<0.0:
            galmass = Blum(gal['B']) #figure out the "getmass" problem
                                     #with I bands
            masses[galpix[i]]+=galmass#

    galmap = masses*skymap

    return galmap #normalised?


def map2gals_new(skymap, gals):
    """Return a copy of the skymap folded with the galaxy catalog"""
    from .skymaptools import cel2sph_v
    masses = np.zeros(len(skymap.skymap))
    ras = gals['ra']
    decs = gals['dec']

    ts, ps = cel2sph_v(gals['ra'], gals['dec'])
    galpix = hp.ang2pix(skymap.nside, ts, ps, nest=skymap.isnested)

    for i, gal in enumerate(gals):
        if gal['B'] < 0.0:
            galmass = Blum(gal['B'])
            masses[galpix[i]] += galmass

    galmap = skymap.copy()
    galmap.skymap = masses * galmap.skymap
    #galmap.skyamp *= masses
    return galmap
