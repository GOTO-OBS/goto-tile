import numpy as np
import healpy as hp
import skymaptools as smt
import astropy.coordinates as acoord
import astropy.units as u

def visiblegals(gals, sidtimes, lat, lon, height, radius):
    observatory = acoord.EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)

    visras,visdecs = [],[]
    ras, decs = [],[]

    ras = gals['ra']*15.
    decs = gals['dec']

    for st in sidtimes:
        frame = acoord.AltAz(obstime=st, location=observatory)
        radecs = acoord.SkyCoord(ra=ras*u.deg, dec=decs*u.deg)
        altaz = radecs.transform_to(frame)
        #print np.where(altaz.alt.degree>(90-radius))
        visras.extend(ras[np.where(altaz.alt.degree>(90-radius))])
        visdecs.extend(decs[np.where(altaz.alt.degree>(90-radius))])

    galradecs = {tuple(row) for row in zip(ras,decs)}
    galradecseen = {tuple(row) for row in zip(visras,visdecs)}
    galradecs = list(galradecs)
    galradecseen = list(galradecseen)

    visgal = []
    for row in galradecseen:
        visgal.append(gals[galradecs.index(row)])

    return np.array(visgal)


def readgals(args,metadata):

    gals = np.genfromtxt('GWGCCatalog_I.txt',skiprows=1,delimiter='|',dtype=None,usecols=(0,1,2,3,7,11,20,21),missing_values='~',filling_values=99.9,names=('PGC','Name','ra','dec','B','I','dist','e_dist'))

    return gals

def Blum(Bmag):
    return pow(10.,(5.48-Bmag)/2.5)

def Ilum(Imag):
    return pow(10.,(4.04-Imag)/2.5)

def getmass(gal):
    return Ilum(gal['I'])*pow(10.,-0.88+0.6*(gal['B']-gal['I']))

def map2gals(skymap,gals,metadata):
    masses = np.zeros(len(skymap))
    ras = gals['ra']*15.
    decs = gals['dec']

    ts,ps = smt.cel2sph(ras,decs)
    galpix = hp.ang2pix(metadata['nside'],ts,ps,nest=metadata['nest'])

    for i,gal in enumerate(gals):
        if gal['B']<0.0:
            galmass = Blum(gal['B']) ###figure out the "getmass" problem with I bands
            masses[galpix[i]]+=galmass#

    galmap = masses*skymap

    return galmap #normalised?
