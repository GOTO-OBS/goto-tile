from __future__ import absolute_import, division
import numpy as np

def read2015(sim):
    
    detdels = (5, 6, 3, 6, 5, 5, 5, 5, 6, 5, 8, 6, 7, 8)
    detdtypes = (int, int, 'S3', float, float, float, float, float, float, 
        float, float, float, float, float)
    detcolnames = ("eid", "sid", "net", "snr-net", "snrH", "snrL", "m1", "m2",
        "B50", "B90", "BS", "L50", "L90", "LS")
    det2015 = np.recfromtxt("{0}/2015_coinc.txt".format(sim), skip_header=36, 
        delimiter=detdels, names = detcolnames, dtype = detdtypes, #
        autostrip=True)

    injdels = (5, 6, 12, 6, 6, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6)
    injdtypes = (int, int, float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float)
    injcolnames = ("eid", "sid", "mjd", "ra", "dec", "inc", "pol", "phase", 
        "dist", "m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z")
    inj2015 = np.recfromtxt("{0}/2015_inj.txt".format(sim), skip_header=37, 
        delimiter=injdels, names = injcolnames, dtype = injdtypes, 
        autostrip=True)

    alldels = (5, 3, 12, 6, 6, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6)
    alldtypes = (int, 'S3', float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float)
    allcolnames = ("sid", "net", "mjd", "ra", "dec", "inc", "pol", "phase", 
        "dist", "m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z")
    all2015 = np.recfromtxt("{0}/2015_all_inj.txt".format(sim), skip_header=33, 
        delimiter=alldels, names = allcolnames, dtype = alldtypes, 
        autostrip=True)
    
    return inj2015, det2015, all2015

def read2016(sim):
    
    detdels = (7, 6, 4, 6, 5, 5, 5, 5, 5, 7, 8, 10, 7, 8, 9)
    detdtypes = (int, int, 'S3', float, float, float, float, float, float, 
        float, float, float, float, float)
    detcolnames = ("eid", "sid", "net", "snr-net", "snrH", "snrL", "snrV", 
        "m1", "m2", "B50", "B90", "BS", "L50", "B90", "LS")
    det2016 = np.recfromtxt("{0}/2016_coinc.txt".format(sim), skip_header=37, 
        delimiter=detdels, names = detcolnames, dtype = detdtypes, 
        autostrip=True)

    injdels = (7, 6, 12, 6, 6, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6)
    injdtypes = (int, int, float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float)
    injcolnames = ("eid", "sid", "mjd", "ra", "dec", "inc", "pol", "phase", 
        "dist", "m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z")
    inj2016 = np.recfromtxt("{0}/2016_inj.txt".format(sim), skip_header=37, 
        delimiter=injdels, names = injcolnames, dtype = injdtypes, 
        autostrip=True)

    alldels = (5, 4, 12, 6, 6, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6)
    alldtypes = (int, 'S3', float, float, float, float, float, float, float, 
        float, float, float, float, float, float, float, float)
    allcolnames = ("sid", "net", "mjd", "ra", "dec", "inc", "pol", "phase", 
        "dist", "m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z")
    all2016 = np.recfromtxt("{0}/2016_all_inj.txt".format(sim), skip_header=33, 
        delimiter=alldels, names = allcolnames, dtype = alldtypes, 
        autostrip=True)

    return inj2016, det2016, all2016
    
def findinj(injid,simpath):
    injlist2015,_,_ = read2015(simpath)
    injlist2016,_,_ = read2016(simpath)
    found2015 = False
    for inj in injlist2015:
        listid = inj['eid']
        if int(listid)==int(injid):
            found2015=True
            foundinj = inj
            year = 2015
            break
    
    if not found2015:
        found2016 = False
        for inj2 in injlist2016:
            listid2 = inj2['eid']
            if int(listid2)==int(injid):
                found2016=True
                foundinj = inj2
                year = 2016
                break

    if not found2015 and not found2016: sys.exit("Not found in either year?!")
       
    return foundinj,year

def findinjtile(inj,otiles,opixs):
    
    rainj,decinj,distinj = inj['ra'],inj['dec'],inj['dist']
    tinj,pinj = smt.cel2sph(rainj,decinj)
    injpix = hp.ang2pix(tinj,pinj,nest=True,nside=NSIDE)
    
    for tilenum,[tile,pixels] in enumerate(zip(otiles,opixs)):
        if injpix in set(pixels): 
            tileinj = tilenum+1
            probinj = tprobs[tilenum]
            fprobinj = ftprobs[tilenum]
            break
    return tile,tilenum

def addnewgal(metadata,gals,simpath):
    
    objid = metadata['objid'].split(':')
    injid = objid[-1]
    inj,year = findinj(injid,simpath)

    closegals = [gal for gal in gals if 
            abs(gal['dist']-inj['dist'])<3.0*gal['e_dist']]
    closegals = np.array(closegals,dtype=gals.dtype)
    newgalidx = np.argmin(closegals['B'])
    newgal = closegals[newgalidx]

    newgal['PGC'] = '0'
    newgal['ra'] = inj['ra']/15.0
    newgal['dec'] = inj['dec']

    gals = np.append(gals,newgal)
    
    return gals
