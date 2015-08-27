from __future__ import print_function
from __future__ import absolute_import
import healpy as hp
import numpy as np
import itertools as it
try:
    import cPickle as pickle
except ImportError:
    import pickle
from . import skymaptools as smt
import gzip
import os

def pixelsky(tilesdir,tilelist,scope):

    nside = 256
    nest = True
    pointlist = [smt.getvectors(tile)[0] for tile in tilelist]
    pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=nest) 
                        for points in pointlist])

    outfile = "{}/{}_nside{}_nest{}.pgz".format(tilesdir,scope,nside,nest)
    with gzip.GzipFile(outfile, 'w') as f:
        pickle.dump([tilelist,pixlist], f) #makes gzip compressed pickles
        f.close()

    return


def tileallsky(tilesdir):

    if not os.path.exists(tilesdir):
        os.makedirs(tilesdir)

    scopes = ['GOTO4','GOTO8']

    for scope in scopes:
        print(scope)
        delns,delew = smt.getdels(scope)

        tilelist = []

        north = np.arange(0.0,90.0,delns)
        south = -1*north
        n2s = np.append(south[::-2],north)
        e2w = np.arange(0.0,360.,delew)

        tilelist = np.array([smt.findFoV(lon,lat,delns,delew) 
                             for lat,lon in it.product(n2s,e2w)])

        pixelsky(tilesdir,tilelist,scope)

    return

def readtiles(infile,metadata,tilesdir):

    with gzip.GzipFile('{}/{}'.format(tilesdir,infile), 'r') as f:
        tilelist,pixlist = pickle.load(f)
        f.close()

    return tilelist,pixlist

if __name__=='__main__':

    tileallsky()
