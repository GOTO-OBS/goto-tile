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


def tileallsky(filename, scopename, nside):

    delns, delew = smt.getdels(scopename)

    north = np.arange(0.0, 90.0, delns)
    south = -1*north
    n2s = np.append(south[::-2], north)
    e2w = np.arange(0.0, 360., delew)

    tilelist = np.array([smt.findFoV(lon, lat, delns, delew) 
                         for lat, lon in it.product(n2s, e2w)])

    pointlist = [smt.getvectors(tile)[0] for tile in tilelist]
    pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=True) 
                        for points in pointlist])

    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump([tilelist, pixlist], f) #makes gzip compressed pickles


def readtiles(infile,metadata):

    with gzip.GzipFile(infile, 'r') as f:
        tilelist,pixlist = pickle.load(f)
        f.close()

    return tilelist,pixlist


if __name__=='__main__':

    tileallsky()
