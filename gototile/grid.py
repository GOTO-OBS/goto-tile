from __future__ import print_function
from __future__ import absolute_import
import healpy as hp
import numpy as np
import itertools as it
try:
    import cPickle as pickle
except ImportError:
    import pickle
from gototile import skymaptools as smt
import gzip
import os
import tempfile

def makegrid(tilesdir, name=None):
    if not os.path.exists(tilesdir):
        os.makedirs(tilesdir)
    if name:
        fp = tempfile.NamedTemporaryFile(prefix='temp__', dir='.', delete=False)
        path = fp.name
        fp.close()
        print("Creating temporary grid in file {}".format(path))
        tileallsky(path, name, NSIDE)
        return path
    print("Creating the fixed grid GOTO-4, GOTO-8 and SuperWASP-N "
          "This could take some time.")
    for scope in (4, 8):
        scopename = "GOTO{}".format(scope)
        filename = "{}_nside{}_nestTrue.pgz".format(scopename, NSIDE)
        path = os.path.join(tilesdir, filename)
        tileallsky(path, scopename, NSIDE)
    scopename = "SuperWASP-N"
    filename = "{}_nside{}_nestTrue.pgz".format(scopename, NSIDE)
    path = os.path.join(tilesdir, filename)
    grid.tileallsky(path, scopename, NSIDE)
    return path

def tileallsky(filename, scopename, nside):

    delns, delew, _, _, _ = smt.getscopeinfo(scopename)

    north = np.arange(0.0, 90.0, delns)
    south = -1*north
    n2s = np.append(south[::-2], north)
    e2w = np.arange(0.0, 360., delew)

    tilelist = np.array([smt.findFoV(ra, dec, delns, delew) 
                         for ra, dec in it.product(n2s, e2w)])

    pointlist = [smt.getvectors(tile)[0] for tile in tilelist]
    pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=True) 
                        for points in pointlist])

    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump([tilelist, pixlist], f) #makes gzip compressed pickles


def readtiles(infile):

    with gzip.GzipFile(infile, 'r') as f:
        tilelist,pixlist = pickle.load(f)
        f.close()

    return tilelist,pixlist

def writetiles(outfile,tilelist,pixlist):
    
    with gzip.GzipFile(outfile, 'w') as f:
        pickle.dump([tilelist, pixlist], f) #makes gzip compressed pickles
    return

