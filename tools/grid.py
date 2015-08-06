import healpy as hp
import numpy as np
import itertools as it
import cPickle
import skymaptools as smt
import gzip
import os

def pixelsky(tilelist,scope):

	nside = 512
	nest = True
	pointlist = [smt.getvectors(tile)[0] for tile in tilelist]
	pixlist = np.array([hp.query_polygon(nside, points[:-1], nest=nest) for points in pointlist])
		
	outfile = "{}/{}_nside{}_nest{}.pgz".format(args.tiles,scope,nside,nest)
	with gzip.GzipFile(outfile, 'w') as f: 
		cPickle.dump([tilelist,pixlist], f) #makes gzip compressed pickles
		f.close()

	return


def tileallsky(args):

	if not os.path.exists(args.tiles):
		os.makedirs(args.tiles)
	
	scopes = ['GOTO4','GOTO8']
	
	for scope in scopes:
		print scope
		delns,delew = smt.getdels(scope)
		
		tilelist = []
	
		north = np.arange(0.0,90.0,delns)
		south = -1*north
		n2s = np.append(south[::-2],north)
		e2w = np.arange(0.0,360.,delew)
			
		tilelist = np.array([smt.findFoV(lon,lat,delns,delew) for lat,lon in it.product(n2s,e2w)])
	
		pixelsky(tilelist,scope)
	
	return
	
def readtiles(infile,metadata,args):

	with gzip.GzipFile('{}/{}'.format(args.tiles,infile), 'r') as f: 
		tilelist,pixlist = cPickle.load(f)
		f.close()
	
	return tilelist,pixlist
	
if __name__=='__main__':
	
	tileallsky()
	
	

