from __future__ import division, print_function
import sys
import os
import logging
from copy import deepcopy
try:
    import cPickle as pickle
    class FileExistsError(IOError):
        pass
except ImportError:
    import pickle
import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from gototile.skymaptools import findFoV
from spherical_geometry.vector import radec_to_vector
from gototile import fits as gtfits
from gototile.skymap import SkyMap
from gototile.skymaptools import findtiles
from gototile.telescope import GOTOS8, GOTON8


MAXTILES = 150
TILESPATH = '/home/evert/code/python/gototile/simtiles'
NSIDE = 256
DATAFILE='https://dcc.ligo.org/public/0109/P1300187/025/2016_inj.txt'
TILE_OBS_DUR = 5 * units.min


def is_source_in_tiles(tilelist, source):
    for i, tile in enumerate(tilelist):
        if tile.contains_point(
            radec_to_vector(source.ra.to(units.deg).value,
                            source.dec.to(units.deg).value)):
            return True, tile, i
    return False, None, -1


class Options(object):
    pass

def main():
    logging.basicConfig(level='INFO',
                        format='%(asctime)s [%(levelname)-8s] %(name)-15s:  %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S')
    options = Options()
    options.date = None
    options.minfrac = 0.0
    options.maxfrac = 0.95
    options.nightsky = True
    options.catalog = {'path': False, 'key': None}
    options.jobs = 6
    options.coverage = {'min': options.minfrac, 'max': options.maxfrac}
    options.outputfile = "4sitesaus_lapalma_fov8_bns-bbh-sims.pck"

    locations = [EarthLocation(118.144 * units.deg, -22.608 * units.deg,
                               height = 1200 * units.m),
                 EarthLocation(149.0672 * units.deg, -31.2754 * units.deg,
                               height = 1130 * units.m),
                 EarthLocation(118.588 * units.deg, -22.980 * units.deg,
                               height = 1200 * units.m),
                 EarthLocation(132.20 * units.deg, -23.88611 * units.deg,
                               height = 950 * units.m),
                 GOTON8().location]
    names = ['MtBruce', 'SSO', 'MtMeharry', 'Mereenie', 'lapalma']
    totals = {}
    for within_ in (1, 2, 4, 8, None):
        if within_:
            within = within_ * units.hour
            maxtiles = int(within / TILE_OBS_DUR)
        else:
            within = None
            maxtiles = 150
        scoperesults = {}
        for location, name in zip(locations, names):
            telescope = GOTOS8(location=location, name=name)
            try:
                telescope.makegrid(TILESPATH)
            except FileExistsError:
                pass
            results = {}
            logging.info("====== %s ======", name)
            for mapfile in sys.argv[1:]:
                skymap = SkyMap.from_fits(mapfile)
                objid = skymap.object
                objid = objid.split(':')[-1]
                skymap.regrade(nside=NSIDE)
                date = skymap.date_det if options.date is None else options.date
                logging.info("---- Map %s, for source ID %s", mapfile, objid)
                if within:
                    logging.info("Calculating tiling between %s and %s UT",
                                 date.datetime.strftime("%Y-%m-%dT%H:%M:%S"),
                                 (date+within).datetime.strftime("%Y-%m-%dT%H:%M:%S"))
                else:
                    logging.info("Calculating unconstrained tiling")
                pointings, tilelist, pixlist, tiledmap, allskymap = calculate_tiling(
                    skymap,
                    [telescope],
                    date=date,
                    coverage=options.coverage,
                    maxtiles=maxtiles,
                    within=within,
                    nightsky=options.nightsky,
                    catalog=options.catalog,
                    visible=False,
                    tilespath=TILESPATH,
                    njobs=options.jobs)
                result = {'map': mapfile}
                if len(pointings) == 0:
                    result['coverage'] = 0
                    result['bestcoverage'] = 0
                    result['observed'] = False
                    result['pointing'] = None
                    result['tileno'] = -1
                    result['tile'] = None
                    result['time_since_t0'] = np.inf
                    results[objid] = result
                    logging.info("Coverage 0.0, not observed")
                    continue
                coverage = pointings[-1][3]  # coverage compared to all sky
                bestcoverage = pointings[-1][5]  # coverage compared to night sky
                result['coverage'] = coverage
                result['bestcoverage'] = bestcoverage

                source = SkyCoord
                source = SkyCoord(skymap.header['injra'] * units.deg,
                                  skymap.header['injdec'] * units.deg)
                source.id = objid

                observed, tile, tileno = is_source_in_tiles(tilelist, source)
                result['observed'] = observed
                result['tile'] = tile
                result['tileno'] = tileno
                result['pointing'] = pointings[tileno] if tileno > -1 else None
                time_since_t0 = ((tileno + 0.5) * TILE_OBS_DUR) if tileno > -1 else np.inf * units.hour
                result['time_since_t0'] = time_since_t0.decompose(bases=[units.hour]).value
                logging.info("Coverage %f, observed: %s", coverage, observed)
                if observed:
                    logging.info("Tile (%s, %s) covers source at (%s, %s)",
                                 pointings[tileno][0][0], pointings[tileno][1][0],
                                 source.ra.to(units.deg).value,
                                 source.dec.to(units.deg).value)
                    logging.info("Source was observed in tile # %d, after %s",
                                 tileno, time_since_t0)
                results[objid] = result
            scoperesults[name] = deepcopy(results)
        key = within.decompose(bases=[units.hour]).value if within else None
        totals[key] = deepcopy(scoperesults)

    with open(options.outputfile, "wb") as outfile:
        pickle.dump(totals, outfile)


if __name__ == '__main__':
    main()
