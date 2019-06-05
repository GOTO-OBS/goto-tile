"""Core functionality"""

from __future__ import absolute_import, print_function, division

import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import astropy
from astropy.time import Time
from .skymaptools import calculate_tiling
from .skymap import SkyMap
from . import settings
from . import extinction
from . import catalog
from . import telescope as telmodule
from .telescope import build_scope, read_config_file
from .utils import pointings_to_text
from .log import set_logging
from .parser import parse_args, parse_date, parse_object
try:
    FileExistsError
except NameError:
    from .utils import FileExistsError


def run(skymap, telescopes, nside=None, date=None,
        coverage=None, maxtiles=100, within=None,
        nightsky=False, catalog=None, tilespath='./tiles',
        njobs=1, command='',
        outputoptions=None, plotoptions=None):

    if coverage is None:
        coverage = {'min': 0.05, 'max': 0.95}
    if catalog is None:
        catalog = {'path': None, 'key': None}
    if nside is None:
        nside = getattr(settings, 'NSIDE')
    if outputoptions is None:
        outputoptions = {}
    if plotoptions is None:
        plotoptions = {}

    # Replace telescope classes or class names with their instances
    for i, telescope in enumerate(telescopes):
        if isinstance(telescope, type):
            telescopes[i] = telescope()
        elif isinstance(telescope, str):
            telclass = getattr(telmodule, telescope)
            telescopes[i] = telclass()

    skymap.regrade(nside=nside)

    date = skymap.date_det if date is None else date

    pointings, tiledmap, allskymap = calculate_tiling(
        skymap, telescopes, date=date, coverage=coverage,
        maxtiles=maxtiles, within=within,
        nightsky=nightsky, catalog=catalog,
        tilespath=tilespath, njobs=njobs)

    gwtot = tiledmap.sum()
    allsky = allskymap.sum()
    pointings.meta['comments'] = [
        "Tiling map obtained for {}".format(date.datetime),
        "The total probability visible during the next observing "
        "period is {:.3f}".format(gwtot),
        "This is {:5.2f}% of the original skymap".format((gwtot/allsky)*100.)
    ]
    pointings.meta['command'] = command
    pointings.meta['time-created'] = Time.now().datetime.strftime(
        "%Y-%m-%dT%H:%M:%S")
    pointings.meta['time-planning'] = date.datetime.strftime(
        "%Y-%m-%dT%H:%M:%S")

    if outputoptions.get('text'):
        table = pointings_to_text(pointings, catalog=catalog)
        table.write(outputoptions['text'], format='ascii.ecsv', overwrite=True)

    if outputoptions.get('latex'):  # Very similar to output, but with less
                             # precision (more human readable when rendered)
        table = pointings[['fieldname', 'prob', 'cumprob', 'telescope']].copy()
        table['prob'] = ["{:.2f}".format(100 * prob)
                         for prob in table['prob']]
        table['cumprob'] = ["{:.2f}".format(100*prob)
                            for prob in  table['cumprob']]
        table['ra'] = ["{:.2f}".format(center.ra.deg)
                       for center in pointings['center']]
        table['dec'] = ["{:.2f}".format(center.dec.deg)
                        for center in pointings['center']]
        table['time'] = [time.datetime.strftime('%Y-%m-%d %H:%M')
                         for time in pointings['time']]
        table['dt'] = ["{:.2f}".format(dt.jd*24) for dt in pointings['dt']]
        table[['telescope', 'fieldname', 'ra', 'dec', 'time', 'dt',
               'prob', 'cumprob']].write(outputoptions['latex'],
                                         format='latex', overwrite=True)

    if outputoptions.get('pickle'):  # For re-use within Python
        with open(outputoptions['pickle'], 'wb') as outfile:
            pickle.dump(pointings, outfile, protocol=2)

    if plotoptions.get('output'):
        options = dict(sun=plotoptions.get('sun'),
                       moon=plotoptions.get('moon'),
                       coverage=plotoptions.get('coverage'),
                       delay=plotoptions.get('delay'))
        skymap.plot(filename=plotoptions['output'], telescopes=telescopes,
                    date=date, pointings=pointings,
                    geoplot=plotoptions.get('geo'), catalog=catalog,
                    nightsky=nightsky,
                    title=plotoptions.get('title'),
                    options=options)

    return pointings


def print_pointings(pointings):
    print("\n".join(pointings.meta['comments']))
    print("# fieldname  RA       Dec   obs-sky-frac   cum-obs-sky-frac   "
          "tileprob   cum-prob  coverage (%)  telescope  dt (hour)       time",
          end=" ")
    if (len(pointings) and
        np.any([len(sources) > 0 for sources in pointings['sources']])):
        print("# of cat. srcs")
    else:
        print("")
    for pointing in pointings:
        print("{p[fieldname]} {ra:8.3f}  {dec:+8.3f}  {p[prob]:13.6f}  "
              "{p[cumprob]:17.6f}  {p[relprob]:9.6f}  {p[cumrelprob]:9.6f}  "
              "{coverage:12.2f}  {p[telescope]:>9s}  "
              "{dt:10.2f} {p[time].datetime:%Y-%m-%d %H:%M} UT".format(
                  p=pointing, ra=pointing['center'].ra.deg,
                  dec=pointing['center'].dec.deg,
                  coverage=pointing['cumprob']*100,
                  dt=pointing['dt'].jd*24),
              end=' ')
        if len(pointing['sources']):
            print("{:-8d}".format(len(pointing['sources'])))
        else:
            print("")

def main(args=None):
    args = parse_args(args=args)
    set_logging(args.verbose, args.quiet)

    date = parse_date(args.date)
    command = " ".join(sys.argv)
    outputoptions = {'text': args.output,
                     'latex': args.latex,
                     'pickle': args.pickle}
    plotoptions = {'output': args.plot,
                   'geo': args.geoplot,
                   'title': args.title,
                   'sun': args.sun,
                   'moon': args.moon,
                   'objects': [parse_object(obj) for obj in args.object],
                   'coverage': args.plot_coverage,
                   'delay': args.plot_delay}
    if (args.skymap == None) and (args.gaussian == []):
        print('ERROR: Skymap argument is missing. Please provide skymap or gaussian arguments.')
    elif args.skymap != None:
        args.skymap = SkyMap.from_fits(args.skymap)
    else:
        args.skymap = SkyMap.from_position(args.gaussian[0][0],
                                           args.gaussian[0][1],
                                           args.gaussian[0][2])

    pointings = run(args.skymap, args.scope, date=date,
        coverage=args.coverage,
        maxtiles=args.maxtiles, within=args.within,
        nightsky=args.nightsky, catalog=args.catalog,
        tilespath=args.tiles, njobs=args.njobs,
        command=command,
        outputoptions=outputoptions,
        plotoptions=plotoptions)
    print_pointings(pointings)
