"""Core functionality"""

from __future__ import absolute_import, print_function, division

import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import astropy
from .skymaptools import calculate_tiling
from .skymap import SkyMap
from .settings import NSIDE
from . import telescope as telmodule
from .telescope import build_scope, read_config_file
from .utils import pointings_to_text
from .log import set_logging
from .parser import parse_args, parse_date, parse_object
try:
    FileExistsError
except NameError:
    from .utils import FileExistsError


def run(skymap, telescopes, nside=NSIDE, date=None,
        coverage=None, maxtiles=100, within=None,
        nightsky=False, catalog=None, tilespath='./tiles',
        njobs=1, command='',
        outputoptions=None, plotoptions=None):

    if coverage is None:
        coverage = {'min': 0.05, 'max': 0.95}
    if catalog is None:
        catalog = {'path': None, 'key': None}
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

    if not isinstance(skymap, SkyMap):
        skymap = SkyMap(skymap)
    skymap.regrade(nside=nside)

    date = skymap.header['date-det'] if date is None else date

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

    print("\n".join(pointings.meta['comments']))
    print("#     RA       Dec   obs-sky-frac   cum-obs-sky-frac   "
          "tileprob   cum-prob  coverage (%)  telescope  dt (hour)       time",
          end=" ")
    if (len(pointings) and
        np.any([len(sources) > 0 for sources in pointings['sources']])):
        print("# of cat. srcs")
    else:
        print("")
    for pointing in pointings:
        print("{ra:8.3f}  {dec:+8.3f}  {p[prob]:13.6f}  "
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

    if outputoptions.get('text'):
        table = pointings_to_text(pointings, catalog=catalog)
        table.write(outputoptions['text'], format='ascii.ecsv')

    if outputoptions.get('latex'):  # Very similar to output, but with less
                             # precision (more human readable when rendered)
        table = pointings[['prob', 'cumprob', 'telescope']].copy()
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
        table[['telescope', 'ra', 'dec', 'time', 'dt',
               'prob', 'cumprob']].write(outputoptions['latex'], format='latex')

    if outputoptions.get('pickle'):  # For re-use within Python
        with open(outputoptions['pickle'], 'wb') as outfile:
            pickle.dump(pointings, outfile, protocol=2)

    if plotoptions.get('output'):
        options = dict(sun=plotoptions.get('sun'),
                       moon=plotoptions.get('moon'),
                       coverage=plotoptions.get('coverage'),
                       delay=plotoptions.get('delay'))
        skymap.plot(plotoptions['output'], telescopes, date, pointings,
                    geoplot=plotoptions['geo'], catalog=catalog,
                    nightsky=nightsky,
                    title=plotoptions.get('title'),
                    objects=plotoptions.get('objects'),
                    options=options)

    return pointings


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

    run(args.skymap, args.scope, date=date,
        coverage=args.coverage,
        maxtiles=args.maxtiles, within=args.within,
        nightsky=args.nightsky, catalog=args.catalog,
        tilespath=args.tiles, njobs=args.njobs,
        command=command,
        outputoptions=outputoptions,
        plotoptions=plotoptions)
