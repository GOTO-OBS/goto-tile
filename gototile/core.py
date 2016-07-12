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
from .telescope import (GOTON4, GOTON8, GOTOS4, GOTOS8,
                                SuperWASPN, VISTA, GOTOLS4, GOTOLS8)
from .telescope import build_scope, read_config_file
try:
    FileExistsError
except NameError:
    from .utils import FileExistsError
from .log import set_logging
from .parser import parse_args, parse_date, parse_object


def run(skymap, telescopes, nside=NSIDE, date=None,
        coverage=None, maxtiles=100, within=None,
        nightsky=False, catalog=None, tilespath='./tiles',
        njobs=1, tileduration=None,
        command='', output=None, plot=None):

    if coverage is None:
        coverage = {'min': 0.05, 'max': 0.95}
    if catalog is None:
        catalog = {'path': None, 'key': None}
    if output is None:
        output = {}
    if plot is None:
        plot = {}

    skymap = SkyMap(skymap)
    skymap.regrade(nside=nside)
    date = skymap.header['date-det'] if date is None else date

    pointings, tiledmap, allskymap = calculate_tiling(
        skymap, telescopes, date=date, coverage=coverage,
        maxtiles=maxtiles, within=within,
        nightsky=nightsky, catalog=catalog,
        tilespath=tilespath, njobs=njobs, tileduration=tileduration)

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

    if output.get('text'):
        table = pointings[['prob', 'cumprob', 'telescope']].copy()
        table['prob'] = ["{:.5f}".format(100 * prob)
                         for prob in table['prob']]
        table['cumprob'] = ["{:.5f}".format(100*prob)
                            for prob in  table['cumprob']]
        table['ra'] = ["{:.5f}".format(center.ra.deg)
                       for center in pointings['center']]
        table['dec'] = ["{:.5f}".format(center.dec.deg)
                        for center in pointings['center']]
        # %z was added in Python 3.3, and %Z is deprecated
        table['time'] = [time.datetime.strftime('%Y-%m-%dT%H:%M:%S%z')
                         for time in pointings['time']]
        table['dt'] = ["{:.5f}".format(dt.jd) for dt in pointings['dt']]
        columns = ['telescope', 'ra', 'dec', 'time', 'dt', 'prob', 'cumprob']
        if catalog['path']:
            table['ncatsources'] = [len(sources)
                                    for sources in pointings['sources']]
            columns.append('ncatsources')
        table[columns].write(output['text'], format='ascii.ecsv')

    if output.get('latex'):  # Very similar to output, but with less
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
               'prob', 'cumprob']].write(output['latex'], format='latex')

    if output.get('pickle'):  # For re-use within Python
        with open(output['pickle'], 'wb') as outfile:
            pickle.dump(pointings, outfile, protocol=2)

    if plot.get('output'):
        options = dict(sun=plot.get('sun'), moon=plot.get('moon'),
                       coverage=plot.get('coverage'),
                       delay=plot.get('delay'))
        skymap.plot(plot['output'], telescopes, date, pointings,
                    geoplot=plot['geo'], catalog=catalog,
                    nightsky=nightsky,
                    title=plot.get('title'), objects=plot.get('objects'),
                    options=options)


def main(args=None):
    args = parse_args(args=args)
    set_logging(args.verbose, args.quiet)

    telclasses = {
        'gn4': GOTON4,
        'gn8': GOTON8,
        'gs4': GOTOS4,
        'gs8': GOTOS8,
        'gls4': GOTOLS4,
        'gls8': GOTOLS8,
        'swn': SuperWASPN,
        'vista': VISTA
    }

    telescopes = []
    for scope in args.scope:
        telclass = telclasses[scope]
        telescope = telclass()
        telescopes.append(telescope)
    if args.scopefile:
        telconfigs = read_config_file(args.scopefile)
        for config in telconfigs:
            telescope = build_scope(config)
            try:
                telescope.makegrid(args.tiles)
            except FileExistsError as exc:
                logging.warning(str(exc))
                logging.info("Skipping this grid.")
                logging.info("Remove file if you want to recreate the grid")
            telescopes.append(telescope)
    if not telescopes:
        sys.exit("No telescopes given")

    date = parse_date(args.date)
    command = " ".join(sys.argv)
    output = {'text': args.output, 'latex': args.latex, 'pickle': args.pickle}
    plot = {'output': args.plot,
            'geo': args.geoplot,
            'title': args.title,
            'sun': args.sun,
            'moon': args.moon,
            'objects': [parse_object(obj) for obj in args.object],
            'coverage': args.plot_coverage,
            'delay': args.plot_delay}

    run(args.skymap, telescopes, date=date,
        coverage=args.coverage,
        maxtiles=args.maxtiles, within=args.within,
        nightsky=args.nightsky, catalog=args.catalog,
        tilespath=args.tiles, njobs=args.njobs,
        tileduration=args.exptime, command=command,
        output=output, plot=plot)
