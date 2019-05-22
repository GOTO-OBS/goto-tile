"""Routines to parse user arguments and input"""

import argparse
import os
import sys
from astropy.time import Time, TimeDelta
from astropy import units
from astropy.coordinates import SkyCoord
from . import settings
from . import telescope as telmodule


def parse_args(args=None):

    description = ("This script creates pointings for selected telescopes, "
                   "with given skymap files.")

    parser = argparse.ArgumentParser(
        description = description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-gaussian", nargs=3, type=float, action='append',
                        default=[], help="Create a gaussian skymap with the given "
                        "RA, Dec and 68%% containment radius (in degrees).")
    parser.add_argument("-skymap", help="Skymap FITS file",)
    parser.add_argument('-o', '--output', help="Output file name")
    parser.add_argument('--latex',
                        help="Write LaTeX table of pointings")
    parser.add_argument('--pickle',
                        help="Write the pointing to a pickle file")
    parser.add_argument('--maxtiles', type=int, default=100,
                        help="Maximum number of tiles to return")
    parser.add_argument('--maxfrac', type=float, default=0.95,
                        help="Maximum fraction of visible skymap to tile")
    parser.add_argument('--minfrac', type=float, default=0.05,
                        help="Minimum fraction of visible skymap required to "
                        "attempt tiling")
    parser.add_argument('-s', '--scope',
                        choices=['gn4', 'gn8', 'gs4', 'gs8', 'gls4', 'gls8',
                                 'swasp','vista'],
                        default=[], action='append',
                        help=("Telescope to use. GOTO-4 (default), GOTO-8, "
                              "SuperWASP-North, VISTA. Repeat when using "
                              "multiple telescopes"))
    parser.add_argument("--tiles", default='./tiles/', dest='tiles',
                        help=("File name or base file name of pre-made "
                              "fixed grid of tiles on the sky."))
    parser.add_argument("--makegrid", action="store_true",
                        help=("Create fixed grid of tiles on sky. "
                              "WARNING: Can take some time."))
    parser.add_argument('--makegrid-skip', action='store_true',
                        help="Skip existing grid files")
    parser.add_argument('-S', '--scopefile',
                        help="YAML file with telescope configuration(s)")
    parser.add_argument('--dump-scopes', action='store_true',
                        help="Print a YAML file with standard telescope "
                        "configuration to stdout")
    parser.add_argument('--min-elevation', type=float, action='append',
                        help="Set an alternative minimum elevation in "
                        "degrees. Use this option as many times as "
                        "the --scope option")
    parser.add_argument('--exptime', type=float,
                        help="Use this exposure times, in seconds. "
                        "Applies to *all* telescopes")
    parser.add_argument('-c', '--catalog', 
                        choices=['GLADE','GWGC'],
                        help="Use a catalog to convolve with; specify as an "
                        "astropy readable table format (default catalog: GLADE)")
    parser.add_argument('--catalog-weight-key',
                        help="Field name to serve as a catalog weight. "
                        "Default is no weighting (spatial density only)")
    parser.add_argument('-n', '--nightsky', nargs='?', const=True,
                        help="Use nightsky visbility in tiling/plotting. "
                        "Use the special value 'all' to use all available "
                        "night sky, even the part that has set.")
    parser.add_argument('-d', '--date', nargs='?', default='now',
                        help="Set observation date. If not used, defaults to "
                        "the current date.  If given without argument, "
                        "defaults to the trigger date in the input file(s). "
                        "The optional value can be a date-time string that "
                        "can be parsed by astropy.time.Time, such as "
                        "'2012-12-12T12:12:12'. A single number is interpreted "
                        "as Julian days; use a number with 'mjd' appended to "
                        "specify Modified Julian Days.")
    parser.add_argument('-j', '--jobs', nargs='?', default=1, dest='njobs',
                        help="Number of processes. If specified without a value, "
                        "use all available processes (cores).")
    parser.add_argument("--geoplot", action='store_true',
                        help="Plot in geographic coordinates (lat, lon), "
                        "instead of the default (RA, Dec)")
    parser.add_argument("--plot", nargs='?', const=True,
                        help="Plot in RA-Dec. Optional output file name")
    parser.add_argument("--title", help="Use suppied title in skymap plot.")
    parser.add_argument("--object", nargs=3, action='append',
                        default=[], help="Overplot an object. "
                        "Requires three values: RA, Dec and an object name.")
    parser.add_argument('--sun', action='store_true',
                        help="Plot the Sun position")
    parser.add_argument('--moon', action='store_true',
                        help="Plot the Moon position and phase")
    parser.add_argument('--plot-coverage', action='store_true',
                        help="Plot percentage coverage as outline thickness")
    parser.add_argument('--plot-delay', action='store_true',
                        help="Plot delay as tile transparency")
    parser.add_argument('--timespan', dest='within',
                        help="Only calculate when an area is observable within "
                        "the given amount of time. Default unit is seconds; "
                        "Optionally append a 'd' (day) 'h' (hour), "
                        "'m' (minute) or 's' (second)")
    parser.add_argument('--within', dest='within',
                        help="Alias for --timespan")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='count', default=0,
                       help="Verbose level")
    group.add_argument('-q', '--quiet', action='store_true',
                       help="Turn off warnings")
    args = parser.parse_args(args=args)

    if args.dump_scopes:
        telmodule.print_config_file()
        parser.exit()

    if args.njobs:
        args.njobs = int(args.njobs)

    if args.nightsky is None:
        args.nightsky = False

    # Map args.scope option to actual class names
    telclasses = {
        'gn4': 'GOTON4',
        'gn8': 'GOTON8',
        'gs4': 'GOTOS4',
        'gs8': 'GOTOS8',
        'gls4': 'GOTOLS4',
        'gls8': 'GOTOLS8',
        'swasp': 'SuperWASPN',
        'vista': 'VISTA',
    }
    telescopes = []
    for scope in args.scope:
        name = telclasses[scope]
        telclass = getattr(telmodule, name)
        telescopes.append(telclass())
    if args.min_elevation:
        for i, telescope in enumerate(telescopes):
            telescope.min_elevation = args.min_elevation[i] * units.degree
    args.scope = telescopes

    if args.scopefile:
        telconfigs = telmoculde.read_config_file(args.scopefile)
        for config in telconfigs:
            telescope = telmodule.build_scope(config)
            args.scope.append(telescope)
    if not args.scope:
        sys.exit("No telescopes given")

    if args.exptime:
        settings.TIMESTEP = TimeDelta(args.exptime * units.second)

    if args.within:
        try:
            val = float(args.within) * units.second
        except ValueError:
            val = args.within[:-1]
            if args.within[-1] == 'h':
                val = float(val) * units.hour
            elif args.within[-1] == 'm':
                val = float(val) * units.minute
            elif args.within[-1] == 's':
                val = float(val) * units.second
            elif args.within[-1] == 'd':
                val = float(val) * units.day
            else:
                raise
        args.within = TimeDelta(val)

    args.coverage = {'min': args.minfrac, 'max': args.maxfrac}

    if args.catalog is True:
        if args.catalog == 'GWGC':
            args.catalog = settings.GWGC_PATH
            if not args.catalog_weight_key:
                args.catalog_weight_key = 'weight'
        elif args.catalog == 'GLADE':
            args.catalog = settings.GLADE_PATH
            if not args.catalog_weight_key:
                args.catalog_weight_key = 'weight'
    args.catalog = {'path': args.catalog, 'key': args.catalog_weight_key}

    if args.plot is True:
        if args.skymap:
            args.plot = os.path.split(args.skymap)[-1].split('.')[0] + '.png'
        elif args.gaussian:
            args.plot = 'gaussian.png'
        else:
            args.plot = 'output.png'

    return args


def parse_date(string):
    """Turn a string into an astropy.time.Time date"""
    if string is None:
        return None
    if string == 'now':
        return Time.now()
    if string.lower().endswith('jd'):
        if string.lower().endswith('mjd'):
            return Time(float(string[:-3]), format='mjd', scale='utc')
        return Time(float(string[:-2]), format='jd', scale='utc')
    return Time(string)


def parse_object(args):
    if ':' in args[0] and ':' in args[1]:
        coords = SkyCoord(args[0], args[1], unit=(units.hour, units.degree))
    else:
        try:
            float(args[0]), float(args[1])
        except ValueError:
            coords = SkyCoord(args[0], args[1])
        else:
            coords = SkyCoord(args[0], args[1], unit=units.degree)
    coords.name = args[2]
    return coords


def parse_site(site):
    if site is None:
        return site
    try:
        lon = Angle(site[0])
    except units.UnitsError:
        lon = Angle(site[0], unit=units.hour)
    lat = Angle(site[1], unit=units.degree)
    height = float(site[2])
    location = EarthLocation.from_geodetic(lon, lat, height)
    location.name = site[3] if len(site) == 4 else ""
    return location
