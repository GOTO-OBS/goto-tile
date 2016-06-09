#! /usr/bin/env python

"""Create a grid to use with tileskymap

This script creates an all-sky grid for a given field of view, or
creates a grid from an input file with right ascension and declination
pointings, and a given field of view.

For the first case, the full sky will be gridded into areas; by
default, areas overlap by half their size in right ascension, and half
their size in declination. This allows for the necessary flexibility
when tiling a set of pointings.

The overlap can be changed with the --ra-overlap and --dec-overlap,
which represents the fraction of overlap.

For the second case, a two-column (white-space separated) text file
should given next to the field of view, with the first column the
right ascension (in degrees between 0 and 360) and the second colunm
the declination (between -90 and 90). Each rows thus represents a
pointing on the sky.

"""

import os
import sys
import argparse
import pydoc
import gzip
import pickle
from gototile.grid import tileallsky2


class HelpAction(argparse.Action):
    """Help that pages the doc string when the long option is used
    (compare for example git)"""
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string == '-h':
            parser.print_help()
            parser.exit()
        elif option_string == '--help':
            pydoc.pager(__doc__)
            parser.exit()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('fov-ra', type=float,
                        help="Field-of-view along the Right Ascension, "
                        "in degrees")
    parser.add_argument('fov-dec', type=float,
                        help="Field-of-view along the declination, "
                        "in degrees")
    parser.add_argument('filename',
                        help="Output file name to store grid")
    parser.add_argument('pointings', nargs='?',
                        help="Pointing file (rows of 'ra dec' in degrees)")
    parser.add_argument('--ra-overlap', type=float, default=0.5,
                        help="Amount of overlap between tiles along the RA. "
                        "Ignored when given a pointing file")
    parser.add_argument('--dec-overlap', type=float, default=0.5,
                        help="Amount of overlap between tiles along the "
                        "declination. Ignored when given a pointing file")
    parser.add_argument('--nside', action='append', type=int,
                        help="'nside' parameter to calculate the HEALPIX "
                        "pixels per in a tile. Can be used multiple times.")
    parser.add_argument('-f', '--force', action='store_true',
                        help="Overwrite existing output file")
    parser.add_argument('-h', '--help', nargs=0, action=HelpAction,
                        help="Display this help and exit")
    args = parser.parse_args()
    # Note: positional arguments don't get a hyphen in their name
    # replaced by an underscore
    if not (0 <= getattr(args, 'fov-ra') <= 180):
        parser.error("fov-ra should be between 0 and 180")
    if not (0 <= getattr(args, 'fov-dec') <= 90):
        parser.error("fov-dec should be between 0 and 90")
    if args.ra_overlap is not None:
        if not (0 <= args.ra_overlap <= 0.9):
            parser.error("--ra-overlap should be between 0 and 0.9")
    if args.dec_overlap is not None:
        if not (0 <= args.dec_overlap <= 0.9):
            parser.error("--dec-overlap should be between 0 and 0.9")
    if args.nside is None:
        args.nside = [256]
    return args


def main():
    args = parse_args()
    fov = {'ra': getattr(args, 'fov-ra'),
           'dec': getattr(args, 'fov-dec')}
    overlap = {'ra': args.ra_overlap, 'dec': args.dec_overlap}
    if not args.force and os.path.lexists(args.filename):
        sys.exit("{} already exists; not overwritten".format(args.filename))

    for nside in args.nside:
        gridcoords, tilelist, pixellist = tileallsky2(
            fov, nside, overlap=overlap, nested=True)

    with gzip.GzipFile(args.filename, 'w') as fp:
        pickle.dump((tilelist, pixellist, gridcoords), fp, protocol=2)


if __name__ == "__main__":
    main()
