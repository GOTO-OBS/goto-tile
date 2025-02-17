#! /usr/bin/env python3
"""Core command-line script for the gototile package."""

import argparse

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time, TimeDelta
from matplotlib import pyplot as plt

from .grid import SkyGrid
from .skymap import SkyMap


def run(
    grid,
    skymap=None,
    simulate=True,
    date=None,
    duration=1,
    site=None,
    mounts=1,
    source=None,
    max_tiles=200,
    min_prob=0,
    contour=0.95,
    min_alt=30,
    twilight=-12,
    exptime=300,
    airmass_weight=0.1,
    verbose=False,
    outfile=None,
    plot=None,
):
    if skymap is None:
        # Just output the coordinates for the grid tiles
        table = grid.get_table()
        table = table.to_pandas()[['tilename', 'ra', 'dec']]
        print(table)

        if outfile is not None:
            print('Saving table to', outfile)
            table.to_csv(outfile)

        if plot is not None:
            print('Saving tile plot to', plot)
            grid.plot(filename=plot, title='', dpi=150)

        # That's all
        return

    # Apply the skymap to the grid to find the contained probability within each tile
    print('Applying probability skymap to grid...')
    grid.apply_skymap(skymap)

    if not simulate:
        table = grid.get_table()
        table.sort('prob', reverse=True)
        table = table.to_pandas()
        print(table)

        if outfile is not None:
            print('Saving table to', outfile)
            table.to_csv(outfile)

        if plot is not None:
            print('Saving tile plot to', plot)

            fig = plt.figure(figsize=(12, 6), dpi=150)
            axes = plt.axes(projection='astro hours mollweide')
            axes.grid()

            # Add the tiles coloured by probability
            tiles = grid.plot_tiles(axes, array=grid.probs, ec='none', alpha=0.6, zorder=1)
            grid.plot_tiles(axes, fc='none', ec='0.7', lw=0.3, zorder=1.2)

            # Add the skymap contours
            skymap.plot_contours(
                axes, levels=[0.5, 0.9], colors='black', linewidths=0.5, zorder=1.3
            )

            # Add a colorbar
            tiles.set_cmap('cylon')
            fig.colorbar(
                tiles,
                ax=axes,
                fraction=0.02,
                pad=0.05,
                label='Tile contained probability',
                format=lambda x, _: f'{x:.0%}',
            )

            # Save the figure
            plt.savefig(plot)
            plt.close(fig)

        # That's all
        return

    # We want to simulate observing the tiles with the telescope
    # First select tiles based on the given limits
    grid_tiles = grid.get_table()
    mask = grid_tiles['contour'] < contour & grid_tiles['prob'] > min_prob
    selected_tiles = grid_tiles[mask]
    selected_tiles.sort('prob', reverse=True)
    if max_tiles > 0 and len(selected_tiles) > max_tiles:
        # Select only the N highest probability tiles
        selected_tiles = selected_tiles[:max_tiles]
    print(f'Selected {len(selected_tiles)} tiles based on given limits.')
    print(selected_tiles.to_pandas())
    all_tiles = list(selected_tiles['tilename'])
    print(f'Total probability covered: {grid.get_probability(all_tiles):.2%}')

    # Then we want to run through observing with the given telescope(s) at the given site
    # We will use a simplified version of the GOTO scheduling system, which selects the tiles based
    # on a weighted combination of contained probability and airmass.
    if date is None:
        date = Time.now()
    if site is None:
        site = EarthLocation.of_site('lapalma')
    print(f'Simulating {duration} days starting from {date}...')
    tile_times_observed = np.zeros(len(selected_tiles), dtype=int)
    start_date = date
    end_date = start_date + TimeDelta(duration * u.day)
    while date < end_date:
        if verbose:
            print(date.iso, end=' -   ')

        # First check if the Sun is above the horizon
        altaz_frame = AltAz(obstime=date, location=site)
        sun = get_sun(date)
        if sun.transform_to(altaz_frame).alt > twilight * u.deg:
            if verbose:
                print('daytime')

        else:
            # Find which of the selected tiles are above the horizon
            tile_coords = SkyCoord(selected_tiles['ra'], selected_tiles['dec'])
            tile_altaz = tile_coords.transform_to(altaz_frame)
            horizon_mask = tile_altaz.alt > min_alt * u.deg
            if sum(horizon_mask) == 0:
                if verbose:
                    print('no tiles are visible')
            else:
                # Calculate tile scores based on times observed, contained probability and airmass.
                # Priority is first given to the tiles that have been observed the least time.
                # If there are multiple tiles observed the same time then a tiebreaker is used.
                # Airmass is weighted relative to the contained probability by a factor of X
                # (default X=0.1), and the total score is always between 0 and 1 - lower is better.
                tile_obs = tile_times_observed[horizon_mask]
                tile_probs = 1 - selected_tiles['prob'][horizon_mask]
                tile_airmass = tile_altaz.secz[horizon_mask]
                tile_tiebreaker = tile_probs + airmass_weight * (2 - tile_airmass)
                tile_ranks = tile_obs + tile_tiebreaker / (1 + airmass_weight)

                # Select the N highest scoring tiles, where N is the number of mounts at the site
                highest_indexes = tile_ranks.argsort()[:mounts]
                highest_tiles = np.array(all_tiles)[horizon_mask][highest_indexes]
                if verbose:
                    print(f'observing {highest_tiles} (v={sum(horizon_mask)})')

                # Mark those tiles as observed
                observed_mask = [t in highest_tiles for t in np.array(all_tiles)]
                tile_times_observed[observed_mask] += 1

        date += TimeDelta(exptime * u.s)

    print()
    print(f'Made {sum(tile_times_observed)} observations of {sum(tile_times_observed > 0)} tiles')
    obs_dict = {all_tiles[i]: tile_times_observed[i] for i in range(len(all_tiles))}
    all_obs = [obs_dict[t] if t in obs_dict else 0 for t in grid.tilenames]
    table = grid.get_table()
    table['selected'] = [t in all_tiles for t in grid.tilenames]
    table['nobs'] = all_obs
    tt = table.copy()
    tt.sort('nobs')
    tt.reverse()
    print(tt.to_pandas())
    print()

    if source:
        source_tiles = grid.get_tile(source, overlap=True)
        print(
            f'The source coordinates were located in tile{"s" if len(source_tiles) > 1 else ""}',
            source_tiles,
        )
        source_nobs = sum(obs_dict[tile] if tile in obs_dict else 0 for tile in source_tiles)
        print(f'The source location was observed {source_nobs} times')
    else:
        source_tiles = []
    print()

    if outfile is not None:
        print('Saving table to', outfile)
        table.to_pandas().to_csv(outfile)

    if plot is not None:
        # TODO: REPLACE WITH PROPER PLOT
        print('Saving observation plot to', plot)
        visible_tiles = grid.get_visible_tiles(
            site, time_range=(start_date, end_date), alt_limit=min_alt, sun_limit=twilight
        )
        non_visible_tiles = [t for t in grid.tilenames if t not in visible_tiles]

        grid.plot(
            filename=plot,
            title='',
            dpi=150,
            plot_contours=True,
            color=obs_dict,
            discrete_colorbar=True,
            highlight=[source_tiles, non_visible_tiles],
            highlight_color=['red', '0.4'],
            coordinates=source,
        )


def main():
    def date_validator(date):
        try:
            if date == 'now':
                date = Time.now()
            else:
                date = Time(date)
        except ValueError:
            msg = "invalid date: '{}' not a recognised format".format(date)
            raise argparse.ArgumentTypeError(msg)
        return date

    def site_validator(site):
        try:
            site = EarthLocation.of_site(site)
        except ValueError:
            msg = "unrecognised site: '{}', check EarthLocation.get_site_names().".format(site)
            raise argparse.ArgumentTypeError(msg)
        return site

    description = 'This script creates pointings for selected telescopes, with given skymap files.'
    parser = argparse.ArgumentParser(description=description)

    # Telescope options
    group = parser.add_argument_group(
        'grid options',
        'Either chose a pre-defined telescope from the list given or define a custom grid.',
    )
    mxg = group.add_mutually_exclusive_group(required=True)
    mxg.add_argument(
        '-t',
        '--telescope',
        type=str,
        choices=SkyGrid.get_named_grids().keys(),
        help='Which pre-defined telescope system to simulate.',
    )
    mxg.add_argument(
        '-g',
        '--grid',
        type=float,
        nargs=4,
        metavar=('FOV_RA', 'FOV_DEC', 'OVERLAP_RA', 'OVERLAP_DEC'),
        help='Define custom sky grid parameters: field of view (in degrees) and '
        'overlap (0-0.9) in RA and Dec axes.',
    )

    # Skymap options
    group = parser.add_argument_group('skymap options')
    mxg = group.add_mutually_exclusive_group()
    mxg.add_argument(
        '-s',
        '--skymap',
        type=str,
        metavar='PATH',
        help='Path to skymap FITS file',
    )
    mxg.add_argument(
        '-G',
        '--gaussian',
        type=float,
        nargs=3,
        metavar=('RA', 'DEC', 'RADIUS'),
        help=(
            'Create a gaussian skymap with the given '
            'RA, Dec and 68%% containment radius (in degrees).'
        ),
    )

    # Simulation options
    group = parser.add_argument_group('simulation options')
    group.add_argument(
        '--simulate',
        action='store_true',
        help='Simulate observations of the given skymap. Note this requires either '
        'the --skymap or --gaussian options.',
    )
    group.add_argument(
        '-d',
        '--date',
        type=date_validator,
        default='now',
        help='Date to start observation simulations, in any format that can be '
        'parsed by `astropy.time.Time` (default=Time.now())',
    )
    group.add_argument(
        '-D',
        '--duration',
        type=float,
        default=1,
        help='Number of 24-hour days to simulate (default=1)',
    )
    group.add_argument(
        '-S',
        '--site',
        type=site_validator,
        default='lapalma',
        help=(
            'Site to simulate observing from, any string that can be parsed '
            'by `astropy.coordinates.EarthLocation.of_site()` (default=LaPalma)'
        ),
    )
    group.add_argument(
        '-m',
        '--mounts',
        type=int,
        default=1,
        help=('Number of independent mounts to simulate observing with (default=1)'),
    )
    group.add_argument(
        '-C',
        '--source-coords',
        type=float,
        nargs=2,
        metavar=('RA', 'DEC'),
        help=('Coordinates of the skymap source, in degrees'),
    )
    group.add_argument(
        '--max-tiles',
        type=int,
        default=200,
        help='Maximum number of tiles to select for observing (default=0)',
    )
    group.add_argument(
        '--min-prob', type=float, default=0, help='Minimum probability to select tiles (default=0)'
    )
    group.add_argument(
        '-c',
        '--contour',
        type=float,
        default=0.95,
        help='Probability contour level to select tiles (default=0.95)',
    )
    group.add_argument(
        '--min-alt', type=float, default=30, help='Telescope horizon limit in degrees (default=30)'
    )
    group.add_argument(
        '--twilight',
        type=float,
        default=-12,
        help='Maximum hight of the Sun to allow observing, in degrees (default=-12)',
    )
    group.add_argument(
        '--exptime',
        type=float,
        default=300,
        help='Time spent by each telescope observing each tile, in seconds. '
        'Should include readout time (default=300)',
    )
    group.add_argument(
        '--airmass-weight',
        type=float,
        default=0.1,
        help='Realative amount to weight airmass vs tile probability when '
        'calculating tile scores (default=0.1).',
    )

    # Output options
    group = parser.add_argument_group('output options')
    group.add_argument(
        '-v', '--verbose', action='store_true', help='Print additional logging infomation.'
    )
    group.add_argument(
        '-o', '--outfile', metavar=('FILENAME'), help='Save tile table to the given filename.'
    )
    group.add_argument(
        '-p', '--plot', metavar=('FILENAME'), help='Save a sky plot to the given filename'
    )

    # Parse args
    args = parser.parse_args()

    # Select the grid based on the system
    if args.telescope in SkyGrid.get_named_grids():
        fov, overlap, kind = SkyGrid.get_named_grids()[args.telescope]
        print(f'Using defined sky grid "{args.telescope}":', end=' ')
        print(f'fov={fov}, overlap={overlap}, kind={kind}')
        grid = SkyGrid(fov, overlap, kind)
    elif args.grid:
        fov = (args.grid[0], args.grid[1])
        overlap = (args.grid[2], args.grid[3])
        kind = 'enhanced1011'  # TODO: add grid method arg?
        print(f'Using new sky grid: fov={fov}, overlap={overlap}')
        grid = SkyGrid(fov, overlap, kind)
    else:
        raise ValueError('Missing telescope or grid parameters')
    print(f'Generated grid containing {grid.ntiles} tiles')

    # Create skymap from given path or position
    if args.skymap:
        print('Loading skymap...')
        skymap = SkyMap.from_fits(args.skymap)
        print('Skymap loaded')
    elif args.gaussian:
        print('Creating skymap...')
        skymap = SkyMap.from_position(args.gaussian[0], args.gaussian[1], args.gaussian[2])
    else:
        skymap = None

    # Get source coordinates
    if args.source_coords is not None:
        source = SkyCoord(args.source_coords[0] * u.deg, args.source_coords[1] * u.deg)
    else:
        source = None

    run(
        grid,
        skymap,
        simulate=args.simulate,
        date=args.date,
        duration=args.duration,
        site=args.site,
        mounts=args.mounts,
        source=source,
        max_tiles=args.max_tiles,
        min_prob=args.min_prob,
        contour=args.contour,
        min_alt=args.min_alt,
        twilight=args.twilight,
        exptime=args.exptime,
        airmass_weight=args.airmass_weight,
        verbose=args.verbose,
        outfile=args.outfile,
        plot=args.plot,
    )
