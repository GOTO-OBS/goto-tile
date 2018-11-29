GOTO-tile reads in single/multiple skymaps and attempts to provide the
best tiling layout using a 'greedy tile' strategy. The version here is
designed for use during the early stages of Advanced LIGO/VIRGO using
the GOTO prototype observatory. It is also capable with Fermi reported position (final position).

The user can choose a range of options, such as tiling with galaxy
weighting, or selecting between the two GOTO configurations (4- and
8-astrograph designs).

Output is in the form of a list of tiles, ordered by probability, as
well as optional plots (geocentric, or celestial).

Installation
============

GOTO-tile relies on a few Python packages:
- numpy
- astropy
- healpy
- PyYAML
- pyephem

If you want the plotting abilities as well, you will also need:
- matplotlib
- basemap

The easiest way to install all requirements is:

    $ pip install -r requirements.txt

Note that basemap requires the geos C library, which is usually found
in package managers as `libgeos-dev` or `libgeos-devel`.


You can install GOTO-tile and its dependencies directly from GitHub,
through:

    $ pip install git+https://github.com/GOTO-OBS/goto-tile.git#egg=gototile

Note that if you install it this way, the basemap will not be
installed (matplotlib will be installed, because it's a requirement of
Healpy. It is, however, still not necessary to have it installed when
running GOTO-tile in its non-plotting mode: healpy will work fine
without it).



Running GOTO-tile
=================

The basic method for calling the script with default settings is:

    $ gototile -s<telescope> -skymap<skymap-file>

or

    $ gototile -s<telescope> -gaussian<gaussian-position>

where <telescope> is one of the predefined telescope,
<skymap-file> is the LGIO probability skymap,
and <gaussian-position> is the position and 68% containment radius.

A list of options and telescopes can be seen with:

    $ gototile -h


An example command might be:

    gototile -s gn4 --night --catalog --plot --sun --moon -skymap bayestar.fits.gz

or

    gototile -s gn4 --night --catalog --plot --sun --moon -gaussian 60.640 -15.020 7.16

Notes:
    Currently the script assumes greedy tiling is required. However, this may
    be sub-optimal given the large filed-of-view of GOTO, and poor pointing
    from the GW detectors, particularly in the 2-detector scenario, and is
    an avenue of future work.


Additional scripts
==================

The following two scripts are from previous versions of GOTO-tile.
Your mileage may vary as to how well they work and perform.

Running f2ytile
---------------

The script f2ytile is essentially a wrapper around the tileskymap script, that
allows the user to run the tiling algorith across the first2years simulated
skymaps produced by Singer et al. (http://www.ligo.org/scientists/first2years/).
The usage and list of user-configurable options are:

usage: f2ytile [-h] [--simpath SIMPATH] [--out OUT] [--log LOG]
               [--years YEARS] [--scopes SCOPES] [-d [DATE]] [--args ARGS]

This script creates pointings for selected telescopes, with given skymap
files.

optional arguments:
  -h, --help            show this help message and exit
  --simpath SIMPATH     Input folder containing groups of maps (e.g. by year)
                        (default: ./)
  --out OUT             Output folder for plots and text files (default: None)
  --log LOG             Output log file name (default: f2ytile.log)
  --years YEARS         Folder names within input folder containing simulated
                        maps, comma separated. (default: 2015,2016)
  --scopes SCOPES       Telescope variations to use for tiling, comma
                        separated (default: g4,g8)
  -d [DATE], --date [DATE]
                        Set observation date. If not used, defaults to the
                        current date. If given without argument, defaults to
                        the trigger date in the input file(s). The optional
                        argument can be a date-time string that can be parsed
                        by astropy.time.Time, such as '2012-12-12T12:12:12'. A
                        single number is interpreted as Julian days; use a
                        number with 'mjd' appended to specify Modified Julian
                        Days. (default: now)
  --args ARGS           Additional arguments to be passed to tileskymap (e.g.
                        --plot, --geoplot, --nightsky (default: )

With an example command of:

f2ytile --out nightskyinjgal --log nightskyinjgal.log --years 2015 --scopes g4 --simpath /storage/astro2/phsnap/LIGO/Skymaps --date --args "--plot --injgal --usegals --nightsky"

Notes:
    The script allows users to provide all extra flags to the tileskymap script
    using the --args option. The options to be passed must be contained within
    quotes. If only one option is passed, for example to plot, then a space must
    be added at the end. For example: --args "--plot ". This is a bug that I
    have not been able to figure out just yet.

Running postmap
---------------

The postmap script is included to generate plots, combined data and basic
statistics on the tiles generated by f2ytile.

usage: postmap [-h] [--first] [--out OUT] [--tiles TILES] [--simpath SIMPATH]
               [--lc LC] [-s {g4,g8,swn}] [--tiledists] [--injdists]
               [--visible] [--mags] [--maglim MAGLIM] [--exptime EXPTIME]
               [-d [DATE]]

This script creates pointings for selected telescopes, with given skymap
files.

optional arguments:
  -h, --help            show this help message and exit
  --first               Make tilefiles? (default: False)
  --out OUT             Output folder for plots and text files (default: None)
  --tiles TILES         Location of f2y tiling algorithm output (default: ./)
  --simpath SIMPATH     Input folder containing original first2years
                        simulations (default: /storage/astro2/phsnap/Skymaps)
  --lc LC               Input folder containing kilonova lightcurve
                        simulations (default: /storage/astro2/phsnap/lightcurv
                        es/GOTO/ns_merger_mags/)
  -s {g4,g8,swn}, --scope {g4,g8,swn}
                        Telescope to use. GOTO-4, GOTO-8, SuperWASP-North.
                        (default: g4)
  --tiledists           Find angular distances between successive tiles
                        (default: False)
  --injdists            Find angular distances between injection location and
                        loudest pixel (default: False)
  --visible             Check number of injections above horizon (default:
                        False)
  --mags                Check number of injections above limiting mag
                        (default: False)
  --maglim MAGLIM       Limiting magnitude above which kilonova is visible
                        (default: 21.0)
  --exptime EXPTIME     Exposure time in mins of observations (used for
                        working out total time for all tiles) (default: 5.0)
  -d [DATE], --date [DATE]
                        Set observation date. If not used, defaults to the
                        current date. If given without argument, defaults to
                        the trigger date in the input file(s). The optional
                        argument can be a date-time string that can be parsed
                        by astropy.time.Time, such as '2012-12-12T12:12:12'. A
                        single number is interpreted as Julian days; use a
                        number with 'mjd' appended to specify Modified Julian
                        Days. (default: now)

An example command to produce output from the example f2ytile command above:

postmap -s g4 --tiledists --injdists --visible --mags --simpath /storage/astro2/phsnap/LIGO/Skymaps --tiles nightskyinjgal --out nightskyinjgal/postmap --date --first
