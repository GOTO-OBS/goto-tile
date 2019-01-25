# GOTO-tile

**GOTO-tile** is a skymap processing and sky tiling module for the GOTO Observatory.

Note this module is Python3 only and has been developed for Linux, otherwise use at your own risk.

## Requirements

GOTO-tile requires several Python modules, which should be included during installation. Notably [Cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html#installing) has several requirements: you'll need Cython3, [Proj](https://proj4.org/install.html) and the C package [GEOS](https://trac.osgeo.org/geos/) installed too.

It should not require any other GOTO-specific modules to be installed.

### Installation

Once you've downloaded or cloned the repository, in the base directory run:

    pip3 install . --user

You should then be able to import the module using `import gototile` within Python.

Several scripts from the `scripts` folder should also be added to your path.

## Usage instructions

The basic method for calling the script with default settings is:

    gototile -s <telescope> -skymap <skymap-file>

or

    gototile -s <telescope> -gaussian <gaussian-position>

where `<telescope>` is one of the predefined telescopes, `<skymap-file>` is the LIGO probability skymap, and `<gaussian-position>` is the position and 68% containment radius.

A list of options and telescopes can be seen with:

    gototile -h

An example command might be:

    gototile -s gn4 --night --catalog --plot --sun --moon -skymap bayestar.fits.gz

or

    gototile -s gn4 --night --catalog --plot --sun --moon -gaussian 60.640 -15.020 7.16

Notes:
    Currently the script assumes greedy tiling is required. However, this may
    be sub-optimal given the large filed-of-view of GOTO, and poor pointing
    from the GW detectors, particularly in the 2-detector scenario, and is
    an avenue of future work.
