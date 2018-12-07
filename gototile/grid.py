from __future__ import division
try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools as it
import gzip
import os
import tempfile
import logging
import multiprocessing
import numpy as np
import healpy
import collections
from copy import copy

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable

from .gridtools import create_grid, get_tile_vertices
from .math import lb2xyz, xyz2lb, intersect
from .math import RAD, PI, PI_2


class PolygonQuery(object):
    def __init__(self, nside, nested):
        self.nside = nside
        self.nested = nested
    def __call__(self, vertices):
        return healpy.query_polygon(self.nside, vertices, nest=self.nested)


class SkyGrid(object):
    """An all-sky grid of defined tiles.

    Parameters
    ----------
    fov : list or tuple or dict of int or float or `astropy.units.Quantity`
        The field of view of the tiles in the RA and Dec directions.
        If given as a tuple, the arguments are assumed to be (ra, dec).
        If given as a dict, it should contains the keys 'ra' and 'dec'.
        If not given units the values are assumed to be in degrees.

    overlap : int or float or list or tuple or dict of int or float, optional
        The overlap amount between the tiles in the RA and Dec directions.
        If given a single value, assumed to be the same overlap in both RA and Dec.
        If given as a tuple, the arguments are assumed to be (ra, dec).
        If given as a dict, it should contains the keys 'ra' and 'dec'.
        default is 0.5 in both axes, minimum is 0 and maximum is 0.9

    kind : str, optional
        The tiling method to use. See `gototile.gridtools.create_grid` for options.
        Default is 'cosine'.
    """

    def __init__(self, fov, overlap=None, kind='cosine'):
        # Parse fov
        if isinstance(fov, (list,tuple)):
            fov = {'ra': fov[0], 'dec': fov[1]}
        for key in ('ra', 'dec'):
            # make sure fov is in degrees
            if not isinstance(fov[key], u.Quantity):
                fov[key] *= u.deg
            else:
                fov[key] = fov[key].to(u.deg)
        self.fov = fov

        # Parse overlap
        if overlap is None:
            overlap = {'ra': 0.5, 'dec': 0.5}
        elif isinstance(overlap, (int, float, u.Quantity)):
            overlap = {'ra': overlap, 'dec': overlap}
        elif isinstance(overlap, (list,tuple)):
            overlap = {'ra': overlap[0], 'dec': overlap[1]}
        for key in ('ra', 'dec'):
            # limit overlap to between 0 and 0.9
            overlap[key] = min(max(overlap[key], 0), 0.9)
        self.overlap = overlap

        # Give the grid a unique name
        self.name = 'allsky-{}x{}-{}-{}'.format(self.fov['ra'].value,
                                                self.fov['dec'].value,
                                                self.overlap['ra'],
                                                self.overlap['dec'])

        # Create the grid
        ras, decs = create_grid(self.fov, self.overlap, kind)
        self.coords = SkyCoord(ras, decs, unit=u.deg)
        self.ntiles = len(self.coords)

        # Get the tile vertices
        self.vertices = get_tile_vertices(self.coords,
                                          self.fov['ra'].value,
                                          self.fov['dec'].value)

        # Give the tiles unique ids
        self.tilenums = np.arange(self.ntiles) + 1
        filllen = len(str(max(self.tilenums)))
        self.tilenames = ['T' + str(num).zfill(filllen) for num in self.tilenums]

    def __eq__(self, other):
        try:
            return self.fov == other.fov and self.overlap == other.overlap
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        template = ('SkyGrid(fov=({}, {}), overlap=({}, {}))')
        return template.format(self.fov['ra'].value, self.fov['dec'].value,
                               self.overlap['ra'], self.overlap['dec'])

    def copy(self):
        """Return a new instance containing a copy of the sky grid data."""
        newgrid = SkyGrid(self.fov, self.overlap)
        return newgrid

    def get_pixels(self, nside, nested=True):
        """Calculate the HEALPix indicies within each tile.

        See the `healpy.pixelfunc.ud_grade()` documentation for the parameters.
        """
        polygon_query = PolygonQuery(nside, nested)
        pool = multiprocessing.Pool()
        pixels = pool.map(polygon_query, self.vertices)
        pool.close()
        pool.join()
        return np.array(pixels)

    def apply_skymap(self, skymap):
        """Apply a SkyMap to the grid.

        This means caculate the contained probabiltiy within each tile.
        The probability contained within each tile will be stored in self.probs.

        Parameters
        ----------
        skymap : `gototile.skymap.SkyMap`
            The sky map to map onto this grid.
        """
        # Calculate which pixels are within the tiles
        self.pixels = self.get_pixels(skymap.nside, skymap.isnested)

        # Calculate the contained probabilities within each tile
        probs = np.array([skymap.skymap[pix].sum() for pix in self.pixels])

        # Store skymap details on the class
        self.skymap = skymap.copy()
        self.nside = skymap.nside
        self.isnested = skymap.isnested
        self.probs = probs

        return probs

    def get_table(self):
        """Return an astropy QTable containing infomation on the defined tiles.

        If a sky map has been applied to the grid the table will include a column with
            the contained probability within each tile.
        """
        col_names = ['tilename', 'ra', 'dec', 'prob']
        col_types = ['U', u.deg, u.deg, 'f8']

        if hasattr(self, 'probs'):
            probs = self.probs
        else:
            probs = np.zeros(self.ntiles)

        table = QTable([self.tilenames, self.coords.ra, self.coords.dec, probs],
                        names=col_names, dtype=col_types)
        return table

    def get_stats(self, nside=128):
        """Return a table containing grid statistics.

        """

        # Get the pixels within each tile
        tile_pixels = self.get_pixels(nside, True)

        # Statistics
        npix = healpy.nside2npix(nside)
        count = np.array([0] * npix)
        for tile_pix in tile_pixels:
            for pix in tile_pix:
                count[pix] += 1
        counter = collections.Counter(count)

        # Make table
        col_names = ['in_tiles', 'pix', 'freq']
        col_types = ['i', 'i', 'f8']

        in_tiles = [i for i in counter]
        pix = [counter[i] for i in counter]
        freq = [counter[i]/npix for i in counter]

        table = QTable([in_tiles, pix, freq],
                       names=col_names, dtype=col_types)
        table['freq'].format = '.4f'
        return table


    def plot(self, color='None', alpha=0.3,
             plot_stats=False, plot_skymap=False,
             gridlines=False,
             orthoplot=False, centre=None,
             filename=None, dpi=300):
        """Plot the grid.

        Parameters
        ----------
        color : str or list or dict, optional

        alpha : float
            alpha channel value for the tile facecolours

        plot_stats : bool, default = False
            use HEALPix to color the plot with the times each pixel is within a tile

        plot_skymap : bool, default = False
            color tiles based on their contained probability
            will fail unless a skymap has been applied to the grid using grid.apply_skymap()

        gridlines : bool, default = False
            show gridlines on the sphere

        orthoplot : bool, default = False
            plot the sphere in a orthographic projection, centred on centre

        centre : float or tuple, default = 0 or (0,45)
            coordinates to centre the orthographic plot on (longitude, latitude)
            if orthoplot=False then the plot will be in the Mollweide projection,
                centred on this float (long=0 by default)
            if orthoplot=True then the sphere will be centred on these coordiantes
             (long=0, lat=45 by default)

        filename : str
            filename to save the plot with
            if not given, plot will be displayed using matplotlib.pyplot.show()

        dpi : int, defualt=300
            dpi to save the plot with

        """
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as\
            FigureCanvas
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import cartopy.crs as ccrs
        from .math import xyz2radec
        from .skymap import read_colormaps
        from .skymaptools import getshape

        if filename:
            fig = Figure()
        else:
            fig = plt.figure()
        if orthoplot:
            if centre is None:
                centre = (0, 45)
            projection = ccrs.Orthographic(central_longitude=centre[0], central_latitude=centre[1])
        else:
            if centre is None:
                centre = 0
            projection = ccrs.Mollweide(central_longitude=centre)
        geodetic = ccrs.Geodetic()
        axes = fig.add_subplot(1, 1, 1, projection=projection)

        axes.set_global()
        if gridlines:
            axes.gridlines()

        read_colormaps()

        # Create the polygons, or load them if already made
        if not hasattr(self, '_poly_cache'):
            # Find tile areas
            radecs = []
            for i, tile in enumerate(self.vertices):
                tile = xyz2radec(*tile.T)
                ra, dec = getshape(tile, steps=5)
                # Need to reverse and transpose to get into format for Polygon
                radec = np.array((ra[::-1], dec[::-1])).T
                radecs.append(radec)

            # Create a collection to plot at once
            polys = PatchCollection([Polygon(radec) for radec in radecs],
                                    cmap='jet',
                                    transform=geodetic)

            # Store the polygons so we don't have to recreate them
            self._poly_cache = copy(polys)
        else:
            polys = copy(self._poly_cache)

        # Plot options
        if plot_skymap and plot_stats:
            raise ValueError('Can only either plot skymap or stats, not both')

        # Set transparency
        polys.set_alpha(alpha)

        # Colour the tiles
        if plot_skymap is True:
            # Colour the tiles by contained probability
            polys.set_cmap('cylon')
            polys.set_array(np.array(self.probs))
            fig.colorbar(polys, ax=axes)

        elif plot_stats is True:
            # Colour in areas based on the number of tiles they are within
            polys.set_facecolor('none')
            nside = 128

            # HealPix for the grid
            npix = healpy.nside2npix(nside)
            ipix = np.arange(npix)
            thetas, phis = healpy.pix2ang(nside, ipix, nest=True)
            pix_ras = np.rad2deg(phis) % 360
            pix_decs = np.rad2deg(np.pi / 2 - thetas % np.pi)

            # Statistics
            tile_pixels = self.get_pixels(nside, True)
            pix_freq = np.array([0] * npix)
            for tile_pix in tile_pixels:
                for pix in tile_pix:
                    pix_freq[pix] += 1

            # Plot HealPix points coloured by tile count
            # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
            cmap = plt.cm.jet
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (.5,.5,.5,1.0)
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

            bounds = np.linspace(0,6,7)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

            points = axes.scatter(pix_ras, pix_decs, s=1, transform=geodetic,
                                  c=pix_freq, cmap='gist_rainbow', norm=norm)
            fig.colorbar(points)

        else:
            # Colour in tiles with the colour given
            if isinstance(color, dict):
                # Should be a dict with keys as tile names
                color_array = ['none'] * self.ntiles
                for k in color.keys():
                    i = self.tilenames.index(k)
                    color_array[i] = color[k]
                color = np.array(color_array)

            if isinstance(color, (list, tuple, np.ndarray)):
                # A list-like of colours, should be same length as number of tiles
                if not len(color) == self.ntiles:
                    raise ValueError('List of colors must be same length as grid.ntiles')

                # Could be a list of weights or a list of colours
                try:
                    polys.set_facecolor(np.array(color))
                except:
                    try:
                        polys.set_array(np.array(color))
                        fig.colorbar(polys, ax=axes)
                    except:
                        raise ValueError('Invalid entries in color array')
            else:
                polys.set_facecolor(color)

        # Plot the tiles
        axes.add_collection(polys)

        # Also plot on the lines over the top
        polys2 = copy(self._poly_cache)
        polys2.set_facecolor('none')
        polys2.set_edgecolor('black')
        polys2.set_alpha(0.3)
        axes.add_collection(polys2)

        # Save or display the plot
        if filename:
            canvas = FigureCanvas(fig)
            canvas.print_figure(filename, dpi=dpi)
        else:
            plt.show()
