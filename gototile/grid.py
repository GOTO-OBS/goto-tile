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


from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as\
    FigureCanvas
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from .math import cartesian_to_celestial
from .skymap import read_colormaps



from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable

from .skymaptools import pix2coord
from .gridtools import create_grid, get_tile_vertices, get_tile_edges
from .math import RAD, PI


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
        Default is 'minverlap'.
    """

    def __init__(self, fov, overlap=None, kind='minverlap'):
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

        # Save kind
        self.kind = kind
        self.algorithm = kind

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
        self.vertices = get_tile_vertices(self.coords, self.fov)

        # Give the tiles unique ids
        self.tilenums = np.arange(self.ntiles) + 1
        filllen = len(str(max(self.tilenums)))
        self.tilenames = ['T' + str(num).zfill(filllen) for num in self.tilenums]

    def __eq__(self, other):
        try:
            return (self.fov == other.fov and
                    self.overlap == other.overlap and
                    self.kind == other.kind)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        template = ('SkyGrid(fov=({}, {}), overlap=({}, {}), kind={})')
        return template.format(self.fov['ra'].value, self.fov['dec'].value,
                               self.overlap['ra'], self.overlap['dec'],
                               self.kind)

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

    def _pixels_from_tilenames(self, tilenames):
        """Get the unique pixels contained within the given tile(s)."""
        if isinstance(tilenames, list):
            # Multiple tiles
            indexes = [self.tilenames.index(tile) for tile in tilenames]
            pixels = []
            for i in indexes:
                pixels += list(self.pixels[i])
            pixels = list(set(pixels))  # remove duplicates
        else:
            # An individual tile
            index = self.tilenames.index(tilenames)
            pixels = self.pixels[index]

        return pixels

    def get_probability(self, tilenames):
        """Return the contained probability within the given tile(s).

        If multiple tiles are given, the probability only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the probability within.
        """
        if not hasattr(self, 'probs'):
            raise ValueError('Grid does not have a SkyMap applied')

        # Get pixels
        pixels = self._pixels_from_tilenames(tilenames)

        # Sum the probability within those pixels
        prob = self.skymap.skymap[pixels].sum()

        return prob

    def get_area(self, tilenames):
        """Return the sky area contained within the given tile(s) in square degrees.

        If multiple tiles are given, the area only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the area of.
        """
        # Get pixels
        pixels = self._pixels_from_tilenames(tilenames)

        # Each pixel in the skymap has the same area (HEALPix definition)
        # So just multiply that by number of pixels
        area = self.skymap.pixel_area * len(pixels)

        return area

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

    def plot(self, filename=None, dpi=300, orthoplot=False, center=(0,45),
             color=None, linecolor=None, linewidth=None, alpha=0.3,
             highlight=None, coordinates=None,
             plot_skymap=False, plot_tilenames=False):
        """Plot the grid.

        Parameters
        ----------
        filename : str, optional
            filename to save the plot to
            if not given then the plot will be displayed with plt.show()

        dpi : int, optional
            DPI to save the plot at
            default is 300

        orthoplot : bool, default = False
            plot the sphere in a orthographic projection, centred on `centre`

        center : tuple or `astropy.coordinates.SkyCoord`, default (0,45)
            coordinates to center the orthographic plot on
            if given as a tuple units will be considered to be degrees

        highlight : list or lsit of list, optional
            a list of tile names (as in SkyGrid.tilenames) to highlight
            if a 2d list each set will be highlighted with a different color

        color : str or list or dict, optional
            if str all tiles will be colored using that string
            if list must have length of SkyGrid.ntiles
            if dict the keys should be tile names (as in SkyGrid.tilenames)

        linecolor : str or list or dict, optional
            if str all tiles' outlines will be colored using that string
            if list must have length of SkyGrid.ntiles
            if dict the keys should be tile names (as in SkyGrid.tilenames)

        linewidth : float or list or dict, optional
            if float all tiles' outlines will be set to that width
            if list must have length of SkyGrid.ntiles
            if dict the keys should be tile names (as in SkyGrid.tilenames)

        alpha : float, optional
            all tiles will be set to that alpha
            default = 0.3

        coordinates : `astropy.coordinates.SkyCoord`, optional
            any coordinates to also plot on the image

        plot_skymap : bool, default = False
            color tiles based on their contained probability
            will fail unless a skymap has been applied to the grid using SkyGrid.apply_skymap()

        plot_tilenames : bool, default = False
            plot the name of each tile in its centre
            WARNING: plots can take a long time to render

        """
        fig = plt.figure(figsize=(8,6))
        if not orthoplot:
            axes = plt.axes(projection='astro hours mollweide')
        else:
            if isinstance(center, tuple):
                center = SkyCoord(center[0], center[1], unit='deg')
            axes = plt.axes(projection='astro globe', center=center)
        axes.grid()
        transform = axes.get_transform('world')

        # Create the tile polygons
        polygons = []
        tilenames = []
        for vertices, tilename in zip(self.vertices, self.tilenames):
            # vertices is a (4,3) numpy array - 4 vertices each with x,y,z cartesian coordinates
            # Just plotting those courners with Polygons will draw straight lines between them,
            # which isn't correct since we're on a sphere.
            # Instead, get some intermediate points along the edges by drawing great circles.
            points = get_tile_edges(vertices, steps=5)

            # Convert point coordinates to ra,dec
            ra, dec = cartesian_to_celestial(*points.T)

            # Check if the tile passes over the RA=0 line:
            overlaps_meridian = any(ra<90) and any(ra>270)
            if (not overlaps_meridian) or orthoplot:
                # Need to reverse and transpose to get into the correct format for Polygon
                polygons.append(Polygon(np.array((ra[::-1], dec[::-1])).T))
                tilenames.append(tilename)
            else:
                # Annoyingly tiles that pass over the edge of the plot won't be filled
                # This only applies in 'astro hours mollweide' mode
                # The best workaround is to plot two Polygons, one on each side
                ra1 = ra.copy()
                ra1[ra<180] = 360
                polygons.append(Polygon(np.array((ra1[::-1], dec[::-1])).T))
                ra2 = ra.copy()
                ra2[ra>180] = 0
                polygons.append(Polygon(np.array((ra2[::-1], dec[::-1])).T))
                # The reason for the tilename array is so we can colour both at once
                # See where we deal with colours below, it hopefully makes sense there
                tilenames.extend((tilename, tilename))
        self._polygons = polygons
        self._tilenames = tilenames

        # Create a map between the origional tiles and the polygons
        new_indexes = [np.where(np.array(self.tilenames)==name)[0][0] for name in tilenames]

        # Create a collection to plot all at once
        polys = PatchCollection(polygons, transform=transform)

        # Plot the tiles
        polys.set_facecolor('blue')
        polys.set_edgecolor('none')
        polys.set_linewidth(0)
        polys.set_alpha(alpha)
        polys.set_zorder(2)
        axes.add_collection(polys)

        # Also plot on the lines over the top
        polys2 = copy(polys)
        polys2.set_facecolor('none')
        polys2.set_edgecolor('black')
        polys2.set_linewidth(1)
        polys2.set_alpha(alpha)
        polys2.set_zorder(3)
        axes.add_collection(polys2)

        # Plot tile names
        if plot_tilenames:
            for name, coord in zip(self.tilenames, self.coords):
                plt.text(coord.ra.deg, coord.dec.deg, name,
                         color='k', weight='bold', fontsize=6,
                         ha='center', va='center', clip_on=True,
                         transform=transform)

        # Plot skymap probabilities
        if plot_skymap is True:
            if not hasattr(self, 'skymap'):
                raise ValueError('SkyGrid does not have a SkyMap applied')

            # Set the probability array to the color array, and use the LIGO colormap
            # Plot underneath the other polys, so you can overlay the other tiles (e.g. visibility)
            polys0 = copy(polys)
            polys0.set_array(np.array(self.probs[new_indexes]))
            polys0.set_cmap('cylon')
            polys0.set_alpha(0.5)
            polys0.set_zorder(1)
            axes.add_collection(polys0)
            fig.colorbar(polys0, ax=axes, fraction=0.02, pad=0.05)
            polys.set_facecolor('none')

        # Plot tile colors
        if color is not None:
            if isinstance(color, dict):
                # Should be a dict with keys as tile names
                try:
                    color_array = np.array(['none'] * len(tilenames), dtype=object)
                    for k in color.keys():
                        # Thanks to the edge tiles there may be multiple Polygonswith the same name
                        i = [i for i, x in enumerate(tilenames) if x == k]
                        color_array[i] = color[k]
                    polys.set_facecolor(np.array(color_array))
                except:
                    try:
                        color_array = np.array([0] * len(tilenames))
                        for k in color.keys():
                            i = [i for i, x in enumerate(tilenames) if x == k]
                            color_array[i] = color[k]
                        polys.set_array(np.array(color_array))
                        fig.colorbar(polys, ax=axes, fraction=0.02, pad=0.05)
                    except:
                        raise ValueError('Invalid entries in color array')

            elif isinstance(color, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(color) == self.ntiles:
                    raise ValueError('List of colors must be same length as grid.ntiles')

                # Could be a list of weights or a list of colors
                try:
                    polys.set_facecolor(np.array(color[new_indexes]))
                except:
                    try:
                        polys.set_array(np.array(color[new_indexes]))
                        fig.colorbar(polys, ax=axes, fraction=0.02, pad=0.05)
                    except:
                        raise ValueError('Invalid entries in color array')

            else:
                # Might just be a string color name
                polys.set_facecolor(color)

        # Plot tile linecolors
        if linecolor is not None:
            if isinstance(linecolor, dict):
                # Should be a dict with keys as tile names
                try:
                    linecolor_array = np.array(['black'] * len(tilenames), dtype=object)
                    for k in linecolor.keys():
                        # Thanks to the edge tiles there may be multiple Polygons with the same name
                        i = [i for i, x in enumerate(tilenames) if x == k]
                        linecolor_array[i] = linecolor[k]
                    polys2.set_edgecolor(np.array(linecolor_array))
                except:
                    raise ValueError('Invalid entries in linecolor array')

            elif isinstance(linecolor, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(linecolor) == self.ntiles:
                    raise ValueError('List of linecolors must be same length as grid.ntiles')

                # Sould be a list of color string
                try:
                    polys2.set_edgecolor(np.array(linecolor[new_indexes]))
                except:
                    raise ValueError('Invalid entries in linecolor array')

            else:
                # Might just be a string color name
                polys2.set_edgecolor(linecolor)

        # Plot tile linewidths
        if linewidth is not None:
            if isinstance(linewidth, dict):
                # Should be a dict with keys as tile names
                try:
                    linewidth_array = np.array([1] * len(tilenames))
                    for k in linewidth.keys():
                        # Thanks to the edge tiles there may be multiple Polygons with the same name
                        i = [i for i, x in enumerate(tilenames) if x == k]
                        linewidth_array[i] = linewidth[k]
                    polys2.set_linewidth(np.array(linewidth_array))
                except:
                    raise ValueError('Invalid entries in linewidth array')

            elif isinstance(linewidth, (list, tuple, np.ndarray)):
                # A list-like of floats, should be same length as number of tiles
                if not len(linewidth) == self.ntiles:
                    raise ValueError('List of linewidths must be same length as grid.ntiles')

                # Sould be a list of floats
                try:
                    polys2.set_linewidth(np.array(linewidth[new_indexes]))
                except:
                    raise ValueError('Invalid entries in linewidth array')

            else:
                # Might just be a float
                polys2.set_linewidth(linewidth)

        # Highlight paticular tiles
        if highlight is not None:
            if isinstance(highlight[0], str):
            # Should be a list with keys as tile names
                try:
                    linecolor_array = np.array(['none'] * len(tilenames), dtype=object)
                    linewidth_array = np.array([0] * len(tilenames))
                    for k in highlight:
                        i = [i for i, x in enumerate(tilenames) if x == k]
                        linecolor_array[i] = 'blue'
                        linewidth_array[i] = 3
                    polys3 = copy(polys2)
                    polys3.set_edgecolor(np.array(linecolor_array))
                    polys3.set_linewidth(np.array(linewidth_array))
                    polys3.set_alpha(0.5)
                    polys3.set_zorder(9)
                    axes.add_collection(polys3)
                except:
                    raise ValueError('Invalid entries in highlight list')
            else:
                # Should be a list of lists
                try:
                    colors = ['blue','lime','red','yellow','purple']
                    for j, tilelist in enumerate(highlight):
                        linecolor_array = np.array(['none'] * len(tilenames), dtype=object)
                        linewidth_array = np.array([0] * len(tilenames))
                        for k in tilelist:
                            i = [i for i, x in enumerate(tilenames) if x == k]
                            linecolor_array[i] = colors[j % 5]
                            linewidth_array[i] = 3
                        polys4 = copy(polys2)
                        polys4.set_edgecolor(np.array(linecolor_array))
                        polys4.set_linewidth(np.array(linewidth_array))
                        polys4.set_alpha(0.5)
                        polys4.set_zorder(9 + len(highlight) - j)
                        axes.add_collection(polys4)
                except:
                    raise ValueError('Invalid entries in highlight list')

        # Plot coordinates
        if coordinates:
            axes.scatter(coordinates.ra.value, coordinates.dec.value,
                         transform=transform,
                         s=99, c='blue', marker='*', zorder=9)
            if coordinates.isscalar:
                coordinates = SkyCoord([coordinates])
            for coord in coordinates:
                axes.text(coord.ra.value, coord.dec.value,
                            coord.to_string('hmsdms').replace(' ','\n')+'\n',
                            transform=transform,
                            ha='center', va='bottom',
                            size='x-small', zorder=12,
                            )

        # Set title
        title = 'All sky grid (fov={}x{}, overlap={},{})'.format(self.fov['ra'],
                                                                 self.fov['dec'],
                                                                 self.overlap['ra'],
                                                                 self.overlap['dec'])
        if plot_skymap and hasattr(self, 'skymap'):
            title += '\n' + 'with skymap for trigger {}'.format(self.skymap.objid)
        axes.set_title(title, y=1.05)

        # Save or show
        if filename:
            plt.savefig(filename, dpi=dpi)
        else:
            plt.show()
