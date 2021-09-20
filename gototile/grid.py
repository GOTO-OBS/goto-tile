"""Module containing the SkyGrid class."""

import collections
import os
from copy import copy, deepcopy

from astroplan import AltitudeConstraint, AtNightConstraint, Observer, is_observable

from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u
from astropy.table import QTable

import healpy as hp

import ligo.skymap.plot  # noqa: F401  (for extra projections)

from matplotlib import pyplot as plt
if 'DISPLAY' not in os.environ:
    plt.switch_backend('agg')
from matplotlib.patches import Patch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm

import numpy as np

from .gridtools import create_grid
from .gridtools import get_tile_vertices_astropy as get_tile_vertices
from .gridtools import get_tile_edges_astropy as get_tile_edges
from .gridtools import get_tile_pixels_astropy as get_tile_pixels
from .skymaptools import coord2pix, pix2coord

NAMED_GRIDS = {'GOTO4': [(3.7, 4.9), (0.1, 0.1)],
               'GOTO-4': [(3.7, 4.9), (0.1, 0.1)],
               'GOTO8p': [(7.8, 5.1), (0.1, 0.1)],
               'GOTO-8p': [(7.8, 5.1), (0.1, 0.1)],
               }


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
        if isinstance(fov, (list, tuple)):
            fov = {'ra': fov[0], 'dec': fov[1]}
        for key in ('ra', 'dec'):
            # make sure fov is in degrees
            if not isinstance(fov[key], u.Quantity):
                fov[key] *= u.deg
            else:
                fov[key] = fov[key].to(u.deg)
        self.fov = fov
        self.tile_area = (fov['ra'] * fov['dec']).value

        # Parse overlap
        if overlap is None:
            overlap = {'ra': 0.5, 'dec': 0.5}
        elif isinstance(overlap, (int, float, u.Quantity)):
            overlap = {'ra': overlap, 'dec': overlap}
        elif isinstance(overlap, (list, tuple)):
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

        # Get the tile vertices - 4 points on the corner of each tile
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
        return deepcopy(self)

    @classmethod
    def from_name(cls, name):
        """Initialize a `~gototile.skymap.SkyGrid` object from a name string.

        Parameters
        ----------
        name : str
            the name of the telescope or grid to use.
            either follows the format `allsky-{fov_ra}x{fov_dec}-{overlap_ra}-{overlap_dec}`,
            or one of the predefined names given by `SkyGrid.get_names()`

        Returns
        -------
        `~gototile.skymap.SkyGrid`
            SkyGrid object
        """
        if name.startswith('allsky'):
            try:
                fov = (float(name.split('-')[1].split('x')[0]),
                       float(name.split('-')[1].split('x')[1]))
                overlap = (float(name.split('-')[2]), float(name.split('-')[3]))
            except Exception:
                template = 'allsky-{fov_ra}x{fov_dec}-{overlap_ra}-{overlap_dec}`'
                raise ValueError(f'Grid name "{name}" not recognised, '
                                 'Name format should match ', template)
        else:
            if name in NAMED_GRIDS:
                fov, overlap = NAMED_GRIDS[name]
            else:
                raise ValueError(f'Grid name "{name}" not recognised, '
                                 'check SkyGrid.get_named_grids() for known grids.')

        return cls(fov, overlap)

    @staticmethod
    def get_named_grids():
        """Get a dictionary of pre-defined grid parameters for use with `SkyGrid.from_name()`."""
        return NAMED_GRIDS

    def apply_skymap(self, skymap):
        """Apply a SkyMap to the grid, calculating the contained probability within each tile.

        The tile probabilities are stored in self.probs, and the contour levels in self.contours.

        Parameters
        ----------
        skymap : `gototile.skymap.SkyMap`
            The sky map to map onto this grid.
        """
        # Need to make sure the skymap has order='NESTED' not ring, because it seems there are
        # problems with hp.query_polygon in RING ordering.
        # See https://github.com/GOTO-OBS/goto-tile/issues/65
        # Therefore enforce NESTED when applying a skymap to a grid
        if not skymap.is_nested:
            skymap = skymap.copy()
            skymap.regrade(order='NESTED')

        # Also make sure the skymap is in equatorial coordinates
        if skymap.coordsys != 'C':
            skymap = skymap.copy()
            skymap.rotate('C')

        # Store skymap on the class
        self.skymap = skymap

        # Calculate which pixels are within the tiles
        self.nside = skymap.nside
        self.pixels = self.get_tile_pixels(self.nside)

        # Calculate the tile probabilities and contour levels
        self.probs = self._get_tile_probs()
        self.contours = self._get_tile_contours()

        return self.probs

    def get_tile_pixels(self, nside):
        """Calculate the HEALPix indices within each tile."""
        pixels = get_tile_pixels(self.vertices, nside)
        return np.array(pixels, dtype=object)

    def _get_tile_probs(self):
        """Calculate the contained probabilities within each tile."""
        return np.array([np.sum(self.skymap.skymap[pix]) for pix in self.pixels])

    def _get_tile_contours(self, prob_limit=7):
        """Calculate the minimum contour level of each pixel.

        Unlike for SkyMaps (see `gototile.skymaptools.get_data_contours()`), the calculation for
        tiles is complicated because they can overlap, so the same pixel could be included within
        multiple tiles.

        This method iterates through the tiles by selecting the one with highest probability,
        adding it to the list, then blanking out that portion of the sky and recalculating the
        remaining tile probabilities.

        As this can take quite a while the probability limit (`prob_limit`) is a way to ignore
        tiles with a probability of less than 10^-1**prob_limit (i.e. if the prob_limit is 3 then
        it will only consider tiles with a probability of more than 0.001). The default is 7, which
        will make no difference unless you are considering the 99.9999999% skymap contour level...

        The result is a minimum contour level for every tile, starting at 0 for the highest
        probability tile and increasing from there. To select all tiles within a given contour X you
        can mask for those with a contour level < X.

        You could argue that a faster method would be to only recalculate the probability of the
        tiles that overlap with the high tile. That's true, but the issue is finding which tiles
        overlap. You'll have to loop through every tile and compare its pixels to those of
        the high tile, and do that every time. It's just not worth it. Even if you pre-calculate
        which tiles overlap before you start you don't save any time, because that takes ages
        for any reasonable nside resolution.
        """
        pixel_probs = self.skymap.skymap.copy()
        if prob_limit:
            # Exclude tiles containing a probability of less than 10^-prob_limit
            tile_mask = self.probs > 10**(-1 * prob_limit)
        else:
            tile_mask = np.full(self.ntiles, True)
        tile_pixels = self.pixels[tile_mask]
        tile_probs = np.array([np.sum(pixel_probs[pix]) for pix in tile_pixels])

        sorted_contours = [0]
        sorted_index = []
        for i in range(len(tile_pixels)):
            # Find the tile with the highest probability
            high_tile_prob = max(tile_probs)
            high_tile_index = np.where(tile_probs == high_tile_prob)[0][0]
            # The tile contour value is the probability + cumulative sum of previous tiles
            high_tile_contour = high_tile_prob + sorted_contours[i]
            # Store the tile index and contour value
            sorted_index.append(high_tile_index)
            sorted_contours.append(high_tile_contour)
            # Black out the already-counted pixels
            high_tile_pixels = tile_pixels[high_tile_index]
            pixel_probs[high_tile_pixels] = 0
            # Recalculate the probability within all tiles
            tile_probs = np.array([np.sum(pixel_probs[tile_pix]) for tile_pix in tile_pixels])

        # Start from a contour level of 1, only replace those within the mask
        contours = np.ones(self.ntiles)
        contours[tile_mask] = np.array(sorted_contours)[np.array(sorted_index).argsort()]

        return contours

    def select_tiles(self, contour=0.9, max_tiles=None, min_tile_prob=None):
        """Select tiles based off of the given contour."""
        if not hasattr(self, 'skymap'):
            raise ValueError('SkyGrid does not have a SkyMap applied')

        # Initially mask to cover the entire given contour level
        mask = self.contours < contour

        # Limit to given max tiles, if limit is given
        if max_tiles is not None and sum(mask) > max_tiles:
            # Limit by probability above `max_tiles`th tile
            mask &= self.probs > sorted(self.probs, reverse=True)[max_tiles]

        # Limit to tiles above min prob, if limit is given
        if min_tile_prob is not None:
            mask &= self.probs > min_tile_prob

        # Returned the masked tile table
        table = self.get_table()
        return table[mask]

    def _pixels_from_tilenames(self, tilenames, nside=128):
        """Get the unique pixels contained within the given tile(s)."""
        if hasattr(self, 'pixels'):
            # A skymap has been applied, use those pixels
            tile_pixels = self.pixels
        else:
            # Use the given parameters
            tile_pixels = self.get_tile_pixels(nside)

        if isinstance(tilenames, (list, np.ndarray)):
            # Multiple tiles
            indexes = [self.tilenames.index(tile) for tile in tilenames]
            pixels = []
            for i in indexes:
                pixels += list(tile_pixels[i])
            pixels = list(set(pixels))  # remove duplicates
        else:
            # An individual tile
            index = self.tilenames.index(tilenames)
            pixels = tile_pixels[index]

        return pixels

    def get_tile(self, coord, overlap=False):
        """Find which tile the given coordinates fall within.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            The coordinates to find which tile they are within.

        overlap : bool, optional
            If True then check if the coordinates fall within multiple tiles, and return a list.
            If False (default) just return the closest tile centre.
        """
        # Handle both scalar and vector coordinates
        if coord.isscalar:
            coord = [coord]

        tilenames = []
        if not overlap:
            # Annoyingly SkyCoord.separation requires one or the other to be scalar.
            # So we need this annoying loop to deal with multiple input coordinates.
            for c in coord:
                # Get the separation between the coords and the tile centres
                sep = np.array(c.separation(self.coords))

                # Find which tile has the minimum separation
                index = np.where(sep == (min(sep)))[0][0]

                # Get the tile name and add it to the list
                name = self.tilenames[index]
                tilenames.append(name)
        else:
            if hasattr(self, 'pixels'):
                # A skymap has been applied, use those pixels
                nside = self.nside
                pixels = self.pixels
            else:
                # Use the given parameters
                nside = 128
                pixels = self.get_tile_pixels(nside)

            for c in coord:
                # Get the HEALPix pixel the coords are within
                pixel = coord2pix(nside, c, nest=True)

                # Get the tile indices that contain that pixel and add to list
                names = [self.tilenames[i] for i in range(self.ntiles)
                         if pixel in pixels[i]]
                tilenames.append(names)

        if len(tilenames) == 1:
            return tilenames[0]
        else:
            return tilenames

    def get_visible_tiles(self, locations, time_range=None, alt_limit=30, sun_limit=-15,
                          any_all='any'):
        """Get the tiles that are visible from the given location(s).

        Parameters
        ----------
        locations : `astropy.coordinates.EarthLocation` or list of same
            location(s) to check visibility from

        time_range : 2-tuple of `astropy.time.Time`, optional
            times to check visibility between
            if not given tiles will only be selected based on altitude

        alt_limit : float, optional
            horizon altitude limit to apply
            default is 30 deg

        sun_limit : float, optional
            altitude limit of the Sun to consider night constraints
            default is -15 deg

        any_all : 'any' or 'all', optional
            If 'any' return tiles that are visible from any of the locations.
            If 'all' return tiles that are visible from all of the locations.
            Only valid if len(locations) > 1.
            Default = 'any'

        """
        # Handle multiple locations
        if isinstance(locations, EarthLocation):
            locations = [locations]

        if any_all == 'any':
            mask = np.full(self.ntiles, False)
        elif any_all == 'all':
            mask = np.full(self.ntiles, True)
        else:
            raise ValueError('Invalid value for any_all: "{}"'.format(any_all))

        for location in locations:
            if time_range is None:
                # Find dec limits
                max_dec = location.lat + (90 - alt_limit) * u.deg
                min_dec = location.lat - (90 - alt_limit) * u.deg

                # Find which of the grid tiles are within the limits
                new_mask = np.array(self.coords.dec < max_dec) & np.array(self.coords.dec > min_dec)
            else:
                # Create Astroplan observer
                observer = Observer(location)

                # Create the constraints
                alt_constraint = AltitudeConstraint(min=alt_limit * u.deg)
                night_constraint = AtNightConstraint(max_solar_altitude=sun_limit * u.deg)
                constraints = [alt_constraint, night_constraint]

                # Find which of the grid tiles will be visible
                new_mask = is_observable(constraints, observer, self.coords, time_range=time_range)

            # Combine the mask with the existing one
            if any_all == 'any':
                mask = mask | new_mask
            elif any_all == 'all':
                mask = mask & new_mask

        return list(np.array(self.tilenames)[mask])

    def get_coordinates(self, tilenames):
        """Return the central coordinates of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the coordinates of.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The central coordinates of the given tile(s).
        """
        if isinstance(tilenames, str):
            tilenames = [tilenames]

        # Get indexes
        index = [self.tilenames.index(tile) for tile in tilenames]
        if len(index) == 1:
            index = index[0]

        return self.coords[index]

    def get_vertices(self, tilenames):
        """Return coordinates of the four corners of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the vertices of.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the vertices of the given tile(s).
            Will be an array of shape shape (n, 4), where n = len(tilenames).
        """
        if isinstance(tilenames, str):
            tilenames = [tilenames]

        # Get indexes
        index = [self.tilenames.index(tile) for tile in tilenames]
        if len(index) == 1:
            index = index[0]

        return self.vertices[index]

    def get_edges(self, tilenames, edge_points=5):
        """Return coordinates along the edges of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the edges of.
        steps : int, optional
            The number of points to find along each tile edge.
            If edge_points=0 only the 4 corners will be returned.
            Default=5.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the edge points of the given tile(s).
            Will be an array with shape (n, 4*(edge_points+1)), where n = len(tilenames).

        """
        if isinstance(tilenames, str):
            tilenames = [tilenames]

        # Get indexes
        index = [self.tilenames.index(tile) for tile in tilenames]
        if len(index) == 1:
            index = index[0]

        coords = get_tile_edges(self.coords, self.fov, edge_points)
        return coords[index]

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

    def get_area(self, tilenames, nside=128):
        """Return the sky area contained within the given tile(s) in square degrees.

        If multiple tiles are given, the area only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the area of.
        """
        # Get pixels
        pixels = self._pixels_from_tilenames(tilenames, nside)

        # Each pixel in the skymap has the same area (HEALPix definition)
        # So just multiply that by number of pixels
        if hasattr(self, 'skymap'):
            area = self.skymap.pixel_area * len(pixels)
        else:
            area = hp.nside2pixarea(nside, degrees=True) * len(pixels)

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

    def _get_pixel_count(self, nside=128):
        """Get the count of the number of times each pixel is contained within a grid tile."""
        if hasattr(self, 'pixels'):
            # A skymap has been applied, use those pixels
            tile_pixels = self.pixels
            nside = self.nside
        else:
            # Use the given parameters
            tile_pixels = self.get_tile_pixels(nside)

        # Number of pixels
        npix = hp.nside2npix(nside)

        # For each pixel, create a count of the number of tiles it falls within
        count = np.array([0] * npix)
        for tile_pix in tile_pixels:
            for pix in tile_pix:
                count[pix] += 1

        return count

    def get_stats(self, nside=128):
        """Return a table containing grid statistics."""
        # Get the count
        count = self._get_pixel_count(nside)

        # Create a frequency counter
        counter = collections.Counter(count)

        # Make table
        col_names = ['in_tiles', 'pix', 'freq']
        col_types = ['i', 'i', 'f8']

        in_tiles = [i for i in counter]
        pix = [counter[i] for i in counter]
        freq = [counter[i] / len(count) for i in counter]

        table = QTable([in_tiles, pix, freq],
                       names=col_names, dtype=col_types)
        table['freq'].format = '.4f'
        table = table.group_by('in_tiles')
        return table

    def plot(self, title=None, filename=None, dpi=90, figsize=(8, 6),
             plot_type='mollweide', center=(0, 45), radius=10,
             color=None, linecolor=None, linewidth=None, alpha=0.3,
             discrete_colorbar=False, discrete_stepsize=1,
             colorbar_limits=None, colorbar_orientation='v',
             highlight=None, highlight_color=None, highlight_label=None,
             coordinates=None, tilenames=None, text=None,
             plot_skymap=False, plot_contours=False, plot_stats=False):
        """Plot the grid.

        Parameters
        ----------
        title : str, optional
            title to show above the plot
            if not given a default title will be applied with the name of the grid

        filename : str, optional
            filename to save the plot to
            if not given then the plot will be displayed with plt.show()

        dpi : int, optional
            DPI to display the plot at
            default is 90

        figsize : 2-tuple, optional
            size of the matplotlib figure
            default is (8,6) - matching the GraceDB plots

        plot_type : str, one of 'mollweide', 'globe' or 'zoom', default = 'mollweide'
            type of axes to plot on
            if 'globe' the orthographic plot will be centred on `centre`
            if 'zoom' the plot will be centred on `centre` and have a radius of `radius`

        center : tuple or `astropy.coordinates.SkyCoord`, default (0,45)
            coordinates to center either a globe or zoom plot on
            if given as a tuple units will be considered to be degrees

        radius : float, default 10
            size of the zoomed plot, in degrees
            apparently it can only be a square

        highlight : list of str or list of list or str, optional
            a list of tile names (as in SkyGrid.tilenames) to highlight
            if a 2d list each set will be highlighted with a different color

        highlight_color : str or list of str, optional
            a list of colors to use when highlighting

        highlight_label : str or list of str, optional
            labels to add to the legend when highlighting

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

        discrete_colorbar : bool, optional
            if given a color array or dict, whether to plot using a discrete colorbar or not
            default = False

        discrete_stepsize : int, optional
            if discrete_colorbar is True, the number of steps between labels on the colorbar
            default = 1

        colorbar_limits : 2-tuple, optional
            if given a color array or dict, set the limits of the color bar to (min, max)
            if None the range will be that of the data
            default = None

        colorbar_orientation : 'v' or 'h', optional
            if given a color array or dict, display the colorbar either vertically or horizontally
            default = 'v'

        coordinates : `astropy.coordinates.SkyCoord`, optional
            any coordinates to also plot on the image

        tilenames : list, optional
            should be a list of tile names (as in SkyGrid.tilenames)
            plot the name of the given tiles in their centre

        text : dict, optional
            the keys should be tile names (as in SkyGrid.tilenames)
            the values will be plotted as strings at the tile centres

        plot_skymap : bool, default = False
            color tiles based on their contained probability
            will fail unless a skymap has been applied to the grid using SkyGrid.apply_skymap()

        plot_contours : bool, default = False
            plot 50% and 90% skymap contours
            will fail unless a skymap has been applied to the grid using SkyGrid.apply_skymap()

        plot_stats : bool, default = False
            plot HEALPix pixels colored by the number of tiles they fall within

        """
        fig = plt.figure(figsize=figsize, dpi=dpi)

        if isinstance(center, tuple):
            center = SkyCoord(center[0], center[1], unit='deg')
        if isinstance(center, SkyCoord):
            center = center.to_string('hmsdms')

        if plot_type == 'mollweide':
            axes = plt.axes(projection='astro hours mollweide')
        elif plot_type == 'globe':
            axes = plt.axes(projection='astro globe', center=center)
        elif plot_type == 'zoom':
            axes = plt.axes(projection='astro zoom', center=center, radius=radius * u.deg)
        else:
            raise ValueError('"{}" is not a recognised plot type.')

        axes.grid()
        axes.set_axisbelow(False)
        transform = axes.get_transform('world')

        # We can't just plot the four corners (already saved under self.vertices) because that will
        # plot straight lines between them. That will look bad, because we're on a sphere.
        # Instead we get some intermediate points along the edges, so they look better when plotted.
        # (Admittedly this is only obvious with very large tiles, but it's still good to do).
        edge_points = get_tile_edges(self.coords, self.fov, edge_points=5)

        # Create the tile polygons
        polygons = []
        new_tilenames = []
        for points, tilename in zip(edge_points, self.tilenames):
            # Convert point coordinates to ra,dec
            ra, dec = points.ra.deg, points.dec.deg

            # Check if the tile passes over the RA=0 line:
            overlaps_meridian = any(ra < 90) and any(ra > 270)
            if (not overlaps_meridian) or plot_type != 'mollweide':
                # Need to reverse and transpose to get into the correct format for Polygon
                polygons.append(Polygon(np.array((ra[::-1], dec[::-1])).T))
                new_tilenames.append(tilename)
            else:
                # Annoyingly tiles that pass over the edge of the plot won't be filled
                # This only applies in 'astro hours mollweide' mode
                # The best workaround is to plot two Polygons, one on each side
                ra1 = ra.copy()
                ra1[ra < 180] = 360
                polygons.append(Polygon(np.array((ra1[::-1], dec[::-1])).T))
                ra2 = ra.copy()
                ra2[ra > 180] = 0
                polygons.append(Polygon(np.array((ra2[::-1], dec[::-1])).T))
                # The reason for the tilename array is so we can colour both at once
                # See where we deal with colours below, it hopefully makes sense there
                new_tilenames.extend((tilename, tilename))
        self._polygons = polygons
        self._new_tilenames = new_tilenames

        # Create a map between the original tiles and the polygons
        new_indexes = [np.where(np.array(self.tilenames) == name)[0][0] for name in new_tilenames]
        self._new_indexes = new_indexes

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
        polys2.set_linewidth(0.5)
        polys2.set_alpha(alpha)
        polys2.set_zorder(3)
        axes.add_collection(polys2)

        # Plot tile names
        if tilenames is not None and text is None:
            # Should be a list of tilenames
            for name in tilenames:
                if name not in self.tilenames:
                    continue
                index = np.where(np.array(self.tilenames) == name)[0][0]
                coord = self.coords[index]
                plt.text(coord.ra.deg, coord.dec.deg, name,
                         color='k', weight='bold', fontsize=6,
                         ha='center', va='center', clip_on=True,
                         transform=transform)

        # Plot text
        if text:
            # Should be a dict with keys as tile names
            for name in text:
                if name not in self.tilenames:
                    continue
                index = np.where(np.array(self.tilenames) == name)[0][0]
                coord = self.coords[index]
                plt.text(coord.ra.deg, coord.dec.deg, str(text[name]),
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

        if plot_contours is True:
            if not hasattr(self, 'skymap'):
                raise ValueError('SkyGrid does not have a SkyMap applied')

            # Plot the 50% and 90% skymap contours
            # Taken from SkyMap.plot()
            axes.contour_hpx(self.skymap.contours, nested=self.skymap.is_nested,
                             levels=[0.5 * self.skymap.skymap.sum(),
                                     0.9 * self.skymap.skymap.sum()],
                             colors='black', linewidths=0.5, zorder=99,)

        if plot_stats is True:
            # Colour in areas based on the number of tiles they are within
            polys.set_facecolor('none')

            # Use attributes of the skymap if one has been applied
            if hasattr(self, 'nside'):
                nside = self.nside
            else:
                nside = 128

            # Get the coordinates of each pixel to plot
            npix = hp.nside2npix(nside)
            ipix = range(npix)
            coords = pix2coord(nside, ipix, nest=True)

            # Get count statistics
            count = self._get_pixel_count(nside)

            # Plot HealPix points coloured by tile count
            # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
            # Create the new map
            cmap = plt.cm.get_cmap('gist_rainbow')
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = cmap.from_list('Custom', cmaplist, cmap.N)

            # Normalize
            k = 5
            norm = BoundaryNorm(np.linspace(0, k + 1, k + 2), cmap.N)

            # Plot the points
            points = axes.scatter(coords.ra.deg, coords.dec.deg,
                                  transform=transform,
                                  s=1, c=count,
                                  cmap=cmap, norm=norm,
                                  zorder=0)

            # Add the colorbar
            cb = fig.colorbar(points, ax=axes, fraction=0.02, pad=0.05)
            tick_labels = np.arange(0, k + 1, 1)
            tick_location = tick_labels + 0.5
            tick_labels = [str(label) for label in tick_labels]
            tick_labels[-1] = str(tick_labels[-1] + '+')
            cb.set_ticks(tick_location)
            cb.set_ticklabels(tick_labels)

        # Plot tile colors
        if color is not None:
            if isinstance(color, dict):
                # Should be a dict with keys as tile names
                try:
                    color_array = np.array(['none'] * len(new_tilenames), dtype=object)
                    for k in color.keys():
                        # Thanks to the edge tiles there may be multiple Polygons with the same name
                        i = [i for i, x in enumerate(new_tilenames) if x == k]
                        color_array[i] = color[k]
                    polys.set_facecolor(np.array(color_array))
                except Exception:
                    try:
                        # Create the color array
                        color_array = np.array([np.nan] * len(new_tilenames))
                        for k in color.keys():
                            i = [i for i, x in enumerate(new_tilenames) if x == k]
                            color_array[i] = color[k]
                        color_array = np.array(color_array)

                        # Mask out the NaNs
                        masked_array = np.ma.masked_where(np.isnan(color_array), color_array)
                        polys.set_array(masked_array)

                        if discrete_colorbar:
                            # See above link in plot_stats
                            cmap = copy(plt.cm.jet_r)
                            if colorbar_limits is None:
                                colorbar_limits = (int(np.floor(np.min(masked_array))),
                                                   int(np.ceil(np.max(masked_array))))
                            boundaries = np.linspace(colorbar_limits[0],
                                                     colorbar_limits[1] + 1,
                                                     (colorbar_limits[1] + 1 -
                                                      colorbar_limits[0] + 1))
                            norm = BoundaryNorm(boundaries, cmap.N)
                            polys.set_norm(norm)
                        else:
                            cmap = copy(plt.cm.viridis)

                        # Set the colors of the polygons
                        # Tiles with no data should stay white
                        cmap.set_bad(color='white')
                        polys.set_cmap(cmap)
                        if colorbar_limits is not None:
                            polys.set_clim(colorbar_limits[0], colorbar_limits[1])

                        # Display the color bar
                        if colorbar_orientation.lower()[0] == 'h':
                            cb = fig.colorbar(polys, ax=axes, fraction=0.03, pad=0.05, aspect=50,
                                              orientation='horizontal')
                        else:
                            cb = fig.colorbar(polys, ax=axes, fraction=0.02, pad=0.05)
                        if discrete_colorbar:
                            tick_labels = np.arange(colorbar_limits[0],
                                                    colorbar_limits[1] + 1,
                                                    discrete_stepsize, dtype=int)
                            tick_location = tick_labels + 0.5
                            cb.set_ticks(tick_location)
                            cb.set_ticklabels(tick_labels)

                    except Exception:
                        raise ValueError('Invalid entries in color array')

            elif isinstance(color, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(color) == self.ntiles:
                    raise ValueError('List of colors must be same length as grid.ntiles')

                # Could be a list of weights or a list of colors
                try:
                    polys.set_facecolor(np.array(color[new_indexes]))
                except Exception:
                    try:
                        polys.set_array(np.array(color[new_indexes]))
                        fig.colorbar(polys, ax=axes, fraction=0.02, pad=0.05)
                    except Exception:
                        raise ValueError('Invalid entries in color array')

            else:
                # Might just be a string color name
                polys.set_facecolor(color)

        # Plot tile linecolors
        if linecolor is not None:
            if isinstance(linecolor, dict):
                # Should be a dict with keys as tile names
                try:
                    linecolor_array = np.array(['black'] * len(new_tilenames), dtype=object)
                    for k in linecolor.keys():
                        # Thanks to the edge tiles there may be multiple Polygons with the same name
                        i = [i for i, x in enumerate(new_tilenames) if x == k]
                        linecolor_array[i] = linecolor[k]
                    polys2.set_edgecolor(np.array(linecolor_array))
                except Exception:
                    raise ValueError('Invalid entries in linecolor array')

            elif isinstance(linecolor, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(linecolor) == self.ntiles:
                    raise ValueError('List of linecolors must be same length as grid.ntiles')

                # Should be a list of color string
                try:
                    polys2.set_edgecolor(np.array(linecolor[new_indexes]))
                except Exception:
                    raise ValueError('Invalid entries in linecolor array')

            else:
                # Might just be a string color name
                polys2.set_edgecolor(linecolor)

        # Plot tile linewidths
        if linewidth is not None:
            if isinstance(linewidth, dict):
                # Should be a dict with keys as tile names
                try:
                    linewidth_array = np.array([0.5] * len(new_tilenames))
                    for k in linewidth.keys():
                        # Thanks to the edge tiles there may be multiple Polygons with the same name
                        i = [i for i, x in enumerate(new_tilenames) if x == k]
                        linewidth_array[i] = linewidth[k]
                    polys2.set_linewidth(np.array(linewidth_array))
                except Exception:
                    raise ValueError('Invalid entries in linewidth array')

            elif isinstance(linewidth, (list, tuple, np.ndarray)):
                # A list-like of floats, should be same length as number of tiles
                if not len(linewidth) == self.ntiles:
                    raise ValueError('List of linewidths must be same length as grid.ntiles')

                # Should be a list of floats
                try:
                    polys2.set_linewidth(np.array(linewidth[new_indexes]))
                except Exception:
                    raise ValueError('Invalid entries in linewidth array')

            else:
                # Might just be a float
                polys2.set_linewidth(linewidth)

        # Highlight paticular tiles
        if highlight is not None:
            if isinstance(highlight, str):
                # Might just be one tile
                highlight = [highlight]

            legend_patches = []
            if isinstance(highlight[0], str):
                # Should be a list with keys as tile names
                try:
                    if highlight_color is None:
                        highlight_color = 'blue'
                    linecolor_array = np.array(['none'] * len(new_tilenames), dtype=object)
                    linewidth_array = np.array([0] * len(new_tilenames))
                    for k in highlight:
                        i = [i for i, x in enumerate(new_tilenames) if x == k]
                        linecolor_array[i] = highlight_color
                        linewidth_array[i] = 1.5
                    # Create polygons
                    polys3 = copy(polys2)
                    polys3.set_edgecolor(np.array(linecolor_array))
                    polys3.set_linewidth(np.array(linewidth_array))
                    polys3.set_alpha(0.5)
                    polys3.set_zorder(9)
                    axes.add_collection(polys3)
                    # Add to legend
                    if highlight_label is not None:
                        label = highlight_label + ' ({} tiles)'.format(len(highlight))
                        patch = Patch(facecolor='none',
                                      edgecolor=highlight_color,
                                      linewidth=1.5,
                                      label=label,
                                      )
                        legend_patches.append(patch)
                except Exception:
                    raise ValueError('Invalid entries in highlight list')
            else:
                # Should be a list of lists
                try:
                    if highlight_color is None:
                        colors = ['blue', 'red', 'lime', 'purple', 'yellow']
                    elif isinstance(highlight_color, str):
                        colors = [highlight_color]
                    else:
                        colors = highlight_color
                    for j, tilelist in enumerate(highlight):
                        linecolor_array = np.array(['none'] * len(new_tilenames), dtype=object)
                        linewidth_array = np.array([0] * len(new_tilenames))
                        for k in tilelist:
                            i = [i for i, x in enumerate(new_tilenames) if x == k]
                            linecolor = colors[j % len(colors)]
                            linecolor_array[i] = linecolor
                            linewidth_array[i] = 1.5
                        # Create polygons
                        polys4 = copy(polys2)
                        polys4.set_edgecolor(np.array(linecolor_array))
                        polys4.set_linewidth(np.array(linewidth_array))
                        polys4.set_alpha(0.5)
                        polys4.set_zorder(9 + len(highlight) - j)
                        axes.add_collection(polys4)
                        # Add to legend
                        if highlight_label is not None:
                            label = highlight_label[j] + ' ({} tiles)'.format(len(tilelist))
                            patch = Patch(facecolor='none',
                                          edgecolor=linecolor,
                                          linewidth=1.5,
                                          label=label,
                                          )
                            legend_patches.append(patch)
                except Exception:
                    raise ValueError('Invalid entries in highlight list')

            # Display legend
            if len(legend_patches) > 0:
                plt.legend(handles=legend_patches,
                           loc='center',
                           bbox_to_anchor=(0.5, -0.1),
                           ncol=3,
                           ).set_zorder(999)

        # Plot coordinates
        if coordinates:
            axes.scatter(coordinates.ra.value, coordinates.dec.value,
                         transform=transform,
                         s=99, c='blue', marker='*',
                         zorder=9)
            if coordinates.isscalar:
                coordinates = SkyCoord([coordinates])
            for coord in coordinates:
                axes.text(coord.ra.value, coord.dec.value,
                          coord.to_string('hmsdms').replace(' ', '\n') + '\n',
                          transform=transform,
                          ha='center', va='bottom',
                          size='x-small', zorder=12,
                          )

        # Set title
        if title is None:
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
            plt.close(fig)
        else:
            plt.show()
