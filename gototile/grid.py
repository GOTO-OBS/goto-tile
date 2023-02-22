"""Module containing the SkyGrid class."""

import os
from collections import Counter
from copy import copy, deepcopy

from astroplan import AltitudeConstraint, AtNightConstraint, Observer, is_observable

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import QTable

import ligo.skymap.plot  # noqa: F401  (for extra projections)

from matplotlib import pyplot as plt
if 'DISPLAY' not in os.environ:
    plt.switch_backend('agg')
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path

import numpy as np

from .gridtools import create_grid
from .gridtools import get_tile_edges_astropy as get_tile_edges
from .gridtools import get_tile_vertices_astropy as get_tile_vertices
from .skymap import SkyMap
from .skymaptools import coord2pix, pix2coord

NAMED_GRIDS = {'GOTO4': [(3.7, 4.9), (0.1, 0.1), 'minverlap'],
               'GOTO-4': [(3.7, 4.9), (0.1, 0.1), 'minverlap'],
               'GOTO8p': [(7.8, 5.1), (0.1, 0.1), 'minverlap'],
               'GOTO-8p': [(7.8, 5.1), (0.1, 0.1), 'minverlap'],
               'GOTO8': [(8.0, 5.5), (0.2 / 8.0, 0.2 / 5.5), 'enhanced1011'],
               'GOTO-8': [(8.0, 5.5), (0.2 / 8.0, 0.2 / 5.5), 'enhanced1011'],
               'GOTO': [(8.0, 5.5), (0.2 / 8.0, 0.2 / 5.5), 'enhanced1011'],
               }


class SkyGrid:
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
        fill_len = len(str(max(self.tilenums)))
        self.tilenames = ['T' + str(num).zfill(fill_len) for num in self.tilenums]

        # Properties waiting for a skymap to be applied
        self.skymap = None
        self.pixels = None
        self.probs = None
        self.contours = None

    def __eq__(self, other):
        try:
            return (self.fov == other.fov and
                    self.overlap == other.overlap and
                    self.kind == other.kind)
        except AttributeError:
            return False

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
        """Initialize a `~gototile.grid.SkyGrid` object from a name string.

        Parameters
        ----------
        name : str
            the name of the telescope or grid to use.
            either follows the format `allsky-{fov_ra}x{fov_dec}-{overlap_ra}-{overlap_dec}`,
            or one of the predefined names given by `SkyGrid.get_names()`

        Returns
        -------
        `~gototile.grid.SkyGrid`
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
                fov, overlap, kind = NAMED_GRIDS[name]
            else:
                raise ValueError(f'Grid name "{name}" not recognised, '
                                 'check SkyGrid.get_named_grids() for known grids.')

        return cls(fov, overlap, kind)

    @staticmethod
    def get_named_grids():
        """Get a dictionary of pre-defined grid parameters for use with `SkyGrid.from_name()`."""
        return NAMED_GRIDS

    def apply_skymap(self, skymap, flatten=False):
        """Apply a SkyMap to the grid, calculating the contained probability within each tile.

        The tile probabilities are stored in self.probs, and the contour levels in self.contours.

        Parameters
        ----------
        skymap : `gototile.skymap.SkyMap`
            The sky map to map onto this grid.
        flatten : bool, default=False
            If True, and a multi-order skymap is given, then flatten it before applying.

        """
        # Store a copy of the skymap on the class
        self.skymap = skymap.copy()
        self.nside = self.skymap.nside

        # Flatten multi-order skymaps if requested
        if self.skymap.is_moc and flatten:
            self.skymap.regrade(self.skymap.nside, order='NESTED')

        # Ensure the skymap is in equatorial coordinates
        if self.skymap.coordsys != 'C':
            self.skymap.rotate('C')

        # Ensure the skymap is in units of probability, not probability density
        if self.skymap.density:
            self.skymap.density = False

        # Calculate which skymap pixels are contained within each tile,
        # then find the tile probabilities and contour levels
        self.pixels = self._get_tile_pixels()
        self.probs = self._get_tile_probs()
        self.contours = self._get_tile_contours()

        return self.probs

    def _get_tile_pixels(self, skymap=None):
        """Calculate the skymap pixel indices within each tile."""
        if skymap is None:
            skymap = self.skymap

        # Need to provide tile vertices in cartesian coordinates
        vertices = self.vertices.cartesian.get_xyz(xyz_axis=2).value

        # Use the mhealpy `query_polygon` method on the skymap.
        # Unlike healpy's `query_polygon` this can also deal with multi-order skymaps.
        # HOWEVER it's really, really slow. Which is why we have to flatten skymaps when applying.
        # `inclusive=True` will include pixels that overlap in area even if the centres aren't
        # inside the region (`fact` tells how deep to look, at nside=self.nside*fact)
        ipix = [skymap.query_polygon(v, inclusive=True, fact=32) for v in vertices]

        # Note the number of pixels per tile will vary, so the returned array is an array of lists
        return np.array(ipix, dtype=object)

    def _get_tile_probs(self):
        """Calculate the contained probabilities within each tile."""
        probs = [np.sum(self.skymap.data[ipix]) for ipix in self.pixels]
        return np.array(probs)

    def _get_tile_contours(self, prob_limit=5):
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
        pixel_probs = self.skymap.data.copy()
        if prob_limit:
            # Exclude tiles containing a probability of less than 10^-prob_limit
            tile_mask = self.probs > 10**(-1 * prob_limit)
        else:
            tile_mask = np.full(self.ntiles, True)
        tile_pixels = self.pixels[tile_mask]
        tile_probs = np.array([np.sum(pixel_probs[ipix]) for ipix in tile_pixels])

        sorted_contours = [0]
        sorted_index = []
        for i in range(len(tile_pixels)):
            # Find the tile with the highest probability
            high_tile_prob = max(tile_probs)
            if high_tile_prob == 0:
                # We've already blacked out all the pixels, there's no probability left!
                # This can happen with really low-resolution skymaps, where the grid tiles are
                # of the order or larger than the pixels.
                # Just add all the remaining pixels with a contour value of 1.
                unassigned_pixels = [i for i in range(len(tile_pixels)) if i not in sorted_index]
                sorted_index += unassigned_pixels
                sorted_contours += [1] * len(unassigned_pixels)
                break
            # Find the high tile index
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
            tile_probs = np.array([np.sum(pixel_probs[ipix]) for ipix in tile_pixels])

        # Start from a contour level of 1, only replace those within the mask
        contours = np.ones(self.ntiles)
        contours[tile_mask] = np.array(sorted_contours)[np.array(sorted_index).argsort()]

        return contours

    def select_tiles(self, contour=0.9, max_tiles=None, min_tile_prob=None):
        """Select tiles based off of the given contour."""
        if self.probs is None or self.contours is None:
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

    def _get_test_map(self, nside=None):
        """Create a basic empty skymap, useful for statistical calculations.

        We could do this in __init__, but why not save time and only do if if we need it?
        """
        if hasattr(self, '_base_nside') and self._base_nside == nside:
            return
        if nside is None:
            nside = 128
        self._base_nside = nside
        self._base_skymap = SkyMap(np.zeros(12 * self._base_nside ** 2), order='NESTED')
        self._base_pixels = self._get_tile_pixels(self._base_skymap)

    def get_tile(self, coord, overlap=False):
        """Find which tile(s) the given coordinates fall within.

        Parameters
        ----------
        coord : `astropy.coordinates.SkyCoord`
            The coordinates to find which tile(s) they are within.

        overlap : bool, optional
            If True then check if the coordinates fall within multiple tiles, and return a list.
            If False (default) just return the tile centred closest to the given coordinates.

        Returns
        -------
        tilename : str or list of str
            The name(s) of the tile(s) the coordinates are within

        """
        # Handle both scalar and vector coordinates
        scalar = False
        if coord.isscalar:
            scalar = True
            coord = [coord]

        tilenames = []
        if not overlap:
            # Annoyingly SkyCoord.separation requires one or the other to be scalar.
            # So we need this annoying loop to deal with multiple input coordinates.
            for c in coord:
                # Get the separation between the coords and all tile centres
                sep = np.array(c.separation(self.coords))

                # Find which tile has the minimum separation (i.e. the closest)
                index = np.where(sep == (min(sep)))[0][0]

                # Get the tile name and add it to the list
                name = self.tilenames[index]
                tilenames.append(name)
        else:
            # Use the base skymap to find which pixels the coordinates are within
            self._get_test_map()
            for c in coord:
                # Get the HEALPix pixel the coords are within
                pixel = coord2pix(self._base_nside, c, nest=True)

                # Get the tile indices that contain that pixel and add to list
                names = [self.tilenames[i] for i in range(self.ntiles)
                         if pixel in self._base_pixels[i]]
                tilenames.append(names)

        if scalar:
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

    def _get_tilename_indices(self, tilenames):
        """Return the indices of the given tile(s)."""
        if isinstance(tilenames, str):
            return self.tilenames.index(tilenames)
        else:
            return [self.tilenames.index(tile) for tile in tilenames]

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
        indices = self._get_tilename_indices(tilenames)
        return self.coords[indices]

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
        indices = self._get_tilename_indices(tilenames)
        return self.vertices[indices]

    def get_edges(self, tilenames, edge_points=5):
        """Return coordinates along the edges of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the edges of.
        edge_points : int, optional
            The number of points to find along each tile edge.
            If edge_points=0 only the 4 corners will be returned.
            Default=5.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the edge points of the given tile(s).
            Will be an array with shape (n, 4*(edge_points+1)), where n = len(tilenames).

        """
        indices = self._get_tilename_indices(tilenames)
        coords = get_tile_edges(self.coords, self.fov, edge_points)
        return coords[indices]

    def get_pixels(self, tilenames):
        """Get the skymap pixels contained within the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the pixels of.

        Returns
        -------
        ipix : list of int
            Pixel indices covering the area of the given tile(s).

        """
        if self.pixels is None:
            raise ValueError('SkyGrid does not have a SkyMap applied')

        indices = self._get_tilename_indices(tilenames)
        if isinstance(indices, int):
            pix = self.pixels[indices]
        else:
            pix = [ipix for tile_pix in self.pixels[indices] for ipix in tile_pix]
        return sorted(set(pix))

    def get_probability(self, tilenames):
        """Return the contained probability within the given tile(s).

        If multiple tiles are given, the probability only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the probability within.

        Returns
        -------
        probability : int
            The total skymap value within the area covered by the given tile(s).

        """
        if self.probs is None:
            raise ValueError('SkyGrid does not have a SkyMap applied')

        pixels = self.get_pixels(tilenames)
        return self.skymap.data[pixels].sum()

    def get_area(self, tilenames):
        """Return the sky area contained within the given tile(s) in square degrees.

        If multiple tiles are given, the area only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the area of.

        Returns
        -------
        area : int
            The total sky area covered by the given tile(s), in square degrees.

        """
        self._get_test_map()

        indices = self._get_tilename_indices(tilenames)
        if isinstance(indices, int):
            pix = sorted(set(self._base_pixels[indices]))
        else:
            pix = [ipix for tile_pix in self._base_pixels[indices] for ipix in tile_pix]
        pix = sorted(set(pix))
        return self._base_skymap.get_pixel_area(pix)

    def get_areas(self, tilenames):
        """Return the areas contained within each of the given tile(s) in square degrees.

        Note although every tile should have the same area (equal to fov_ra * fov_dec) there will
        be slight differences from the resolution of the HEALPix grid.

        For most cases you'll want to use the `SkyGrid.get_tile_area()` function instead, which
        gives the COMBINED area of the given tiles (i.e. only including overlapping areas once).

        Parameters
        ----------
        tilenames : str or list of str
            The name(s) of the tile(s) to find the area of.

        Returns
        -------
        area : int or list of int
            The areas covered by the given tile(s), in square degrees.

        """
        if isinstance(tilenames, str):
            return self.get_area(tilenames)
        else:
            return [self.get_area(tile) for tile in tilenames]

    def get_table(self):
        """Return an astropy QTable containing information on the defined tiles.

        If a sky map has been applied to the grid the table will include a column with
            the contained probability within each tile.
        """
        col_names = ['tilename', 'ra', 'dec', 'prob']
        col_data = [self.tilenames, self.coords.ra, self.coords.dec,
                    self.probs if self.probs is not None else np.zeros(self.ntiles)]
        return QTable(col_data, names=col_names)

    def _get_pixel_count(self, nside=128):
        """For each pixel in the base skymap, count of the number of tiles it falls within."""
        # Create test skymap, if it hasn't already been generated
        self._get_test_map(nside)

        count = np.zeros(12 * self._base_nside ** 2)
        for ipix in self._base_pixels:
            count[ipix] += 1
        return count

    def get_stats(self, nside=128):
        """Return a table containing grid statistics."""
        count = self._get_pixel_count(nside)
        counter = Counter(count)
        in_tiles = [int(i) for i in counter]
        npix = [counter[i] for i in counter]
        freq = [counter[i] / len(count) for i in counter]

        col_names = ['in_tiles', 'npix', 'freq']
        col_data = [in_tiles, npix, freq]

        table = QTable(col_data, names=col_names)
        table['freq'].format = '.2%'
        table = table.group_by('in_tiles')
        return table

    def _get_tile_path(self, edge_coords, meridian_split=False):
        """Create a Matplotlib Path for the given tile."""
        ra = edge_coords.ra.deg
        dec = edge_coords.dec.deg

        # Check if the tile passes over the RA=0 line:
        overlaps_meridian = any(ra < 90) and any(ra > 270)
        if meridian_split and overlaps_meridian:
            if any(np.logical_and(ra > 90, ra < 270)):
                # This tile goes over the poles
                # To get it to fill we need to add extra points at the pole itself
                # First sort by RA
                ra, dec = zip(*sorted(zip(ra, dec), key=lambda radec: radec[0]))

                # Now add extra points
                pole = 90 if np.all(np.array(dec) > 0) else -90
                ra = np.array([0] + list(ra) + [360, 360])
                dec = np.array([pole] + list(dec) + [dec[0], pole])

                # Create the closed path
                path = Path(np.array((ra, dec)).T, closed=True)

            else:
                # Tiles that pass over the edges of the plot (at RA=0) won't fill properly,
                # they need to be split into two sections on either side.
                # First create masks, with a little leeway on each side
                mask_l = np.logical_or(ra <= 181, ra > 359)
                mask_r = np.logical_or(ra < 1, ra >= 179)

                # Now mask the arrays
                ra_l = ra[mask_l]
                dec_l = dec[mask_l]
                ra_r = ra[mask_r]
                dec_r = dec[mask_r]

                # Set the points on the meridian to the correct values
                ra_l[(ra_l < 1) | (ra_l > 359)] = 0
                ra_r[(ra_r < 1) | (ra_r > 359)] = 360

                # Add the first point on again to close
                ra_l = list(ra_l) + [ra_l[0]]
                dec_l = list(dec_l) + [dec_l[0]]
                ra_r = list(ra_r) + [ra_r[0]]
                dec_r = list(dec_r) + [dec_r[0]]

                # Create the paths, then combine them
                path_l = Path(np.array((ra_l, dec_l)).T, closed=True)
                path_r = Path(np.array((ra_r, dec_r)).T, closed=True)
                path = Path.make_compound_path(path_l, path_r)

        else:
            # Just make a normal closed path
            path = Path(np.array((ra, dec)).T, closed=True)

        return path

    def _get_tile_paths(self, meridian_split=False):
        """Create and cache Matplotlib Patches to use when plotting tiles."""
        # Used cached versions to save time when repeatedly plotting
        if not meridian_split and hasattr(self, '_paths'):
            return self._paths
        elif meridian_split and hasattr(self, '_paths_split'):
            return self._paths_split

        # We can't just plot the four corners (already saved under self.vertices) because that will
        # plot straight lines between them. That will look bad, because we're on a sphere.
        # Instead we get some intermediate points along the edges, so they look better when plotted.
        # (Admittedly this is only obvious with very large tiles, but it's still good to do).
        if not hasattr(self, 'edges'):
            self.edges = get_tile_edges(self.coords, self.fov, edge_points=4)

        # Get list of matplotlib paths for the tile areas
        paths = [self._get_tile_path(edge_coords, meridian_split) for edge_coords in self.edges]

        if not meridian_split:
            self._paths = paths
        else:
            self._paths_split = paths

        return paths

    def plot_tile(self, axes, tilename, *args, **kwargs):
        """Plot a Patch for the tile onto the given axes."""
        # Add default arguments
        if 'fc' not in kwargs and 'facecolor' not in kwargs:
            kwargs['fc'] = 'tab:blue'
        if 'ec' not in kwargs and 'edgecolor' not in kwargs:
            kwargs['ec'] = 'black'
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 0.5

        # Get tile paths (will be cached after the first use)
        meridian_split = 'Mollweide' in axes.__class__.__name__
        paths = self._get_tile_paths(meridian_split=meridian_split)

        # Create a Patch, applying any arguments
        index = self.tilenames.index(tilename)
        path = paths[index]
        patch = PathPatch(path,
                          transform=axes.get_transform('world'),
                          *args, **kwargs)
        axes.add_patch(patch)

        return patch

    def plot_tiles(self, axes, tilenames=None, *args, **kwargs):
        """Plot a PatchCollection for the grid tiles onto the given axes."""
        # Add default arguments
        if 'fc' not in kwargs and 'facecolor' not in kwargs:
            kwargs['fc'] = 'tab:blue'
        if 'ec' not in kwargs and 'edgecolor' not in kwargs:
            kwargs['ec'] = 'black'
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 0.5

        # Get tile paths (will be cached after the first use)
        meridian_split = 'Mollweide' in axes.__class__.__name__
        paths = self._get_tile_paths(meridian_split=meridian_split)

        # Create a Patch Collection, applying any arguments
        if tilenames is not None:
            indexes = [self.tilenames.index(tilename) for tilename in tilenames]
            paths = np.array(paths)[indexes]
        patches = [PathPatch(path) for path in paths]
        collection = PatchCollection(patches,
                                     transform=axes.get_transform('world'),
                                     *args, **kwargs)
        axes.add_collection(collection)

        return collection

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

        # Plot the tiles
        tile_patches = self.plot_tiles(axes,
                                       fc='blue',
                                       ec='none',
                                       lw=0,
                                       alpha=alpha,
                                       zorder=2,
                                       )

        # Also plot on the lines over the top
        edge_patches = self.plot_tiles(axes,
                                       fc='none',
                                       ec='black',
                                       lw=0.5,
                                       alpha=alpha,
                                       zorder=3,
                                       )

        # Plot text
        if tilenames is not None and text is None:
            # Use the tilenames as the text
            text = {tilename: tilename for tilename in tilenames}
        if text:
            # Should be a dict with keys as tile names
            for name in text:
                try:
                    index = self.tilenames.index(name)
                except ValueError:
                    continue
                coord = self.coords[index]
                plt.text(coord.ra.deg, coord.dec.deg, str(text[name]),
                         color='k', weight='bold', fontsize=6,
                         ha='center', va='center', clip_on=True,
                         transform=transform)

        # Plot skymap probabilities
        if plot_skymap is True:
            if self.skymap is None:
                raise ValueError('SkyGrid does not have a SkyMap applied')

            # Set the probability array to the color array
            skymap_patches = self.plot_tiles(axes,
                                             array=np.array(self.probs),
                                             ec='none',
                                             lw=0,
                                             alpha=0.5,
                                             zorder=1,
                                             )

            # Use the LIGO colormap
            skymap_patches.set_cmap('cylon')
            fig.colorbar(skymap_patches, ax=axes, fraction=0.02, pad=0.05)

            # Plot underneath the other patches, and make them transparent for now
            # (unless overwritten later, e.g. for visibility)
            tile_patches.set_facecolor('none')

        if plot_contours is True:
            if self.skymap is None:
                raise ValueError('SkyGrid does not have a SkyMap applied')

            # Plot the 50% and 90% skymap contours
            # Taken from SkyMap.plot()
            contour_levels = [0.5, 0.9]
            self.skymap.plot_contours(axes, contour_levels=contour_levels,
                                      colors='black', linewidths=0.5, zorder=99)

        if plot_stats is True:
            # Colour in areas based on the number of tiles they are within
            tile_patches.set_facecolor('none')

            # Get count statistics and the coordinates of each pixel to plot
            count = self._get_pixel_count()
            coords = pix2coord(self._base_nside, range(len(count)), nest=True)

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
                    color_array = np.array(['none'] * self.ntiles, dtype=object)
                    for name in color.keys():
                        index = self.tilenames.index(name)
                        color_array[index] = color[name]
                    tile_patches.set_facecolor(np.array(color_array))
                except Exception:
                    try:
                        # Create the color array
                        color_array = np.array([np.nan] * self.ntiles)
                        for name in color.keys():
                            index = self.tilenames.index(name)
                            color_array[index] = color[name]
                        color_array = np.array(color_array)

                        # Mask out the NaNs
                        masked_array = np.ma.masked_where(np.isnan(color_array), color_array)
                        tile_patches.set_array(masked_array)

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
                            tile_patches.set_norm(norm)
                        else:
                            cmap = copy(plt.cm.viridis)

                        # Set the colors of the polygons
                        # Tiles with no data should stay white
                        cmap.set_bad(color='white')
                        tile_patches.set_cmap(cmap)
                        if colorbar_limits is not None:
                            tile_patches.set_clim(colorbar_limits[0], colorbar_limits[1])

                        # Display the color bar
                        if colorbar_orientation.lower()[0] == 'h':
                            cb = fig.colorbar(tile_patches, ax=axes,
                                              fraction=0.03, pad=0.05, aspect=50,
                                              orientation='horizontal')
                        else:
                            cb = fig.colorbar(tile_patches, ax=axes, fraction=0.02, pad=0.05)
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
                    tile_patches.set_facecolor(np.array(color))
                except Exception:
                    try:
                        tile_patches.set_array(np.array(color))
                        fig.colorbar(tile_patches, ax=axes, fraction=0.02, pad=0.05)
                    except Exception:
                        raise ValueError('Invalid entries in color array')

            else:
                # Might just be a string color name
                tile_patches.set_facecolor(color)

        # Plot tile linecolors
        if linecolor is not None:
            if isinstance(linecolor, dict):
                # Should be a dict with keys as tile names
                try:
                    linecolor_array = np.array(['black'] * self.ntiles, dtype=object)
                    for name in linecolor.keys():
                        index = self.tilenames.index(name)
                        linecolor_array[index] = linecolor[name]
                    edge_patches.set_edgecolor(np.array(linecolor_array))
                except Exception:
                    raise ValueError('Invalid entries in linecolor array')

            elif isinstance(linecolor, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(linecolor) == self.ntiles:
                    raise ValueError('List of linecolors must be same length as grid.ntiles')

                # Should be a list of color string
                try:
                    edge_patches.set_edgecolor(np.array(linecolor))
                except Exception:
                    raise ValueError('Invalid entries in linecolor array')

            else:
                # Might just be a string color name
                edge_patches.set_edgecolor(linecolor)

        # Plot tile linewidths
        if linewidth is not None:
            if isinstance(linewidth, dict):
                # Should be a dict with keys as tile names
                try:
                    linewidth_array = np.array([0.5] * self.ntiles)
                    for name in linewidth.keys():
                        index = self.tilenames.index(name)
                        linewidth_array[index] = linewidth[name]
                    edge_patches.set_linewidth(np.array(linewidth_array))
                except Exception:
                    raise ValueError('Invalid entries in linewidth array')

            elif isinstance(linewidth, (list, tuple, np.ndarray)):
                # A list-like of floats, should be same length as number of tiles
                if not len(linewidth) == self.ntiles:
                    raise ValueError('List of linewidths must be same length as grid.ntiles')

                # Should be a list of floats
                try:
                    edge_patches.set_linewidth(np.array(linewidth))
                except Exception:
                    raise ValueError('Invalid entries in linewidth array')

            else:
                # Might just be a float
                edge_patches.set_linewidth(linewidth)

        # Highlight particular tiles
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
                    linecolor_array = np.array(['none'] * self.ntiles, dtype=object)
                    linewidth_array = np.array([0] * self.ntiles)
                    for name in highlight:
                        index = self.tilenames.index(name)
                        linecolor_array[index] = highlight_color
                        linewidth_array[index] = 1.5
                    # Add patches over normal lines
                    self.plot_tiles(axes,
                                    fc='none',
                                    ec=np.array(linecolor_array),
                                    lw=np.array(linewidth_array),
                                    alpha=0.5,
                                    zorder=9,
                                    )
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
                        linecolor_array = np.array(['none'] * self.ntiles, dtype=object)
                        linewidth_array = np.array([0] * self.ntiles)
                        for name in tilelist:
                            index = self.tilenames.index(name)
                            linecolor = colors[j % len(colors)]
                            linecolor_array[index] = linecolor
                            linewidth_array[index] = 1.5
                        # Add patches over normal lines
                        self.plot_tiles(axes,
                                        fc='none',
                                        ec=np.array(linecolor_array),
                                        lw=np.array(linewidth_array),
                                        alpha=0.5,
                                        zorder=9 + len(highlight) - j,
                                        )
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
            if plot_skymap and self.skymap is not None:
                title += '\n' + 'with skymap'
        axes.set_title(title, y=1.05)

        # Save or show
        if filename:
            plt.savefig(filename, dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
