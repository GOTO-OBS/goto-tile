"""Module containing the SkyGrid class."""

import itertools
import math
import re
from collections import Counter
from copy import copy, deepcopy

import ligo.skymap.plot  # noqa: F401  (for extra projections)
import numpy as np
from astroplan import AltitudeConstraint, AtNightConstraint, Observer, is_observable
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import QTable
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch, PathPatch

from .geometry import coords_to_path, interpolate_points, onsky_offset
from .skymap import SkyMap
from .skymaptools import coord2pix, pix2coord

NAMED_GRIDS = {
    'GOTO4': {
        'fov': (3.7, 4.9),
        'overlap': (0.1, 0.1),
        'kind': 'minverlap',
        'array': {'fov': (2.1, 2.8), 'shape': (2, 2), 'overlap': 0.3},
    },
    'GOTO8p': {
        'fov': (7.8, 5.1),
        'overlap': (0.1, 0.1),
        'kind': 'minverlap',
        'array': {'fov': (2.1, 2.8), 'shape': (4, 2), 'overlap': 0.3},
    },
    'GOTO8': {
        'fov': (8.0, 5.5),
        'overlap': (0.2 / 8.0, 0.2 / 5.5),  # Note absolute 0.2 deg overlap, not relative fraction
        'kind': 'enhanced1011',  # integer_fit, polar_edge & corner_align but don't force_equator
        'array': {'fov': (2.21, 2.95), 'shape': (4, 2), 'overlap': 0.2},
    },
}
# Aliases
NAMED_GRIDS['GOTO-4'] = NAMED_GRIDS['GOTO4']
NAMED_GRIDS['GOTO-8p'] = NAMED_GRIDS['GOTO8p']
NAMED_GRIDS['GOTO-8'] = NAMED_GRIDS['GOTO8']
NAMED_GRIDS['GOTO'] = NAMED_GRIDS['GOTO8']


def create_grid(fov, overlap, kind='minverlap'):
    """Create grid coordinates.

    Calculate strips along RA and stacked in declination to cover the full sky.

    The step size in Right Ascension is adjusted with the declination,
    by a factor of 1/cos(declination).

    Parameters
    ----------
    fov : dict of int or float or `astropy.units.Quantity`
        The field of view of the tiles in the RA and Dec directions.
        It should contains the keys 'ra' and 'dec'.
        If not given units the values are assumed to be in degrees.

    overlap : dict of int or float
        The overlap amount between the tiles in the RA and Dec directions.
        It should contains the keys 'ra' and 'dec'.

    kind : str
        The tiling method to use. Options are:
        - 'enhanced':
                An improved algorithm with additional options.
                This is not yet the default, but will be eventually.
        - 'minverlap' (default):
                Uses the overlap as a minimum parameter to fit an integer
                number of evenly-spaced tiles into each row.
        - 'cosine':
                Intermediate algorithm which adjusts RA spacing based on dec.
        - 'cosine_symmetric':
                An alternate version of 'cosine' which rotates each dec stripe
                to be symmetric about the meridian.
        - 'product':
                Old, legacy algorithm.
                This method creates lots of overlap between tiles at high decs,
                which makes it impractical for survey purposes.

    """
    fov = fov.copy()
    overlap = overlap.copy()
    for key in ('ra', 'dec'):
        # Get value of foc
        if isinstance(fov[key], u.Quantity):
            fov[key] = fov[key].to('deg').value

        # Limit overlap to between 0 and 0.9
        overlap[key] = min(max(overlap[key], 0), 0.9)

    if kind == 'cosine':
        return create_grid_cosine(fov, overlap)
    if kind == 'cosine_symmetric':
        return create_grid_cosine_symmetric(fov, overlap)
    if kind == 'product':
        return create_grid_product(fov, overlap)
    if kind == 'minverlap':
        return create_grid_minverlap(fov, overlap)
    if kind == 'enhanced':
        return create_grid_enhanced(fov, overlap)
    if kind.startswith('enhanced'):
        params = [i == '1' for i in kind.replace('enhanced', '')]
        return create_grid_enhanced(fov, overlap, *params)
    raise ValueError(f'Unknown grid tiling method: "{kind}"')


def create_grid_product(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method uses the product of RA and Dec to get the RA spacings.
    """
    # Calculate steps
    step_dec = fov['dec'] * (1 - overlap['dec'])
    step_ra = fov['ra'] * (1 - overlap['ra'])

    # Create the dec strips
    pole = 90 // step_dec * step_dec
    decs = np.arange(-pole, pole + step_dec / 2, step_dec)

    # Arrange the tiles in RA
    ras = np.arange(0.0, 360.0, step_ra)
    allras, alldecs = zip(*[(ra, dec) for ra, dec in itertools.product(ras, decs)])
    allras, alldecs = np.asarray(allras), np.asarray(alldecs)

    return allras, alldecs


def create_grid_cosine(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method adjusts the RA spacings based on the cos of the declination.
    """
    # Calculate steps
    step_dec = fov['dec'] * (1 - overlap['dec'])
    step_ra = fov['ra'] * (1 - overlap['ra'])

    # Create the dec strips
    pole = 90 // step_dec * step_dec
    decs = np.arange(-pole, pole + step_dec / 2, step_dec)

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360.0, step_ra / np.cos(dec * np.pi / 180))
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_cosine_symmetric(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method adjusts the RA spacings based on the cos of the declination.

    Compared to `create_grid_cosine` this method rotates the dec strips so
    they are symmetric around the meridian.
    """
    # Calculate steps
    step_dec = fov['dec'] * (1 - overlap['dec'])
    step_ra = fov['ra'] * (1 - overlap['ra'])

    # Create the dec strips
    pole = 90 // step_dec * step_dec
    decs = np.arange(-pole, pole + step_dec / 2, step_dec)

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360.0, step_ra / np.cos(dec * np.pi / 180))
        ras += (360 - ras[-1]) / 2  # Rotate the strips so they're symmetric
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_minverlap(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method takes the overlaps given as the minimum rather than fixed,
    and then adjusts the number of tiles in RA and Dec until they overlap
    at least by the amount given.
    """
    # Create the dec strips
    pole = 90
    n_tiles = math.ceil(pole / ((1 - overlap['dec']) * fov['dec']))
    step_dec = pole / n_tiles
    north_decs = np.arange(pole, 0, step_dec * -1)
    south_decs = north_decs * -1
    decs = np.concatenate([south_decs, np.array([0]), north_decs[::-1]])

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        n_tiles = math.ceil(360 / ((1 - overlap['ra']) * fov['ra'] / np.cos(dec * np.pi / 180)))
        step_ra = 360 / n_tiles
        ras = np.arange(0, 360, step_ra)
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_enhanced(  # noqa: PLR0912, PLR0913
    fov,
    overlap,
    integer_fit=True,
    force_equator=False,
    polar_edge=False,
    corner_align=True,
):
    """Create a pointing grid to cover the whole sky.

    Parameters
    ----------
    fov : dict, with keys 'ra' and 'dec'
        The field of view in degrees in each axis.

    overlap : dict, with keys 'ra' and 'dec'
        The overlap fraction (0-1) in each axis.

    integer_fit : bool, default=True
        If True, adjust the spacing values to ensure and integer number of tile fit neatly within
            each range (previously called the "minverlap" algorithm).
        If False, use the basic spacing to produce a non-symmetric grid.

    force_equator : bool, default=False
        If True, force a declination band at dec=0 (will always produce a tile at (0,0)).
        If False, the number of bands can either be even (no equator band) or odd (equator band),
            depending on what fits best.

    polar_edge : bool, default=False
        If True, align the highest/lowest tiles with their edge at the pole.
        If False, place a tile centre on the pole instead.

    corner_align : bool, default=True
        If True, reduce gaps between tiles by ensuring the right ascension separation is never more
            than that needed to align the lower tile edges.
        If False, don't.

    """
    # Step 1: Create the dec bands
    fov_dec = fov['dec'] * (1 - overlap['dec'])
    if polar_edge:
        # Align the highest band so the midpoint of the upper edge is at 90 dec
        pole = 90 - fov_dec / 2
    elif not integer_fit:
        # This is just how the cosine algorithm calculated the pole, don't ask me why
        pole = 90 // fov_dec * fov_dec
    else:
        # Place the highest band exactly on the pole
        pole = 90
    if force_equator:  # noqa: SIM108
        # Fit between 0 and the pole
        dec_range = pole
    else:
        # Fit over the whole dec range (pole to pole)
        dec_range = pole * 2
    if integer_fit:
        # Calculate the number of steps to best fit into the given range.
        # Note that n_dec is the ideal number of steps *between* bands, creating n_dec + 1 bands.
        n_dec = math.ceil(dec_range / fov_dec)
        step_dec = dec_range / n_dec
    else:
        # Just use the basic FoV as the step size
        step_dec = fov_dec
    # Create the band arrays
    if force_equator or (integer_fit and n_dec % 2 == 0):
        # If n_dec is even then there are an old number of bands, so there is one at the equator.
        # Note we use we always add in the poles and 0 manually, to avoid floating points.
        north_decs = np.arange(pole, 0, step_dec * -1)
        south_decs = north_decs * -1
        decs = np.concatenate([south_decs, np.array([0]), north_decs[::-1]])
    else:
        # If n_dec is odd then there are an even number of bands, so nothing at the equator.
        # We could do np.arange(-pole, pole + step_dec, step_dec) and the last value should be pole,
        # but sometimes due to fp errors it goes slightly over 90deg and then we get problems...
        decs = np.arange(-1 * pole, pole, step_dec)
        decs = np.concatenate([decs, np.array([pole])])

    # Step 2: Arrange the tiles in RA for each band
    alldecs = []
    allras = []
    fov_ra = fov['ra'] * (1 - overlap['ra'])
    for dec in decs:
        if abs(dec) == 90:
            # Note cos(+/-90) = 0, so we could see issues with the 1/cos factor at the poles.
            # We actually don't, due to floating points, but better to hard code it anyway.
            ras = np.array([0])
        else:
            # Find the effective tile width for this band (including overlap)
            band_fov_ra = fov_ra / np.cos(dec * np.pi / 180)
            if corner_align:
                # Find the effective width of the tile (the difference in RA
                # between the two lower corners).
                # By carefully choosing a tile at ra=0 the ra of the south-east corner is half
                # the effective width, which is the same for all tiles in the band.
                # Note that since the grid is symmetric we take abs(dec).
                centre = SkyCoord(0 * u.deg, abs(dec) * u.deg)
                corner_se = onsky_offset(centre, (fov['ra'] / 2, -fov['dec'] / 2) * u.deg)
                max_fov_ra = corner_se.ra.value * 2
                # This is the maximum spacing, so only override the default if it is larger than
                # the maximum (usually closest to the poles).
                band_fov_ra = min(band_fov_ra, max_fov_ra)
            if integer_fit:
                # Calculate the number of steps to best fit into the given range
                n_ra = math.ceil(360 / band_fov_ra)
                step_ra = 360 / n_ra
            else:
                # Just use the basic FoV as the step size
                step_ra = band_fov_ra
            # Create the point coordinates
            ras = np.arange(0, 360, step_ra)
        # Add coordinates to lists
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    # Concatenate the lists
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_array(fov=(2.21, 2.95), shape=(4, 2), overlap=0.2):
    """Define the relative on-sky offsets for an array of rectangular FoVs.

    The default is the GOTO array of 8 UTs, arranged in 2 rows of 4, with 0.2 deg overlap.

    The fields are numbered starting in the top left, going left to right, top to bottom.
    e.g. for GOTO the (4,2) arrangement is
        +---+---+---+---+
        | 1 | 2 | 3 | 4 |
        +---+---+---+---+
        | 5 | 6 | 7 | 8 |
        +---+---+---+---+
    assuming North (increasing Dec) is up and East (increasing RA) is left.

    The returned positions are given in degrees relative to the centre, defined as (0, 0).
    For actual on-sky coordinates for a given target use the `onsky_offset` function.

    Parameters
    ----------
    fov : 2-tuple of floats, optional
        The field of view of each field in RA and Dec, in degrees.
        Default=(2.21, 2.95) (the GOTO UT FoV)
    shape : 2-tuple of ints, optional
        The number of fields spaced in RA and Dec (N_ra columns x N_dec rows).
        Default=(4, 2) (the GOTO 8-UT array arrangement)
    overlap : float or 2-tuple of floats, optional
        The amount of overlap between fields, in degrees.
        If two values are given they are the overlap in RA and Dec, a single value is used for both.
        Can be negative, in which case there will be gaps between the fields.
        Default=0.2 (the GOTO UT overlap, the same in both RA and Dec)

    Returns
    -------
    centres : `numpy.ndarray`, shape=(N_ra x N_dec, 2)
        The on-sky positions of the centres of each field, relative to the centre of the array.
    corners : `numpy.ndarray`, shape=(N_ra x N_dec, 4, 2)
        The on-sky positions of the four corners of each field, relative to the centre of the array.

    """
    if isinstance(overlap, (int, float)):
        overlap = (overlap, overlap)

    # We define UTs going in "reading" order, left to right, top to bottom.
    # This is why we reverse the ranges for each offset, and loop in Dec then RA.
    centres = []
    for j in range(shape[1])[::-1]:
        for i in range(shape[0])[::-1]:
            centres.append(  # noqa: PERF401
                (
                    (i - (shape[0] - 1) / 2) * (fov[0] - overlap[0]),
                    (j - (shape[1] - 1) / 2) * (fov[1] - overlap[1]),
                ),
            )

    # Now simply offset half the fov in each direction from the centres to find the four corners.
    # We go in order NW>NE>SE>SW, same as the grid tiles.
    corners = [
        [
            (c[0] - fov[0] / 2, c[1] + fov[1] / 2),
            (c[0] + fov[0] / 2, c[1] + fov[1] / 2),
            (c[0] + fov[0] / 2, c[1] - fov[1] / 2),
            (c[0] - fov[0] / 2, c[1] - fov[1] / 2),
        ]
        for c in centres
    ]

    return np.array(centres), np.array(corners)


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
        The tiling method to use. See `gototile.grid.create_grid` for options.
        Default is 'minverlap'.

    array_params : dict or 3-tuple, optional
        If given, define a sub-array for each position on the grid.
        The dict should contain the keys 'fov', 'shape', and 'overlap', or the parameters can be
        given as a 3-tuple of (fov, shape, overlap).
        See `gototile.grid.create_array()` for details on how the array is defined.

    """

    def __init__(self, fov, overlap=None, kind='minverlap', array_params=None):
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
        self.name = 'allsky-{}x{}-{}-{}'.format(
            self.fov['ra'].value,
            self.fov['dec'].value,
            self.overlap['ra'],
            self.overlap['dec'],
        )

        # Create the grid
        ras, decs = create_grid(self.fov, self.overlap, kind)
        self.coords = SkyCoord(ras, decs, unit=u.deg)
        self.ntiles = len(self.coords)

        # Get the tile vertices - 4 points on the corner of each tile
        # We get these by offsetting on-sky from the tile centres by half the fov in each direction.
        # Order of corners is NW>NE>SE>SW (remember RA increases going east).
        corner_offsets = [
            (-self.fov['ra'] / 2, +self.fov['dec'] / 2) * u.deg,
            (+self.fov['ra'] / 2, +self.fov['dec'] / 2) * u.deg,
            (+self.fov['ra'] / 2, -self.fov['dec'] / 2) * u.deg,
            (-self.fov['ra'] / 2, -self.fov['dec'] / 2) * u.deg,
        ]
        self.vertices = onsky_offset(self.coords, corner_offsets)

        # Give the tiles unique ids
        self.tilenums = np.arange(self.ntiles) + 1
        fill_len = len(str(max(self.tilenums)))
        self.tilenames = ['T' + str(num).zfill(fill_len) for num in self.tilenums]

        # Define the sub-array, if given
        if array_params is not None:
            if isinstance(array_params, dict):
                fov = array_params['fov']
                shape = array_params['shape']
                overlap = array_params['overlap']
            else:
                fov, shape, overlap = array_params
            self.array_params = (fov, shape, overlap)

            # For each grid point get the centres and corners of each field in the array
            array_centres, array_corners = create_array(fov, shape, overlap)
            self.array_coords = onsky_offset(self.coords, array_centres * u.deg)
            all_corners = SkyCoord([onsky_offset(self.coords, c * u.deg) for c in array_corners])
            all_corners = all_corners.reshape(len(array_corners), self.ntiles, 4)
            all_corners = all_corners.transpose(1, 0, 2)
            self.array_vertices = all_corners
        else:
            self.array_params = None
            self.array_coords = None
            self.array_vertices = None

        # Properties waiting for a skymap to be applied
        self.skymap = None
        self.pixels = None
        self.probs = None
        self.contours = None

    def __eq__(self, other):
        try:
            return (
                self.fov == other.fov and self.overlap == other.overlap and self.kind == other.kind
            )
        except AttributeError:
            return False

    def __repr__(self):
        template = 'SkyGrid(fov=({}, {}), overlap=({}, {}), kind={})'
        return template.format(
            self.fov['ra'].value,
            self.fov['dec'].value,
            self.overlap['ra'],
            self.overlap['dec'],
            self.kind,
        )

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
            pattern = r'allsky-(\d+(\.\d+)?)x(\d+(\.\d+)?)-(\d+(\.\d+)?)'
            match = re.match(pattern, name)
            if not match:
                template = 'allsky-{fov_ra}x{fov_dec}-{overlap_ra}-{overlap_dec}'
                msg = f'Grid name "{name}" not recognised, name format should match "{template}".'
                raise ValueError(msg)
            fov = (float(match.group(1)), float(match.group(3)))
            overlap = (float(match.group(5)), float(match.group(7)))
            return cls(fov, overlap)
        if name in NAMED_GRIDS:
            fov = NAMED_GRIDS[name]['fov']
            overlap = NAMED_GRIDS[name]['overlap']
            kind = NAMED_GRIDS[name]['kind']
            array_params = NAMED_GRIDS[name].get('array')
            return cls(fov, overlap, kind, array_params)

        raise ValueError(f'Name "{name}" not recognised, check `SkyGrid.get_named_grids()`.')

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
        # Store a copy of the skymap on the class
        self.skymap = skymap.copy()
        self.nside = self.skymap.nside

        # Ensure the skymap is in equatorial coordinates
        if self.skymap.coordsys != 'C':
            self.skymap.rotate('C')

        # Ensure the skymap is in units of probability, not probability density
        if self.skymap.density:
            self.skymap.density = False

        # Calculate which skymap pixels are contained within each tile,
        # then find the tile probabilities and contour levels
        self.pixels = self._get_tile_pixels(self.skymap)
        self.probs = self._get_tile_probs(self.skymap)
        self.contours = self._get_tile_contours(self.skymap)

        return self.probs

    def _get_tile_pixels(self, skymap):
        """Calculate the skymap pixel indices within each tile."""
        # Need to provide tile vertices in cartesian coordinates
        tile_vertices = self.vertices.cartesian.get_xyz(xyz_axis=2).value

        # Get the pixels within each polygon on the map.
        # `inclusive=True` will include pixels that overlap in area even if the centres aren't
        # inside the region (`fact` tells how deep to look, at nside=self.nside*fact)
        tile_pixels = [skymap.query_polygon(v, inclusive=True, fact=32) for v in tile_vertices]

        # Note the number of pixels per tile will vary, so we need dtype=object
        # (skymap.query_polygon already returns numpy arrays)
        return np.array(tile_pixels, dtype=object)

    def _get_pixel_tiles(self, skymap):
        """Calculate which tiles each skymap pixel is within."""
        if skymap == self.skymap and self.pixels is not None:
            # Use cached pixels if available
            tile_pixels = self.pixels
        else:
            tile_pixels = self._get_tile_pixels(skymap)

        # We do this by "inverting" the pixels array, so we have a list of pixels for each tile
        # rather than a list of tiles for each pixel.
        pixel_tile_dict = {}
        for tile, pixels in enumerate(tile_pixels):
            for pixel in pixels:
                if pixel not in pixel_tile_dict:
                    pixel_tile_dict[pixel] = []
                pixel_tile_dict[pixel].append(tile)

        # Convert the dictionary to a numpy array of arrays
        # Again the number of tiles each pixel is in will vary, so we need
        # to convert each sub-list to and array and use dtype=object
        pixel_tiles = [[] for _ in range(max(pixel_tile_dict.keys()) + 1)]
        for key, indices in pixel_tile_dict.items():
            pixel_tiles[key] = indices
        return np.array([np.array(tiles) for tiles in pixel_tiles], dtype=object)

    def _get_tile_probs(self, skymap):
        """Calculate the contained probabilities within each tile."""
        if skymap == self.skymap and self.pixels is not None:
            # Use cached pixels if available
            tile_pixels = self.pixels
        else:
            tile_pixels = self._get_tile_pixels(skymap)
        pixel_probs = skymap.data

        return np.array([np.sum(pixel_probs[pixels]) for pixels in tile_pixels])

    def _get_tile_contours(self, skymap, prob_limit=5):
        """Calculate the minimum contour level of each pixel.

        Unlike for SkyMaps (see `gototile.skymaptools.get_data_contours()`), the calculation for
        tiles is complicated because they can overlap, so the same pixel could be included within
        multiple tiles.

        This method iterates through the tiles by selecting the one with highest probability,
        adding it to the list, then blanking out that portion of the sky and recalculating the
        probabilities of any overlapping tiles.

        To save time the probability limit (`prob_limit`) is a way to ignore any
        tiles with a probability of less than 10^-1**prob_limit (i.e. if the prob_limit is 3 then
        it will only consider tiles with a probability of more than 0.001). The default is 5, which
        will make no difference unless you are considering the 99.99999% skymap contour level...

        The result is a minimum contour level for every tile, starting at 0 for the highest
        probability tile and increasing from there. To select all tiles within a given contour X you
        can mask for those with a contour level < X.
        """
        if skymap == self.skymap and self.pixels is not None:
            # Use cached pixels if available
            tile_pixels = self.pixels
            # Copy both, probs since we will be blanking out pixels
            tile_probs = self.probs.copy()
            pixel_probs = self.skymap.data.copy()
        else:
            tile_pixels = self._get_tile_pixels(skymap)
            tile_probs = self._get_tile_probs(skymap)
            pixel_probs = skymap.data.copy()
        pixel_tiles = self._get_pixel_tiles(skymap)

        sorted_contours = [0]
        sorted_index = []
        for i in range(len(tile_pixels)):
            # Find the tile with the highest probability
            high_tile_prob = max(tile_probs)
            if (
                prob_limit is not None and high_tile_prob < 10 ** (-1 * prob_limit)
            ) or high_tile_prob == 0:
                # We've reached the probability limit, or we've already blacked out all the pixels
                # and there's no probability left!
                # Any remaining tiles will have their contour level set to 1.
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
            # Recalculate the probability within any changed tiles
            changed_tiles = np.unique(np.concatenate(pixel_tiles[high_tile_pixels]))
            tile_probs[changed_tiles] = np.array(
                [np.sum(pixel_probs[pixels]) for pixels in tile_pixels[changed_tiles]],
            )

        # Start from a contour level of 1, only replace those with the calculated values
        # Note we don't include the last value in the contours list, since we want the
        # contour level to start at 0.
        # This ensures the highest probability tile has a contour of 0, so it is always selected
        # when masking for tiles below a given contour level.
        contours = np.ones(self.ntiles)
        contours[np.array(sorted_index)] = np.array(sorted_contours)[:-1]

        return contours

    def _get_test_map(self, nside=None):
        """Create a basic empty skymap, useful for statistical calculations.

        We could do this in __init__, but why not save time and only do if if we need it?
        """
        if hasattr(self, '_base_nside') and self._base_nside == nside:
            return
        if nside is None:
            nside = 128
        self._base_nside = nside
        self._base_skymap = SkyMap(np.zeros(12 * self._base_nside**2), order='NESTED')
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
        # TODO: spherical_geometry has better functions for things like overlaps, areas and
        #       finding if a point is within an area

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
                names = [
                    self.tilenames[i] for i in range(self.ntiles) if pixel in self._base_pixels[i]
                ]
                tilenames.append(names)

        if scalar:
            return tilenames[0]
        return tilenames

    def get_field(self):
        """Find which field(s) the given coordinates fall within."""
        # TODO: With the array system we should be able to find the exact tile and field within
        #       that tile's array the coordinates are in.
        #       However the way we do it with the base HEALPix map would involve finding which
        #       pixels are in every field.
        #       Instead I think spherical_geometry is the way to go, but that would require
        #       a bigger requite of some of the internal functions.
        raise NotImplementedError

    def get_visible_tiles(
        self,
        locations,
        time_range=None,
        alt_limit=30,
        sun_limit=-15,
        any_all='any',
    ):
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
            mask = np.full(self.ntiles, fill_value=False)
        elif any_all == 'all':
            mask = np.full(self.ntiles, fill_value=True)
        else:
            raise ValueError(f'Invalid value for any_all: "{any_all}"')

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
        return [self.tilenames.index(tile) for tile in tilenames]

    def get_coordinates(self, tilenames=None):
        """Return the central coordinates of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the coordinates of.
            If not given, return the central coordinates of all tiles in the grid.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The central coordinates of the given tile(s).

        """
        if tilenames is None:
            return self.coords

        indices = self._get_tilename_indices(tilenames)
        return self.coords[indices]

    def get_vertices(self, tilenames=None):
        """Return coordinates of the four corners of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the vertices of.
            If not given, return the vertices of all tiles in the grid.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the vertices of the given tile(s).
            Will be an array with shape (ntiles, 4).

        """
        if tilenames is None:
            return self.vertices

        indices = self._get_tilename_indices(tilenames)
        return self.vertices[indices]

    def get_edges(self, tilenames=None, edge_points=4):
        """Return coordinates along the edges of the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the edges of.
            If not given, return the edges of all tiles in the grid.
        edge_points : int, optional
            The number of points to find along each tile edge (not including the corners).
            If edge_points=0 only the 4 corners will be returned.
            Default=4.

        Returns
        -------
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the edge points of the given tile(s).
            Will be an array with shape (ntiles, 4*(edge_points+1)).

        """
        if tilenames is None:
            return interpolate_points(self.vertices, edge_points)

        indices = self._get_tilename_indices(tilenames)
        return interpolate_points(self.vertices[indices], edge_points)

    def get_pixels(self, tilenames=None):
        """Get the skymap pixels contained within the given tile(s).

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the pixels of.
            If not given, return the pixels of all tiles in the grid.

        Returns
        -------
        ipix : list of int
            Pixel indices covering the area of the given tile(s).

        """
        if self.pixels is None:
            raise ValueError('SkyGrid does not have a SkyMap applied')
        if tilenames is None:
            return sorted({ipix for tile_pix in self.pixels for ipix in tile_pix})

        indices = self._get_tilename_indices(tilenames)
        if isinstance(indices, int):
            pixels = self.pixels[indices]
        else:
            pixels = [ipix for tile_pix in self.pixels[indices] for ipix in tile_pix]
        return sorted(set(pixels))

    def get_probability(self, tilenames=None):
        """Return the contained probability within the given tile(s).

        If multiple tiles are given, the probability only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the probability within.
            If not given, return the total probability covered by all tiles in the grid.

        Returns
        -------
        probability : int
            The total skymap value within the area covered by the given tile(s).

        """
        pixels = self.get_pixels(tilenames)
        return self.skymap.data[pixels].sum()

    def get_area(self, tilenames=None):
        """Return the sky area contained within the given tile(s) in square degrees.

        If multiple tiles are given, the area only be included once in any overlaps.

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the area of.
            If not given, return the total area covered by all tiles in the grid.

        Returns
        -------
        area : int
            The total sky area covered by the given tile(s), in square degrees.

        """
        self._get_test_map()
        if tilenames is None:
            pixels = sorted({ipix for tile_pix in self._base_pixels for ipix in tile_pix})
            return self._base_skymap.get_pixel_area(pixels)

        indices = self._get_tilename_indices(tilenames)
        if isinstance(indices, int):
            pixels = sorted(set(self._base_pixels[indices]))
        else:
            pixels = [ipix for tile_pix in self._base_pixels[indices] for ipix in tile_pix]
        pixels = sorted(set(pixels))
        return self._base_skymap.get_pixel_area(pixels)

    def get_areas(self, tilenames=None):
        """Return the areas contained within each of the given tile(s) in square degrees.

        Note although every tile should have the same area (equal to fov_ra * fov_dec) there will
        be slight differences from the resolution of the HEALPix grid.

        For most cases you'll want to use the `SkyGrid.get_tile_area()` function instead, which
        gives the COMBINED area of the given tiles (i.e. only including overlapping areas once).

        Parameters
        ----------
        tilenames : str or list of str, optional
            The name(s) of the tile(s) to find the area of.
            If not given, return the areas covered by all tiles in the grid.

        Returns
        -------
        area : int or list of int
            The areas covered by the given tile(s), in square degrees.

        """
        if tilenames is None:
            return [self.get_area(tile) for tile in self.tilenames]

        if isinstance(tilenames, str):
            return self.get_area(tilenames)
        return [self.get_area(tile) for tile in tilenames]

    def get_table(self):
        """Return an astropy QTable containing the name and coordinates of each tile in the grid.

        If a sky map has been applied to the grid the table will include columns with
            the contained probability within each tile and the contour level it is within.
        """
        col_names = ['tilename', 'ra', 'dec', 'prob', 'contour']
        col_data = [
            self.tilenames,
            self.coords.ra,
            self.coords.dec,
            self.probs if self.probs is not None else np.zeros(self.ntiles),
            self.contours if self.contours is not None else np.zeros(self.ntiles),
        ]
        return QTable(col_data, names=col_names)

    def _get_pixel_count(self, nside=128):
        """For each pixel in the base skymap, count of the number of tiles it falls within."""
        # Create test skymap, if it hasn't already been generated
        self._get_test_map(nside)

        count = np.zeros(12 * self._base_nside**2)
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
        return table.group_by('in_tiles')

    def _get_tile_paths(self, meridian_split=False, edge_points=4):
        """Create and cache Matplotlib Patches to use when plotting tiles."""
        # Used cached versions to save time when repeatedly plotting
        if not meridian_split and hasattr(self, '_paths') and self._paths_points == edge_points:
            return self._paths
        if (
            meridian_split
            and hasattr(self, '_paths_split')
            and self._paths_split_points == edge_points
        ):
            return self._paths_split

        # We can't just plot the four corners (already saved under self.vertices) because that will
        # plot straight lines between them. That will look bad, because we're on a sphere.
        # Instead we get some intermediate points along the edges, so they look better when plotted.
        # (Admittedly this is only obvious with very large tiles, but it's still good to do).
        edges = interpolate_points(self.vertices, edge_points)

        # Get list of matplotlib paths for the tile areas
        paths = [coords_to_path(edge_coords, meridian_split) for edge_coords in edges]

        if not meridian_split:
            self._paths = paths
            self._paths_points = edge_points
        else:
            self._paths_split = paths
            self._paths_split_points = edge_points

        return paths

    def _get_array_paths(self, meridian_split=False, edge_points=1):
        """Create and cache Matplotlib Patches to use when plotting sub-array fields."""
        # Used cached versions to save time when repeatedly plotting
        if (
            not meridian_split
            and hasattr(self, '_array_paths')
            and self._array_paths_points == edge_points
        ):
            return self._array_paths
        if (
            meridian_split
            and hasattr(self, '_array_paths_split')
            and self._array_paths_split_points == edge_points
        ):
            return self._array_paths_split

        if self.array_params is None:
            raise ValueError('SkyGrid does not have an array defined')

        # It's quicker go through each field at once rather than loop though each tile
        # (actually if you don't interpolate it takes about the same time, but if you do have
        #  edge_points>0 then doing that e.g. 8 times is much quicker than 1048!)
        all_corners = self.array_vertices.transpose(1, 0, 2)
        all_paths = []
        for corners in all_corners:
            edges = interpolate_points(corners, edge_points)
            paths = [coords_to_path(edge_coords, meridian_split) for edge_coords in edges]
            all_paths.append(paths)
        all_paths = np.array(all_paths).T

        if not meridian_split:
            self._array_paths = all_paths
            self._array_paths_points = edge_points
        else:
            self._array_paths_split = all_paths
            self._array_paths_split_points = edge_points

        return all_paths

    def plot_tile(self, axes, tilename, *args, **kwargs):
        """Plot a Patch for a single tile onto the given axes."""
        return self.plot_tiles(axes, [tilename], *args, **kwargs)

    def plot_tiles(self, axes, tilenames=None, edge_points=4, *args, **kwargs):
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
        paths = self._get_tile_paths(meridian_split, edge_points)
        if tilenames is not None:
            indexes = [self.tilenames.index(tilename) for tilename in tilenames]
            paths = np.array(paths)[indexes]

        if len(paths) > 1:
            # Fomr multiple tiles add all the patches together as a collection
            patches = [PathPatch(path) for path in paths]
            collection = PatchCollection(
                patches,
                *args,
                transform=axes.get_transform('world'),
                **kwargs,
            )
            axes.add_collection(collection)
            return collection
        # For a single tile we can just return a single patch
        patch = PathPatch(paths[0], *args, transform=axes.get_transform('world'), **kwargs)
        axes.add_patch(patch)
        return patch

    def plot_array(self, axes, tilenames=None, edge_points=1, *args, **kwargs):
        """Plot a sub-array of fields aligned to the grid onto the given axes."""
        if self.array_params is None:
            raise ValueError('SkyGrid does not have an array defined')

        # Add default arguments
        if 'fc' not in kwargs and 'facecolor' not in kwargs:
            kwargs['fc'] = 'tab:blue'
        if 'ec' not in kwargs and 'edgecolor' not in kwargs:
            kwargs['ec'] = 'black'
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 0.5

        # Get the patches for each field (will be cached after the first use)
        meridian_split = 'Mollweide' in axes.__class__.__name__
        paths = self._get_array_paths(meridian_split, edge_points)
        if tilenames is not None:
            indices = self._get_tilename_indices(tilenames)
            paths = np.array(paths)[indices]

        # Make a single collection from all the paths
        # PatchCollections are 1D, so we have to flatten the array first
        paths = paths.flatten()
        patches = [PathPatch(path) for path in paths]
        collection = PatchCollection(
            patches,
            *args,
            transform=axes.get_transform('world'),
            **kwargs,
        )
        axes.add_collection(collection)
        return collection

    def plot(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        title=None,
        filename=None,
        dpi=90,
        figsize=(8, 6),
        plot_type='mollweide',
        center=(0, 45),
        radius=10,
        color=None,
        linecolor=None,
        linewidth=None,
        alpha=0.3,
        discrete_colorbar=False,
        discrete_stepsize=1,
        colorbar_limits=None,
        colorbar_orientation='v',
        highlight=None,
        highlight_color=None,
        highlight_label=None,
        coordinates=None,
        tilenames=None,
        text=None,
        plot_skymap=False,
        plot_contours=False,
        plot_stats=False,
    ):
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
        tile_patches = self.plot_tiles(
            axes,
            fc='blue',
            ec='none',
            lw=0,
            alpha=alpha,
            zorder=2,
        )

        # Also plot on the lines over the top
        edge_patches = self.plot_tiles(
            axes,
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
                plt.text(
                    coord.ra.deg,
                    coord.dec.deg,
                    str(text[name]),
                    color='k',
                    weight='bold',
                    fontsize=6,
                    ha='center',
                    va='center',
                    clip_on=True,
                    transform=transform,
                )

        # Plot skymap probabilities
        if plot_skymap is True:
            if self.skymap is None:
                raise ValueError('SkyGrid does not have a SkyMap applied')

            # Set the probability array to the color array
            skymap_patches = self.plot_tiles(
                axes,
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
            self.skymap.plot_contours(
                axes,
                levels=[0.5, 0.9],
                colors='black',
                linewidths=0.5,
                zorder=99,
            )

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
            points = axes.scatter(
                coords.ra.deg,
                coords.dec.deg,
                transform=transform,
                s=1,
                c=count,
                cmap=cmap,
                norm=norm,
                zorder=0,
            )

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
                    for name in color:
                        index = self.tilenames.index(name)
                        color_array[index] = color[name]
                    tile_patches.set_facecolor(np.array(color_array))
                except Exception:  # noqa: BLE001
                    try:
                        # Create the color array
                        color_array = np.array([np.nan] * self.ntiles)
                        for name in color:
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
                                colorbar_limits = (
                                    int(np.floor(np.min(masked_array))),
                                    int(np.ceil(np.max(masked_array))),
                                )
                            boundaries = np.linspace(
                                colorbar_limits[0],
                                colorbar_limits[1] + 1,
                                (colorbar_limits[1] + 1 - colorbar_limits[0] + 1),
                            )
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
                            cb = fig.colorbar(
                                tile_patches,
                                ax=axes,
                                fraction=0.03,
                                pad=0.05,
                                aspect=50,
                                orientation='horizontal',
                            )
                        else:
                            cb = fig.colorbar(tile_patches, ax=axes, fraction=0.02, pad=0.05)
                        if discrete_colorbar:
                            tick_labels = np.arange(
                                colorbar_limits[0],
                                colorbar_limits[1] + 1,
                                discrete_stepsize,
                                dtype=int,
                            )
                            tick_location = tick_labels + 0.5
                            cb.set_ticks(tick_location)
                            cb.set_ticklabels(tick_labels)

                    except Exception:  # noqa: BLE001
                        raise ValueError('Invalid entries in color array') from None

            elif isinstance(color, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(color) == self.ntiles:
                    raise ValueError('List of colors must be same length as grid.ntiles')

                # Could be a list of weights or a list of colors
                try:
                    tile_patches.set_facecolor(np.array(color))
                except Exception:  # noqa: BLE001
                    try:
                        tile_patches.set_array(np.array(color))
                        fig.colorbar(tile_patches, ax=axes, fraction=0.02, pad=0.05)
                    except Exception:  # noqa: BLE001
                        raise ValueError('Invalid entries in color array') from None

            else:
                # Might just be a string color name
                tile_patches.set_facecolor(color)

        # Plot tile linecolors
        if linecolor is not None:
            if isinstance(linecolor, dict):
                # Should be a dict with keys as tile names
                try:
                    linecolor_array = np.array(['black'] * self.ntiles, dtype=object)
                    for name in linecolor:
                        index = self.tilenames.index(name)
                        linecolor_array[index] = linecolor[name]
                    edge_patches.set_edgecolor(np.array(linecolor_array))
                except Exception:  # noqa: BLE001
                    raise ValueError('Invalid entries in linecolor array') from None

            elif isinstance(linecolor, (list, tuple, np.ndarray)):
                # A list-like of colors, should be same length as number of tiles
                if not len(linecolor) == self.ntiles:
                    raise ValueError('List of linecolors must be same length as grid.ntiles')

                # Should be a list of color string
                try:
                    edge_patches.set_edgecolor(np.array(linecolor))
                except Exception:  # noqa: BLE001
                    raise ValueError('Invalid entries in linecolor array') from None

            else:
                # Might just be a string color name
                edge_patches.set_edgecolor(linecolor)

        # Plot tile linewidths
        if linewidth is not None:
            if isinstance(linewidth, dict):
                # Should be a dict with keys as tile names
                try:
                    linewidth_array = np.array([0.5] * self.ntiles)
                    for name in linewidth:
                        index = self.tilenames.index(name)
                        linewidth_array[index] = linewidth[name]
                    edge_patches.set_linewidth(np.array(linewidth_array))
                except Exception:  # noqa: BLE001
                    raise ValueError('Invalid entries in linewidth array') from None

            elif isinstance(linewidth, (list, tuple, np.ndarray)):
                # A list-like of floats, should be same length as number of tiles
                if not len(linewidth) == self.ntiles:
                    raise ValueError('List of linewidths must be same length as grid.ntiles')

                # Should be a list of floats
                try:
                    edge_patches.set_linewidth(np.array(linewidth))
                except Exception:  # noqa: BLE001
                    raise ValueError('Invalid entries in linewidth array') from None

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
                    self.plot_tiles(
                        axes,
                        fc='none',
                        ec=np.array(linecolor_array),
                        lw=np.array(linewidth_array),
                        alpha=0.5,
                        zorder=9,
                    )
                    # Add to legend
                    if highlight_label is not None:
                        label = highlight_label + f' ({len(highlight)} tiles)'
                        patch = Patch(
                            facecolor='none',
                            edgecolor=highlight_color,
                            linewidth=1.5,
                            label=label,
                        )
                        legend_patches.append(patch)
                except Exception:  # noqa: BLE001
                    raise ValueError('Invalid entries in highlight list') from None
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
                        self.plot_tiles(
                            axes,
                            fc='none',
                            ec=np.array(linecolor_array),
                            lw=np.array(linewidth_array),
                            alpha=0.5,
                            zorder=9 + len(highlight) - j,
                        )
                        # Add to legend
                        if highlight_label is not None:
                            label = highlight_label[j] + f' ({len(tilelist)} tiles)'
                            patch = Patch(
                                facecolor='none',
                                edgecolor=linecolor,
                                linewidth=1.5,
                                label=label,
                            )
                            legend_patches.append(patch)
                except Exception:  # noqa: BLE001
                    raise ValueError('Invalid entries in highlight list') from None

            # Display legend
            if len(legend_patches) > 0:
                plt.legend(
                    handles=legend_patches,
                    loc='center',
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=3,
                ).set_zorder(999)

        # Plot coordinates
        if coordinates:
            axes.scatter(
                coordinates.ra.value,
                coordinates.dec.value,
                transform=transform,
                s=99,
                c='blue',
                marker='*',
                zorder=9,
            )
            if coordinates.isscalar:
                coordinates = SkyCoord([coordinates])
            for coord in coordinates:
                axes.text(
                    coord.ra.value,
                    coord.dec.value,
                    coord.to_string('hmsdms').replace(' ', '\n') + '\n',
                    transform=transform,
                    ha='center',
                    va='bottom',
                    size='x-small',
                    zorder=12,
                )

        # Set title
        if title is None:
            title = 'All sky grid (fov={}x{}, overlap={},{})'.format(
                self.fov['ra'],
                self.fov['dec'],
                self.overlap['ra'],
                self.overlap['dec'],
            )
            if plot_skymap and self.skymap is not None:
                title += '\n' + 'with skymap'
        axes.set_title(title, y=1.05)

        # Save or show
        if filename:
            plt.savefig(filename, dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
