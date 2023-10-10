"""Module containing utility functions for the SkyGrid class."""

import itertools
import math

from astropy import units as u
from astropy.coordinates import SkyCoord

import numpy as np


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
    elif kind == 'cosine_symmetric':
        return create_grid_cosine_symmetric(fov, overlap)
    elif kind == 'product':
        return create_grid_product(fov, overlap)
    elif kind == 'minverlap':
        return create_grid_minverlap(fov, overlap)
    elif kind == 'enhanced':
        return create_grid_enhanced(fov, overlap)
    elif kind.startswith('enhanced'):
        params = [i == '1' for i in kind.strip('enhanced')]
        return create_grid_enhanced(fov, overlap, *params)
    else:
        raise ValueError('Unknown grid tiling method: "{}"'.format(kind))


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
    ras = np.arange(0.0, 360., step_ra)
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
        ras = np.arange(0.0, 360., step_ra / np.cos(dec * np.pi / 180))
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
        ras = np.arange(0.0, 360., step_ra / np.cos(dec * np.pi / 180))
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


def create_grid_enhanced(fov, overlap, integer_fit=True, force_equator=False, polar_edge=False,
                         corner_align=True):
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
    if force_equator:
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
                # By choosing a tile at ra=0 the ra of the south-east corner is half
                # the effective width, which is the same for all tiles in the band.
                # Note that since the grid is symmetric we take abs(dec).
                centre = SkyCoord(0 * u.deg, abs(dec) * u.deg)
                corners = get_tile_vertices(centre,
                                            {'ra': fov['ra'] * u.deg,
                                             'dec': fov['dec'] * u.deg})
                corner_se = corners[2]
                max_fov_ra = corner_se.ra.value * 2
                # This is the maximum spacing, so only override the default if it is larger than
                # the maximum (usually closest to the poles).
                if band_fov_ra > max_fov_ra:
                    band_fov_ra = max_fov_ra
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


def get_tile_vertices(centre, fov):
    """Calculate the coordinates of the tile vertices.

    Parameters
    ----------
    centre : `astropy.coordinates.SkyCoord`
        The coordinates of the tile centre.
    fov : dict
        The field of view of the tile, with keys of 'ra' and 'dec'

    Returns
    -------
    vertices : `astropy.coordinates.SkyCoord`
        An array with length `n` the same as centre and shape `(n, 4)`

    """
    # Find the tan of half the angles
    tan_ra = np.tan(fov['ra'] / 2)
    tan_dec = np.tan(fov['dec'] / 2)

    # Find the angle from the centre to the corners
    theta = np.arctan(tan_ra / tan_dec)

    # Find the distance to move from the centre to the corners
    phi = np.arctan(tan_ra / np.sin(theta))  # == np.arctan(tan_dec/np.cos(theta))

    # Get array of angles (NW>NE>SE>SW)
    angles = u.Quantity([-theta, theta, 180 * u.deg - theta, 180 * u.deg + theta])

    # Offset all at once
    # (see https://github.com/astropy/astropy/issues/10083)
    corners = centre.directional_offset_by(angles[:, np.newaxis], phi).T

    # If input was a single coordinate return a 1D array
    if centre.isscalar:
        corners = corners[0]

    return corners


def get_tile_edges(centre, fov, edge_points=5):
    """Get points along the edges of the tile.

    Parameters
    ----------
    centre : `astropy.coordinates.SkyCoord`
        The coordinates of the tile centre.
    fov : dict
        The field of view of the tile, with keys of 'ra' and 'dec'
    edge_points : int, default=5
        The number of points to find along each tile edge
        If edge_points=0 only the 4 corners will be returned

    Returns
    -------
    coords : `astropy.coordinates.SkyCoord`
        An array with length `n` the same as centre and shape `(n, 4*(edge_points+1))`

    """
    # We need a vector input, we can convert back at the end
    scalar = False
    if centre.isscalar:
        centre = SkyCoord([centre])
        scalar = True

    # First get the positions of the corners for each tile
    corners = get_tile_vertices(centre, fov)

    # If edge_points=0 then just return the corners
    if edge_points == 0:
        if scalar:
            return corners[0]
        return corners

    # Split out arrays for each corner
    corner_nw, corner_ne, corner_se, corner_sw = corners.T

    # Find the angles between the corners
    ang_n = corner_nw.position_angle(corner_ne)
    ang_e = corner_ne.position_angle(corner_se)
    ang_s = corner_se.position_angle(corner_sw)
    ang_w = corner_sw.position_angle(corner_nw)

    # Find the separations between the corners
    # Because the tiles all have the same FoV they have the same separation N/S and E/W
    # (this is not true for the angles!), so we only need to do it for the first tile
    sep_ns = corner_nw[0].separation(corner_ne[0])  # = corner_se[0].separation(corner_sw[0])
    sep_ew = corner_ne[0].separation(corner_se[0])  # = corner_sw[0].separation(corner_nw[0])

    # Calculate the position of the points along the edges
    steps = edge_points + 1
    steps_ns = np.arange(0, sep_ns.deg, sep_ns.deg / steps) * u.deg
    steps_ew = np.arange(0, sep_ew.deg, sep_ew.deg / steps) * u.deg

    # Find the points by moving from corners
    # (see https://github.com/astropy/astropy/issues/10083)
    # Maybe this could all be done as one action, but that might get confusing
    points_n = SkyCoord(corner_nw.directional_offset_by(ang_n, steps_ns[:, np.newaxis]))
    points_e = SkyCoord(corner_ne.directional_offset_by(ang_e, steps_ew[:, np.newaxis]))
    points_s = SkyCoord(corner_se.directional_offset_by(ang_s, steps_ns[:, np.newaxis]))
    points_w = SkyCoord(corner_sw.directional_offset_by(ang_w, steps_ew[:, np.newaxis]))

    # Combine, reshape and transpose
    points = SkyCoord([points_n, points_e, points_s, points_w])
    points = points.reshape(4 * steps, len(centre)).T

    # If input was a single coordinate return a 1D array
    if scalar:
        points = points[0]

    return points
