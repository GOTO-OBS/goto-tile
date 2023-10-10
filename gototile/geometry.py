"""Module containing utility functions for spherical geometry."""

from astropy import units as u
from astropy.coordinates import SkyCoord

import numpy as np


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
