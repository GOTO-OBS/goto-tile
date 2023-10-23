"""Module containing utility functions for spherical geometry."""

from astropy import units as u
from astropy.coordinates import SkyCoord

from matplotlib.path import Path

import numpy as np


def onsky_offset(coords, offsets):
    """Find coordinates of points offset from the target coordinates by the given on-sky distances.

    This function is used when considering points offset off of the celestial sphere, e.g.
    for finding the edge of the field of view of a telescope.

    Parameters
    ----------
    coords : `astropy.coordinates.SkyCoord`, or list of `astropy.coordinates.SkyCoord`
        The coordinates of target to offset from.

    offsets : 2-tuple of `astropy.units.Quantity`, or list of same
        The on-sky offsets in RA and Dec to apply to the target coordinates.
        Should be in angular units.

    Returns
    -------
    offset_coords : `astropy.coordinates.SkyCoord`
        The coordinates of the points offset from the target coordinates.
        Will be a single coordinate if one target and one offset are given,
        otherwise will be an N x M array for N targets and M offsets.

    """
    # The coordinates should be a SkyCoord object, scalar or array
    # If it's a list of SkyCoords then convert it into a single array
    if not isinstance(coords, SkyCoord):
        coords = SkyCoord(coords)

    # The offsets also have to be Quantity arrays, split into RA and Dec
    if len(offsets) == 2 and isinstance(offsets[0], u.Quantity) and offsets[0].isscalar:
        ra_off = u.Quantity([offsets[0]])
        dec_off = u.Quantity([offsets[1]])
    else:
        ra_off = u.Quantity([o[0] for o in offsets])
        dec_off = u.Quantity([o[1] for o in offsets])

    # Find the tan of the offset distances, these are used a lot in the formulae below.
    tan_ra = np.tan(ra_off)
    tan_dec = np.tan(dec_off)

    # 1) Find the bearing from north of each point from the target coordinate
    # This is given by tan(theta) = tan(ra)/tan(dec), unless...
    # If tan_ra is 0 the offset is in dec only and the point is directly above or below the target
    # If tan_dec is 0 the offset is in ra only and the point is directly to the left or right
    # Either of these can lead to problems with this method.
    theta = np.zeros_like(tan_ra) * u.rad
    # If tan_dec is 0 then you get a division by zero error, so set theta directly
    theta[(tan_dec == 0) & (tan_ra > 0)] = (np.pi / 2) * u.rad
    theta[(tan_dec == 0) & (tan_ra < 0)] = -(np.pi / 2) * u.rad
    # If tan_ra is 0 then it's okay, except that the arctan of 0 is 0 regardless of if the
    # point is above or below the target. So set theta directly here too.
    theta[(tan_ra == 0) & (tan_dec > 0)] = 0 * u.rad
    theta[(tan_ra == 0) & (tan_dec < 0)] = np.pi * u.rad
    # If they're both zero then there's no offset, so theta is undefined - just leave as zero
    theta[(tan_ra == 0) & (tan_dec == 0)] = 0 * u.rad
    # Set the rest using the correct formula
    mask = (tan_ra != 0) & (tan_dec != 0)
    theta[mask] = np.arctan(tan_ra[mask] / tan_dec[mask])

    # 2) Find the on-sky distance from the given coordinate to each point
    # There are two formulae, tan(phi) = tan(ra)/sin(theta) and tan(phi) = tan(dec)/cos(theta)
    # They're the same, unless either offset is 0 in which case it breaks.
    # Hence the careful masking below.
    phi = np.zeros_like(theta)
    # If they're both zero then the distance is obviously zero...
    phi[(tan_ra == 0) & (tan_dec == 0)] = 0 * u.rad
    # If tan_ra is 0 then use the formula with tan_dec
    mask = (tan_ra == 0) & (tan_dec != 0)
    phi[mask] = np.arctan(tan_dec[mask] / np.cos(theta[mask]))
    # If tan_dec is 0 then use the formula with tan_ra
    mask = (tan_ra != 0) & (tan_dec == 0)
    phi[mask] = np.arctan(tan_ra[mask] / np.sin(theta[mask]))
    # Any other points can use either formula, here we just use the one with tan_ra
    mask = (tan_ra != 0) & (tan_dec != 0)
    phi[mask] = np.arctan(tan_ra[mask] / np.sin(theta[mask]))

    # 3) With the bearing and distance we can offset and return the new coordinates
    # The [:, np.newaxis] allows multiple coordinates to be offset at once
    offset_coords = coords.directional_offset_by(theta[:, np.newaxis], phi[:, np.newaxis]).T

    # If the input was a single coordinate then return a 1D array
    if coords.isscalar:
        offset_coords = offset_coords[0]
    # If only a single offset was given then return a single coordinate
    if len(ra_off) == 1:
        offset_coords = offset_coords[0]
    # Together the above should produce the expected output:
    # scalar target, single offset -> scalar output
    # scalar target, array offsets -> 1D array output
    # array target, single offset -> 1D array output
    # array target, array offsets -> 2D array output

    return offset_coords


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
    # Get the offsets from the centre to the corners
    # Going in order NW>NE>SE>SW (remember RA increases going east)
    offsets = [
        (+ fov['dec'] / 2, - fov['ra'] / 2),
        (+ fov['dec'] / 2, + fov['ra'] / 2),
        (- fov['dec'] / 2, + fov['ra'] / 2),
        (- fov['dec'] / 2, - fov['ra'] / 2),
    ]
    return onsky_offset(centre, offsets)


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


def get_tile_path(edge_coords, meridian_split=False):
    """Create a Matplotlib Path for a region defined by the given edge coordinates.

    The main benefit of this function is that it can handle shapes that pass over the RA=0 meridian,
    which would otherwise not fill properly when plotted using the standard mollweide projection.
    It can also account for shapes that pass over one of the poles.

    Parameters
    ----------
    edge_coords : `astropy.coordinates.SkyCoord`
        The coordinates of the points that define the edge of the shape.
        Should be a closed loop, i.e. the first and last points should be the same.
    meridian_split : bool, default=False
        If True and the given shape would over the meridian or the pole the Path will be adjusted
        to allow it to be plotted correctly.

    Returns
    -------
    path : `matplotlib.path.Path`
        A closed path object that can be used to plot the shape.
        If the Path would cross over the meridian then it will be split into two sections, but
        a single Path object will be returned (using Path.make_compound_path).

    """
    ra = edge_coords.ra.deg
    dec = edge_coords.dec.deg

    # Check if the tile passes over the RA=0 line
    overlaps_meridian = any(ra < 90) and any(ra > 270)
    if not overlaps_meridian or not meridian_split:
        # If not then just make a normal closed path
        return Path(np.array((ra, dec)).T, closed=True)

    # If it does then we need to do some extra work
    if any(np.logical_and(ra > 90, ra < 270)):
        # This tile goes over the poles
        # To get it to fill we need to add extra points at the pole itself
        # First sort by RA
        ra, dec = zip(*sorted(zip(ra, dec), key=lambda radec: radec[0]))

        # Now add extra points
        pole = 90 if np.all(np.array(dec) > 0) else -90
        ra = np.array([0] + list(ra) + [360, 360])
        dec = np.array([pole] + list(dec) + [dec[0], pole])

        # Return the path with the extra points
        return Path(np.array((ra, dec)).T, closed=True)

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

        # Create the paths, then combine them and return a single Path object
        path_l = Path(np.array((ra_l, dec_l)).T, closed=True)
        path_r = Path(np.array((ra_r, dec_r)).T, closed=True)
        return Path.make_compound_path(path_l, path_r)
