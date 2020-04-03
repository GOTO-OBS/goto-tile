from __future__ import division

try:
    import cPickle as pickle
except ImportError:
    import pickle
import itertools as it
import gzip
import os
import math
import tempfile
import logging
import multiprocessing
import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable

from .math import spherical_to_cartesian, cartesian_to_spherical, intersect, interpolate, cartesian_to_celestial
from .math import RAD, PI


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
        - 'minverlap' (default):
                The latest algorithm, uses the overlap as a minimum parameter to fit an integer
                number of evenly-spaced tiles into each row.
        - 'minverlap_enhanced':
                An attempted enhnaced version of the `minverlap` algorithm which minimises gaps
                at high altitudes. Currently not very efficient, so 'minverlap' is reccomended.
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
    elif kind == 'minverlap_enhanced':
        return create_grid_minverlap_enhanced(fov, overlap)
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
    decs = np.arange(-pole, pole+step_dec/2, step_dec)

    # Arrange the tiles in RA
    ras = np.arange(0.0, 360., step_ra)
    allras, alldecs = zip(*[(ra, dec) for ra, dec in it.product(ras, decs)])
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
    decs = np.arange(-pole, pole+step_dec/2, step_dec)

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360., step_ra/np.cos(dec*RAD))
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
    decs = np.arange(-pole, pole+step_dec/2, step_dec)

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        ras = np.arange(0.0, 360., step_ra/np.cos(dec*RAD))
        ras += (360-ras[-1])/2  # Rotate the strips so they're symmetric
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
    n_tiles = math.ceil(pole/((1-overlap['dec'])*fov['dec']))
    step_dec = pole/n_tiles
    north_decs = np.arange(pole, 0, step_dec * -1)
    south_decs = north_decs * -1
    decs = np.concatenate([south_decs, np.array([0]), north_decs[::-1]])

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        n_tiles = math.ceil(360/((1-overlap['ra'])*fov['ra']/np.cos(dec*RAD)))
        step_ra = 360/n_tiles
        ras = np.arange(0, 360, step_ra)
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def create_grid_minverlap_enhanced(fov, overlap):
    """Create a pointing grid to cover the whole sky.

    This method takes the overlaps given as the minimum rather than fixed,
    and then adjusts the number of tiles in RA and Dec until they overlap
    at least by the amount given.

    This is the second version of the minverlap algorithm.
    In this version the tiles are placed slightly closer to close some of
    the gaps in RA.
    Instead of aligning the tiles based on the declination of the centre
    the declination of the lower (in the north) / upper (in the south) courners
    is used.
    This has the effect of overlapping the courners rather than the centre of
    the sides of adjacent tiles, thereby reducing the gaps between the tiles.
    """
    # Create the dec strips
    pole = 90
    n_tiles = math.ceil(pole/((1-overlap['dec'])*fov['dec'])) + 1  # Bodge
    step_dec = pole/n_tiles
    north_decs = np.arange(pole, 0, step_dec * -1)
    south_decs = north_decs * -1
    decs = np.concatenate([south_decs, np.array([0]), north_decs[::-1]])

    # Arrange the tiles in RA
    alldecs = []
    allras = []
    for dec in decs:
        if 90 > abs(dec) > 0:
            dec2 = abs(dec) - fov['dec']/2
            dec3 = 90 - np.sqrt((90-dec2)**2 + (fov['ra']/2)**2)
        else:
            dec2 = dec
            dec3 = dec
        n_tiles = math.ceil(360/((1-overlap['ra'])*fov['ra']/np.cos(dec3*RAD)))
        step_ra = 360/n_tiles
        ras = np.arange(0, 360, step_ra)
        allras.append(ras)
        alldecs.append(dec * np.ones(ras.shape))
    allras = np.concatenate(allras)
    alldecs = np.concatenate(alldecs)

    return allras, alldecs


def get_tile_vertices(coords, fov):
    """Get points defining the tile vertices from a list of coordinates and field of view.

    Understanding this is a work in progress...

    Returns a numpy array of shape (4,3) - 4 vertices (courners) each with x,y,z coordinates

    NB: ew = RA
        ns = Dec
        l = lon = longitude = RA
        b = lat = latitude = Dec
        """
    # Get latitude/longitude arrays in radians
    # (NB this isn't tecnically latitude/longitude, but it's what spherical coordinate formulae use)
    lon = coords.ra.to('radian').value
    lat = coords.dec.to('radian').value

    # Poles???
    poles = {}
    poles['w'] = lon - (np.pi / 2), 0 * lat
    poles['e'] = lon + (np.pi / 2), 0 * lat
    mask = lat < 0
    poles['s'] = lon + 0, lat - (np.pi / 2)
    poles['s'][0][mask], poles['s'][1][mask] = lon[mask] + np.pi, -lat[mask] - (np.pi / 2)

    poles['n'] = lon + np.pi, (np.pi / 2) - lat
    poles['n'][0][mask], poles['n'][1][mask] = lon[mask], lat[mask] + (np.pi / 2)

    poles['w'] = spherical_to_cartesian(*poles['w'])
    poles['e'] = spherical_to_cartesian(*poles['e'])
    poles['n'] = spherical_to_cartesian(*poles['n'])
    poles['s'] = spherical_to_cartesian(*poles['s'])


    # Get phi angles in radians
    # (NB divided by 2, since angle from centre to edge is half the FoV)
    phi_ra = fov['ra'].to('radian').value / 2
    phi_dec = fov['dec'].to('radian').value / 2

    # Convert lat/lon to cartesian
    xyz = spherical_to_cartesian(lon, lat)

    # Edges
    edges = {}
    fcos, fsin = np.cos(phi_ra), np.sin(phi_ra)

    edges['e'] = xyz * fcos + poles['e'] * fsin
    le, be = cartesian_to_spherical(*edges['e'])

    edges['w'] = xyz * fcos + poles['w'] * fsin
    lw, bw = cartesian_to_spherical(*edges['w'])

    ls, bs = lon, lat-phi_dec
    edges['s'] = spherical_to_cartesian(ls, bs)

    ln, bn = lon, lat+phi_dec
    edges['n'] = spherical_to_cartesian(ln, bn)

    # Something
    for key in edges.keys():
        edges[key] = edges[key].T
    for key in poles.keys():
        poles[key] = poles[key].T

    # Corners
    corners = []
    corners.append(intersect(edges['n'], poles['w'], edges['w'], poles['n']))
    corners.append(intersect(edges['n'], poles['e'], edges['e'], poles['n']))
    corners.append(intersect(edges['s'], poles['e'], edges['e'], poles['s']))
    corners.append(intersect(edges['s'], poles['w'], edges['w'], poles['s']))

    corners = np.asarray(corners)
    corners = np.rollaxis(corners, 0, 2)
    return corners


def get_tile_vertices_astropy(centre, fov):
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


def get_tile_edges(vertices, steps=5):
    """Interpolate a tile with corners to a full shape to be drawn.

    Parameters
    ----------
    vertices : `numpy.ndarray`
        An array of shape (4,3) defining 4 vertices in cartesian coordinates.
    """
    all_points = []
    # Loop through all four edges by rolling through the array of vertices and taking each pair
    for corner1, corner2 in zip(vertices, np.roll(vertices, 1, axis=0)):
        points = interpolate(corner2, corner1, steps)  # Note the order is flipped
        all_points.extend(points[:-1])  # We remove the final point, since it's duplicated
    return np.array(all_points)


def get_tile_edges_astropy(centre, fov, edge_points=5):
    """Get points along the edges of the tile.

    Parameters
    ----------
    centre : `astropy.coordinates.SkyCoord`
        The coordinates of the tile centre.
    fov : dict
        The field of view of the tile, with keys of 'ra' and 'dec'
    edge_points : int, edge_points=5
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
    corners = get_tile_vertices_astropy(centre, fov)

    # If edge_points=0 then just return the corners
    if edge_points == 0:
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


class PolygonQuery(object):
    def __init__(self, nside):
        self.nside = nside
    def __call__(self, vertices):
        # Note nest is always True
        # See https://github.com/GOTO-OBS/goto-tile/issues/65
        return hp.query_polygon(self.nside, vertices, nest=True, inclusive=True, fact=32)


def get_tile_pixels(vertices, nside):
    """Find the HEALPix pixels within the given vertices.

    Parameters
    ----------
    tile_vertices : `numpy.ndarray`
        A 1D array containing arrays of shape (4,3) defining 4 vertices in cartesian coordinates,
        for each tile.

    nside : float
        The HEALPix Nside resolution parameter.
    """
    polygon_query = PolygonQuery(nside)

    pool = multiprocessing.Pool()
    pixels = pool.map(polygon_query, vertices)
    pool.close()
    pool.join()

    return np.array(pixels)


def get_tile_pixels_astropy(tile_edges, nside=256, order='NESTED', inclusive=True):
    """Find the HEALPix pixels within the given vertices.

    Parameters
    ----------
    tile_edges : `astropy.coordinates.SkyCoord`
        Coordinates describing the edges of each tile
    nside : float, default=256
        The HEALPix Nside resolution parameter
    order : string, default='NESTED'
        The HEALPix ordering scheme to use
    inclusive : bool, default=True
        See `healpy.query_polygon`

    Returns
    -------
    ipix : list of `numpy.array`
        The indices of the pixels contained within each tile.
        Note that at high resolutions tiles might contain different numbers of pixels,
        which is why this is not necessarily a 2D array.

    """
    xyz_points = tile_edges.cartesian.get_xyz(xyz_axis=2).value

    # Note nest is always True, see https://github.com/GOTO-OBS/goto-tile/issues/65
    ipix = [hp.query_polygon(nside, p, nest=True, inclusive=inclusive, fact=32) for p in xyz_points]
    if order == 'RING':
        ipix = [np.array(sorted(hp.nest2ring(nside, ip))) for ip in ipix]

    return ipix
