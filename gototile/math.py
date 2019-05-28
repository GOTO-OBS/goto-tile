"""Some spherical math routines"""

import numpy as np

PI = np.pi
DEG = 180/PI
RAD = PI/180


def cartesian_to_celestial(x, y, z):
    """Convert cartesian coordinates (x,y,z) to celestial (ra,dec)."""
    # First convert cartesian to spherical
    lon, lat = cartesian_to_spherical(x, y, z)

    # Then spherical to celestial
    ra, dec = np.rad2deg(lon), np.rad2deg(lat)

    # Make sure they're within the valid range
    ra[ra < 0] = ra[ra < 0] + 360

    return np.array([ra, dec])


def celestial_to_cartesian(ra, dec):
    """Convert celestial coordinates (ra,dec) to cartesian (x,y,z)."""
    lon, lat = np.deg2rad(ra), np.deg2rad(dec)
    return spherical_to_cartesian(lon, lat)


def spherical_to_cartesian(lon, lat):
    """Convert spherical coordinates (lon,lat) to cartesian (x,y,z)."""
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    """Convert cartesian coordinates (x,y,z) to spherical (lon,lat)."""
    lat = np.arctan2(z, np.sqrt(y ** 2 + x ** 2))
    lon = np.arctan2(y, x)
    return np.array([lon, lat])


def cross(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.empty(x.shape)
    z[...,0] = x[...,1] * y[...,2] - x[...,2] * y[...,1]
    z[...,1] = x[...,2] * y[...,0] - x[...,0] * y[...,2]
    z[...,2] = x[...,0] * y[...,1] - x[...,1] * y[...,0]
    return z


def dot(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x[...,0] * y[...,0] + x[...,1] * y[...,1] + x[...,2] * y[...,2]


def intersect(x1, x2, y1, y2):
    """Returns the intersection of the two lines given by x1-x2 and y1-y2.

    All coordinates should be Cartesian coordinates, and be a (N, 3)
    or a (3,) numpy array, or a 3-element list.

    """
    p = cross(x1, x2)
    q = cross(y1, y2)
    t = cross(p, q)
    mask = dot(cross(p, x1), t) < 0
    sign = np.ones(np.asarray(x1).shape)
    sign[mask, ...] = -1
    t = t * sign
    return t

def interpolate(point1, point2, steps=50):
    """Interpolate along the great circle arc between two points.

    Code from `spherical_geometry.great_circle_arc.interpolate`,
    (https://spacetelescope.github.io/spherical_geometry/),
    which uses Slerp (spherical linear interpolation).

    point1 and point2 should be length 3 (x,y,z) lists or arrays.
    """

    steps = int(max(steps, 2))
    t = np.linspace(0.0, 1.0, steps, endpoint=True).reshape((steps, 1))

    omega = np.arccos(np.clip(dot(point1, point2), -1, 1))
    if omega == 0.0:
        offsets = t
    else:
        sin_omega = np.sin(omega)
        offsets = np.sin(t * omega) / sin_omega

    return offsets[::-1] * point1 + offsets * point2
