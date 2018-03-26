"""Some spherical math routines"""

import numpy as np


PI2 = np.pi * 2
PI_2 = np.pi / 2
PI = np.pi
DEG = 180/PI
RAD = PI/180


def xyz2radec(x, y, z):
    l, b = xyz2lb(x, y, z)
    return np.array([l*DEG, b*DEG])


def radec2xyz(ra, dec):
    l, b = np.deg2rad(ra), np.deg2rad(dec)
    return lb2xyz(l, b)


def lb2xyz(l, b):
    x = np.cos(l) * np.cos(b)
    y = np.sin(l) * np.cos(b)
    z = np.sin(b)
    return np.array([x, y, z])


def xyz2lb(x, y, z):
    # Using arctan2 for b instead of b = np.arcsin(z), allows for [x,
    # y, z] to be unnormalized
    b = np.arctan2(z, np.sqrt(y*y+x*x))
    l = np.arctan2(y, x)
    return np.array([l, b])


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



def point_in_polygon(point, polygon, center):
    ras = polygon[0,:] * RAD
    decs = polygon[1,:] * RAD
    x, y, z = lb2xyz(ras, decs)
    vertices = np.vstack(lb2xyz(ras, decs)).T
    vertices = np.append(vertices, vertices[0:1,:], axis=0)
    cra, cdec = center.ra.rad, center.dec.rad
    ra, dec = point.ra.rad, point.dec.rad
    poly = polygon.copy()
    poly[0,:] += 360
    #print(poly)
    #print(point.ra.degree, point.dec.degree)
    c = lb2xyz(cra, cdec)
    p = lb2xyz(ra, dec)

    frontside = 0
    for v1, v2 in zip(vertices[:-1,:], vertices[1:,:]):
        n = cross(v1, v2)
        frontside += dot(p, n) > 0
        #intersections += intersect(v1, v2, c, p)
    return frontside == 0
