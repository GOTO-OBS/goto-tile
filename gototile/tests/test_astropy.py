"""Testing that the new astropy functions match the output of the old ones."""

from astropy import units as u
from astropy.coordinates import SkyCoord

from gototile import gridtools

import numpy as np


test_coords = SkyCoord(np.arange(360, step=2) * u.deg, np.arange(-90, 90) * u.deg)
fov = {'ra': 3 * u.deg, 'dec': 4 * u.deg}


def test_vertices():
    """Test the functions to get tile vertices."""
    v_old = gridtools.get_tile_vertices(test_coords, fov)
    v_new = gridtools.get_tile_vertices_astropy(test_coords, fov)

    # v_old is already in cartesian but is not normalised, v_new needs to be converted
    v_old = np.array([[cc / np.sqrt(sum(x**2 for x in cc)) for cc in c] for c in v_old])
    v_new = v_new.cartesian.get_xyz(xyz_axis=2).value

    # compare, to within given dp
    dp = 13
    assert np.all(np.round(v_old, dp) == np.round(v_new, dp))


def test_pixels():
    """Test the functions to get HEALPix pixels within the tiles."""
    for nside in [2, 4, 8, 16, 32, 64, 128, 256]:
        v_old = gridtools.get_tile_vertices(test_coords, fov)
        p_old = gridtools.get_tile_pixels(v_old, nside)
        v_new = gridtools.get_tile_vertices_astropy(test_coords, fov)
        p_new = gridtools.get_tile_pixels_astropy(v_new, nside)

        # compare
        assert np.all([set(o) == set(n) for o, n in zip(p_old, p_new)])
