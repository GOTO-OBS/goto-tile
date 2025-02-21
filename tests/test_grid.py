"""Test the grid module."""

import numpy as np
from astropy import units as u

from gototile.grid import SkyGrid


def test_gridding():
    """Test the SkyGrid class."""
    fov = {'ra': 4 * u.deg, 'dec': 2 * u.deg}

    # 512 requires more memory, but has higher precision. Some
    # assertions fails with lower precision, because e.g., the maximum
    # tile comes out as containing more than 9 square degrees
    nside = 512
    for overlap in [0.0, 0.5, 0.8]:
        grid = SkyGrid(fov, overlap=overlap)
        tile_pixels = grid.get_tile_pixels(nside)

        assert len(tile_pixels) == grid.ntiles  # noqa: S101
        tile_pixcount = np.array([len(pixlist) for pixlist in tile_pixels])
        tile_areas = tile_pixcount / (12 * nside**2) * 41253
        if nside >= 512:
            assert np.min(tile_areas) > 7  # noqa: S101
            assert np.max(tile_areas) < 9  # noqa: S101
        assert np.abs(np.mean(tile_areas) - 8) < 0.1  # noqa: S101
