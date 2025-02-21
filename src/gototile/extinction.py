"""Module to create a skymap of weighted galactic extinction."""

from pathlib import Path

import numpy as np
import pkg_resources

from .skymap import SkyMap


def create_extinction_skymap(min_weight=0, exp_k=5):
    """Create a skymap of weighted galactic extinction.

    Parameters
    ----------
    min_weight : float, optional
        minimum weight to scale the skymap
        default is 0

    exp_k: int
        exponential constant for weight skymap

    Returns
    -------
    skymap : `gototile.skymap.SkyMap`
        the data in a SkyMap class

    """
    # Find the extinction file path
    data_path = Path(pkg_resources.resource_filename('gototile', 'data'))
    filename = 'extinction.fits'
    filepath = data_path / filename

    # Create the SkyMap
    skymap = SkyMap.from_fits(filepath)

    # Scale skymap data between `min_weight` and 1
    data = skymap.data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Weight and invert the data using an exponential
    weighted_data = np.exp(-exp_k * data)

    # Force minimum weight
    weighted_data = (1 - min_weight) * weighted_data + min_weight

    # Update the SkyMap class
    skymap._save_data(weighted_data, skymap.order, skymap.coordsys)  # noqa: SLF001

    return skymap
