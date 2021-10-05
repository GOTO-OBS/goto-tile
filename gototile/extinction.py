import os
import pkg_resources

import numpy as np

from .skymap import SkyMap

def create_extinction_skymap(min_weight=0, exp_k=5):
    """Create a skymap of weighted galactic extinction.

    Parameters
    ------------
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
    data_path = pkg_resources.resource_filename('gototile', 'data')
    filename = 'extinction.fits'

    # Create the SkyMap
    skymap = SkyMap.from_fits(os.path.join(data_path, filename))

    # Scale skymap data between `min_weight` and 1
    data = skymap.data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Weight and invert the data using an exponential
    weighted_data = np.exp(-exp_k*data)

    # Force minimum weight
    weighted_data = (1 - min_weight) * weighted_data + min_weight

    # Update the SkyMap class
    skymap._save_skymap(weighted_data, skymap.order)

    return skymap
