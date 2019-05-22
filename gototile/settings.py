import os
import numpy as np
from astropy import units
from astropy.time import TimeDelta
import pkg_resources

NSIDE = 64
TILESDIR = 'tiles'
DATA_DIR = pkg_resources.resource_filename('gototile', 'data')
GWGC_PATH = os.path.join(DATA_DIR, 'GWGC.csv')
GLADE_PATH = os.path.join(DATA_DIR, 'GLADE.csv')
TIMESTEP = TimeDelta(300 * units.second)
SUNALTITUDE = -18 * units.degree
COVERAGE = {'min': 0.05, 'max': 0.95}
ARC_PRECISION = 9000
MINPROB = 1e-3
DTYPE = np.float64
IDTYPE = np.int64
