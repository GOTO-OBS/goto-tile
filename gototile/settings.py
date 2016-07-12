import os
from astropy import units
from astropy.time import TimeDelta

NSIDE = 256
TILESDIR = 'tiles'
GWGC_PATH = os.path.join(os.path.dirname(__file__), 'GWGC.csv')
TIMESTEP = TimeDelta(300 * units.second)
SUNALTITUDE = -18 * units.degree
COVERAGE = {'min': 0.05, 'max': 0.95}
ARC_PRECISION = 9000
MINPROB = 1e-6
