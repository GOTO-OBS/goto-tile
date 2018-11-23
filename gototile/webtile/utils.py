import os
import tempfile
import inspect
import gototile.grid
import gototile.telescope
from gototile.skymaptools import calculate_tiling
from gototile import settings



def save_uploadedfile(uploadedfile):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        for chunk in uploadedfile.chunks():
            fp.write(chunk)
    return fp.name


def get_telescopes():
    """Return the predefined telescopes and their class names"""
    predicate = (lambda item: inspect.isclass(item) and
                 issubclass(item, gototile.telescope.Telescope) and
                 item is not gototile.telescope.Telescope)
    telescopes = dict(inspect.getmembers(gototile.telescope, predicate))
    return telescopes


class TiledMapError(Exception):
    pass


def create_grid(telescopes=None, overlap=None, nside=None):
    """Create the GOTO tiled pointing map, one per telescope"""

    if telescopes is None:
        telescopes = []
    if nside is None:
        nside = getattr(settings, 'NSIDE')

    grid = {}
    for telescope in telescopes:
        grid[telescope] = telescope.get_grid(overlap)

    return grid
