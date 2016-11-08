import os
import tempfile
import inspect
import gototile.grid
import gototile.telescope
from gototile.skymaptools import calculate_tiling
from gototile.settings import NSIDE



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


def create_grid(telescopes=None, overlap=None, nside=NSIDE):
    """Create the GOTO tiled pointing map, one per telescope"""

    if telescopes is None:
        telescopes = []

    grid = {}
    for telescope in telescopes:
        grid[telescope] = gototile.grid.tileallsky(
            telescope.fov, nside, overlap=overlap, gridcoords=None, nested=True)

    return grid


def makegrid(tilesdir, name=None):
    if not os.path.exists(tilesdir):
        os.makedirs(tilesdir)
    fp = tempfile.NamedTemporaryFile(prefix='temp__', dir=tilesdir,
                                     delete=False)
    path = fp.name
    fp.close()
    gototile.grid.tileallsky(path, name, NSIDE)
    return path
