#! /usr/bin/env python

from distutils.core import setup
from distutils.command.install import INSTALL_SCHEMES
import os.path
import glob


# Not the best option to hack the distutils installation schemes, but
# installing data files wth distutils are a pain, in particular if you
# want them to be installed inside the package (i.e., next to
# __init__.py).
# Below, we set the data installation scheme to that of the library /
# package installation scheme
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(
    name='GOTO-tile',
    version='0.1',
    description=('Create a set of tiled observation pointings '
                 'for GOTO for GW follow-up'),
    author='Darren White',
    author_email='Darren.White@warwick.ac.uk',
    url='http://goto-observatory.org',
    scripts=['scripts/tileskymap','scripts/f2ytile','scripts/postmap'],
    packages=['gototile'],
    package_dir={'gototile': 'gototile'},
    data_files=[('gototile', ['gototile/GWGCCatalog_I.txt',
                              'gototile/cylon.csv']),],
)
