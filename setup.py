#! /usr/bin/env python

from setuptools import setup
from distutils.command.install import INSTALL_SCHEMES


# Not the best option to hack the distutils installation schemes, but
# installing data files wth distutils are a pain, in particular if you
# want them to be installed inside the package (i.e., next to
# __init__.py).
# Below, we set the data installation scheme to that of the library /
# package installation scheme
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(
    name="gototile",
    version="0.2",
    description=("Create a set of tiled observation pointings to "
                 "cover an extended sky area, for one or more telescopes"),
    author="Darren White, Evert Rol",
    author_email="Darren.White@warwick.ac.uk, evert.rol@monash.edu",
    url="http://goto-observatory.org",
    scripts=["scripts/tileskymap", "scripts/create-grid"],
    packages=["gototile"],
    package_dir={'gototile': "gototile"},
    install_requires=["numpy", "astropy", "matplotlib", "healpy",
                      "pyephem", "basemap"],
    data_files=[('gototile', ["gototile/GWGC.csv",
                              "gototile/cylon.csv"]),],
)
