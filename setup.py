#! /usr/bin/env python

import os
import subprocess
from setuptools import setup
from distutils.command.install import INSTALL_SCHEMES
from distutils.command.build_py import build_py as _build_py
from distutils.command.sdist import sdist as _sdist


# Not the best option to hack the distutils installation schemes, but
# installing data files wth distutils are a pain, in particular if you
# want them to be installed inside the package (i.e., next to
# __init__.py).
# Below, we set the data installation scheme to that of the library /
# package installation scheme
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']


with open('VERSION') as infile:
    VERSION = infile.read().strip()
VERSION_MODULE = 'gototile/version.py'


# Routines for manipulating the git hash and version number during
# installation
def add_git_hash():
    """Obtain and add the git hash to the version file"""
    versionlines = None
    try:
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        git_hash = output.decode('ascii').strip()  # strip newline
        with open(VERSION_MODULE, 'r+') as fp:
            versionlines = fp.readlines()
            fp.seek(0)
            # Alter a *copy* of versionlines; we return the original
            for line in versionlines[:]:
                if line.startswith('__git_hash__'):
                    line = "__git_hash__ = '{}'\n".format(git_hash)
                if line.startswith('__version__'):
                    line = "__version__ = '{}'\n".format(VERSION)
                fp.write(line)
    except subprocess.CalledProcessError: # No git or git repository found
        pass
    return versionlines


def reset_version_file(versionlines):
    """Restore the version file back to its original form"""
    if not versionlines:
        return
    with open(VERSION_MODULE, 'w') as fp:
        for line in versionlines:
            fp.write(line)



# Override the build and sdist classes to dynamically add the current
# git hash
class build_py(_build_py):
    def run(self):
        versionlines = add_git_hash()
        _build_py.run(self)
        reset_version_file(versionlines)


class sdist(_sdist):
    def run(self):
        versionlines = add_git_hash()
        _sdist.run(self)
        reset_version_file(versionlines)


setup(
    name="gototile",
    version=VERSION,
    description=("Create a set of tiled observation pointings to "
                 "cover an extended sky area, for one or more telescopes"),
    author="Darren White, Evert Rol",
    author_email="Darren.White@warwick.ac.uk, evert.rol@monash.edu",
    url="http://goto-observatory.org",
    scripts=["scripts/gototile", "scripts/create-grid"],
    packages=["gototile"],
    package_dir={'gototile': "gototile"},
    install_requires=["numpy", "astropy", "matplotlib", "healpy",
                      "pyephem", "basemap"],
    data_files=[('gototile', ["gototile/GWGC.csv",
                              "gototile/cylon.csv"]),],
    cmdclass={'build_py': build_py,
              'sdist': sdist},
)
