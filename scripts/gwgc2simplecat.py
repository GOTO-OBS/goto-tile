#! /usr/bin/env python

"""Convert the GWGCCatalog_I.txt file to a very basic form that can be
used as a generic catalog.

"""

import sys
import numpy as np
from astropy.table import Table
from astropy.io.ascii import read as read_ascii


def Blum(Bmag):
    return 10**((5.48-Bmag)/2.5)


def Ilum(Imag):
    return 10**((4.04-Imag)/2.5)


def getmass(Ilum, I, B):
    return Ilum * 10**(-0.88+0.6*(B - I))


with open(sys.argv[1]) as infile:
    fullstring = infile.read()
    fullstring = fullstring.replace('~', '')

table = read_ascii(fullstring, header_start=0, data_start=1, delimiter='|')
print('Input table contains:', '  '.join(table.colnames))
table['err_Abs_Mag_I'].dtype = np.float # This column is completely
                                        # empty, resulting in an
                                        # integer type

table['ra'] = 15 * table['RA']
table['dec'] = table['Dec']
table['Blum'] = Blum(table['Abs_Mag_B'])
table['Ilum'] = Ilum(table['Abs_Mag_I'])
table['mass'] = getmass(table['Ilum'], table['Abs_Mag_I'], table['Abs_Mag_B'])
table['weight'] = table['Blum']
print("{} out of {} galaxies have a weight".format(
    (table['weight'].mask == False).sum(), len(table)))


columns = ['PGC', 'Name', 'ra', 'dec', 'weight',
           'Blum', 'Ilum', 'mass',
           'Type',
           'App_Mag_B', 'err_App_Mag_B', 'App_Mag_I', 'err_App_Mag_I',
           'Abs_Mag_B', 'err_Abs_Mag_B', 'Abs_Mag_I', 'err_Abs_Mag_I',
           'Maj_Diam (a)', 'err_Maj_Diam', 'Min_Diam (b)', 'err_Min_Diam',
           'b/a', 'err_b/a', 'PA', 'Dist', 'err_Dist',]
table[columns].write(sys.argv[2], format='csv')

if len(sys.argv) > 3:
    table[columns].write(sys.argv[3], format='votable')
