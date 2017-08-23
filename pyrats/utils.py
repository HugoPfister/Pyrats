"""Some utils to read Ramses and RamsesRT outputs

TODO:
    - Start with a better description
    - Lots and lots of things

"""
from scipy.io import FortranFile as FF
import numpy as np

# Classes stuff
class ImplementError(Exception):
    """Custom class for 'not implemented yet' errors"""
    # TODO: replace with NotImplementedError

    def __init__(self):
        Exception.__init__(self, 'Not implemented yet')

# Useful functions


def read_header(ds):
    """Read a RAMSES output header file."""
    import re

    folder = './'
    iout = int(str(ds)[-5:])

    hname = '{0}/output_{1:05d}/header_{1:05d}.txt'.format(folder, iout)
    with open(hname, 'r') as hfile:
        htxt = hfile.read()
    header_re = re.compile("Total number of particles\s+([0-9]+)\s+"
                           "Total number of dark matter particles\s+([0-9]+)\s+"
                           "Total number of star particles\s+([0-9]+)\s+"
                           "Total number of sink particles\s+([0-9]+)\s*"
                           "Particle fields\s+([\w ]+)")
    match = header_re.search(htxt.strip())
    header = {'total': 0, 'DM': 0, 'stars': 0, 'sinks': 0, 'fields': []}
    if match:
        ntot, ndm, nstar, nsink, fields = match.groups()
        header['total'] = int(ntot)
        header['DM'] = int(ndm)
        header['stars'] = int(nstar)
        header['sinks'] = int(nsink)
        # Format fields
        if 'tform' in fields:
            fields = fields.replace('tform', 'epoch')
        if 'metal' in fields:
            fields = fields.replace('metal', 'metals')
        if 'iord' in fields:
            fields = fields.replace('iord', 'id')
        header['fields'] = fields.split()

    return header


def read_cooling(ds):
    """Read the cooling table from the cooling.out file."""

    folder = '.'
    iout = int(str(ds)[-5:])

    fname = '{F}/output_{I:05d}/cooling_{I:05d}.out'.format(F=folder,
                                                            I=iout)
    cool = dict()
    with FF(fname, 'r') as cf:
        n1, n2 = cf.read_ints()
        cool['n_nH'] = n1
        cool['n_T2'] = n2
        cool['nH'] = cf.read_reals(np.float64)
        cool['T2'] = cf.read_reals(np.float64)
        cool['cooling'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['heating'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['cooling_com'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['heating_com'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['metal'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['cooling_prime'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['heating_prime'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['cooling_com_prime'] = cf.read_reals(
            np.float64).reshape((n1, n2), order='F')
        cool['heating_com_prime'] = cf.read_reals(
            np.float64).reshape((n1, n2), order='F')
        cool['metal_prime'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['mu'] = cf.read_reals(np.float64).reshape((n1, n2), order='F')
        cool['spec'] = cf.read_reals(np.float64).reshape((n1, n2, 6), order='F')

    return cool
