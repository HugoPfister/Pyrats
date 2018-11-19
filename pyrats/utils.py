"""Some utils to read Ramses and RamsesRT outputs

TODO:
    - Start with a better description
    - Lots and lots of things

"""
from scipy.io import FortranFile as FF
import numpy as np
from glob import glob
import os
import yt
from .import sink, trees

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


def find_outputs(path='.'):
    pattern = os.path.join(path, 'output_?????')

    outputs = []
    for d in sorted(glob(pattern)):
        iout = d.split('_')[-1]
        full_path = os.path.join(d, 'info_%s.txt' % iout)
        if os.path.exists(full_path):
            outputs.append(full_path)

    return outputs

def filter_outputs(snap=[-1], hnum=None, timestep=None, Galaxy=False, bhid=None):
    
    yt.funcs.mylog.setLevel(0)
    files = find_outputs()
    ToPlot = [True] * len(files)
    hid = [None for f in files]

    if snap != [-1]:
        for i in range(len(files)):
            ToPlot[i] = ((i+1) in snap)

    if ((hnum is not None) and (bhid is not None)):
        #print('Please specify only hnum or bhid but not both')
        raise AttributeError('Please specify only hnum or bhid but not both')

    if hnum is not None:
        t = trees.Forest(Galaxy=Galaxy)
        prog = t.get_family(hnum=hnum, timestep=timestep)
        for i in prog.index:
            hid[prog.loc[i].halo_ts.astype(int)-1] = prog.loc[i].halo_num.astype(int)
        for i in range(len(files)):
            ToPlot[i] = (ToPlot[i]) & (i+1 in np.array(prog.halo_ts))

    if bhid is not None:
        s = sink.Sinks(ID=[bhid])
        tform = s.sink[bhid].t.min()
        tmerge = s.sink[bhid].t.max()
        for isnap, f in enumerate(files):
            ds = yt.load(f)
            ToPlot[isnap] = (ToPlot[isnap] &
                             ((ds.current_time >= ds.arr(tform, 'Gyr')) &
                              (ds.current_time <= ds.arr(tmerge, 'Gyr'))))

    return ToPlot, hid

def _get_ncpus():
    '''
    output the number of cpus available for OMP tasks
    '''
    try:
        ncpus = int(os.environ['PBS_NUM_PPN'])
    except KeyError:
        ncpus = 1
    return ncpus

def _get_extension(hnum=None, timestep=None, Galaxy=False, bhid=None, radius=None):
    '''
    when centering around BH/Halo/Gal w/ w/o a given radius
    output is the extension to keep track of the informations
    '''
    if hnum != None:
        if Galaxy:
            name = 'Galaxy{}'.format(hnum)
        else:
            name = 'Halo{}'.format(hnum)
        if timestep != None:
            name += '_output{:05}'.format(timestep)

    if bhid != None:
        name = 'BH{}'.format(bhid)

    if ((type(radius) is int) | (type(radius) is float)):
        name += '_{}dx'.format(radius)
    elif (type(radius) is tuple):
        if ((type(radius[0]) is int) | (type(radius) is float)) & (type(radius[1]) is str):
            name += '_{}{}'.format(radius[0],radius[1])
        else:
            raise TypeError('Please give the radius with the form (10,\'pc\')')
    else:
        raise TypeError('Please give an int or a tuple like (10, \'pc\') for the radius')


    return name

def _mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

