# This script converts files from the HaloFinder
# into hdf5 readable files.

# Not all quantities are dumped, should be modified depending on what is needed
# and the version of the HaloFinder

import numpy as np
import pandas as pd
import yt.utilities.fortran_utils as fpu
from yt.utilities.logger import ytLogger as mylog
from yt.funcs import get_pbar
import yt
import os


class GalList(object):
    def __init__(self, ds, contam=False, prefix=''):
        self.iout = ds.ids
        self.ds = ds
        self.prefix = prefix
        filename = self.prefix+'/Structures/hdf5/tree_bricks{:03}.hdf'.format(self.iout)
        if os.path.exists(filename):
            self.gal = pd.read_hdf(filename)
        else:
            mylog.info('Did not find {}'.format(filename))
            mylog.info('Use Struc_To_hdf5 to convert to hdf5 file')
            mylog.info('Keep going anyway')
            self.gal = pd.DataFrame(columns=['level', 'nstar','mstar'])

        self.gal['bhid'] = -1; self.gal['msink'] = -1; self.gal['BH_dist'] = -1


    def read_part(self, igal):
        '''
        Read the particle files, output in this order:
            r,v,m,ID,age,z
        '''
        filename = self.prefix+'/Structures/AdaptaHOP/GAL_{:05}/gal_stars_{:07}'.format(
            self.iout, igal)
        if not os.path.exists(filename):
            mylog.info('Could not find {}'.format(filename))
            mylog.info('End of reading.')
            return
        with open(filename, 'rb') as f:
            igal_file = fpu.read_vector(f, 'i')
            if igal_file != igal:
                mylog.info('Error, file galaxy is {}'.format(igal_file))
            fpu.read_vector(f, 'i') #level
            fpu.read_vector(f, 'd') #mgal
            rgal = fpu.read_vector(f, 'd') #xgal
            vgal = fpu.read_vector(f, 'd') #vgal
            fpu.read_vector(f, 'd') #Lgal
            fpu.read_vector(f, 'q') #nstar
            r = np.array([fpu.read_vector(f, 'd') for _ in range(3)])
            v = np.array([fpu.read_vector(f, 'd') for _ in range(3)])
            m = fpu.read_vector(f, 'd') 
            ID = fpu.read_vector(f, 'q') 
            age = fpu.read_vector(f, 'd') 
            z = fpu.read_vector(f, 'd') 
            
            r = r.T - rgal
            v = v.T - vgal

        return r,v,m,ID,age,z



