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
            self.gal = pd.DataFrame(columns=['level', 'nstar'])

        self.gal['bhid'] = -1; self.gal['msink'] = -1
