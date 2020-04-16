import yt
from yt.utilities.logger import ytLogger as mylog
import numpy as np
import os as os
import numbers
from glob import glob
import pandas as pd
from tqdm import tqdm

from . import fields, sink, galaxies, utils

def load(files='',
         haloID=None, Galaxy=False, bhID=None, 
         radius=None, bbox=None,
         MatchObjects=False, fvir=[1,'r90'], contam=False,
         old_ramses=False, prefix='./', verbose=True):
    """
    Loads a ramses output

    Note
    ----
    by default the outputs must be in a folder Outputs

    Parameters
    ----------
    prefix (str) : path to the outputs
    files : int or path
       Path or output number to load.
    haloID : int, optional
       ID of the halo (or galaxy) to zoom on.
    Galaxy : logical, optional
       If true, interpret haloID as the id of a galaxy. Default: False
    bhID : int, optional
       ID of the black hole to zoom on.
    radius : tuple, optional
       Radius in the form of (value, unit).
    bbox : array like
       Bounding box of the region to load in code_unit, in the form of
       [[left, bottom], [right, top]].
    MathObjects : logical, optional
       If True, match BH to galaxies
    fvir : 3-tuple, optional
       Fraction of the virial radius to look at when matching objects.
         fvir[0] -> galaxies to halos
         fvir[1] -> sinks to halos
         fvir[2] -> sinks to galaxies
    old_ramses : load old ramses 
    prefix : str, optional
       Set this to the relative path to the root folder containing all
       the outputs.
    contam (logical) : required by the halo finder

    """

    if isinstance(files, numbers.Number):
        if not os.path.exists(prefix+'/Outputs'):
            mylog.info('Put all your outputs in a folder Outputs')
        if files == -1:
            files = os.path.join(prefix,'Outputs')
            files = utils.find_outputs(path=files)[-1]
        else:
            files = os.path.join(prefix,'Outputs', 'output_{files:05d}', 'info_{files:05d}.txt')\
              .format(files=files)

    if not verbose:
        yt.funcs.mylog.setLevel(40)
    else:
        yt.funcs.mylog.setLevel(20)

    ds = yt.load(files)

    ids = int(str(ds).split('_')[1])
    ds.ids = ids
    ds.prefix = prefix
    files = os.path.join(prefix,'Outputs', 'output_{:05}'.format(ds.ids))
    ds.files = files

    # read csv file for sinks
    mylog.info('Reading sinks')
    sinks = sink.get_sinks(ds)

    # load halos and galaxies
    mylog.info('Reading halos and galaxies')
    gal = galaxies.GalList(ds, contam=contam, prefix=ds.prefix)

    ds.sink = sinks
    ds.gal = gal
    if MatchObjects: matching(ds, fvir)

    # Load only the relevant part of the simulation
    if haloID is not None:
        if Galaxy:
            ext = 'star'
            mylog.info('Loading around galaxy {}'.format(haloID))
        else:
            ext = 'DM'
            mylog.info('Loading around halo {}'.format(haloID))

        h = ds.gal.gal.loc[haloID]
        center = h[[_+ext for _ in ['x','y','z']]].values

        if type(radius) in (float, int):
            Nrvir = radius
            mylog.info('size of the region {} Mpc'.format(Nvir*h.r))
            w = Nrvir*h.r/float(ds.length_unit.in_units('Mpc'))
        elif radius is not None:
            w = ds.quan(radius[0]*2, radius[1]).to('code_length').value
            mylog.info('size of the region {} {}'.format(radius[0], radius[1]))
        else:
            mylog.info('Be more specific for the size of the region...')
        bbox = [center-w, center+w]

    if bhID is not None:
        h = ds.sink.loc[ds.sink.ID == bhID]
        center = h[[_ for _ in ['x','y','z']]].values[0]
        if radius is None:
            mylog.info('Please specify a radius, e.g. (10, \'kpc\') for the region')
        else:
            w = float(ds.arr(radius[0], radius[1]).in_units('code_length'))
        bbox = [center-w, center+w]
   
    ###########################
    #read old ramses format (to be removed at some point)
    #may be broken as not updated anymore....
    if old_ramses:
        yt.funcs.mylog.setLevel(40)
        ds = yt.load(files+'/info_{:05}.txt'.format(ds.ids),
                extra_particle_fields=[("particle_birth_time", "d"),
                ("particle_metallicity", "d")], bbox=bbox)
        mylog.info('Filtering stars')
        yt.add_particle_filter("star", function=fields._star,
                               filtered_type="io")
        ds.add_particle_filter("star")
        mylog.info('Filtering dark matter')
        yt.add_particle_filter("DM", function=fields._DM,
                               filtered_type="io")
        ds.add_particle_filter("DM")
        ds.sink = sinks
        ds.gal = gal
        if MatchObjects: matching(ds, fvir)
        yt.funcs.mylog.setLevel(20)
    ###########################
    else:
        ds._bbox = bbox 

    return ds

def get_sphere(ds, width, bhid=None, hnum=None, Galaxy=None):
    '''
    Create directly a sphere around a BH/halo/galaxy
    '''
    yt.funcs.mylog.setLevel(40)
    if ((hnum is not None) and (bhid is not None)):
        raise AttributeError('Please specify only hnum or bhid but not both')

    if bhid is not None:
        h = ds.sink.loc[ds.sink.ID == bhid]
        c = [h.x.item(), h.y.item(), h.z.item()]

    if hnum is not None:
        if Galaxy:
            h = ds.gal.gal.loc[hnum]
        else:
            h = ds.halo.halos.loc[hnum]
        c = [h.x.item(), h.y.item(), h.z.item()]
    sp = ds.sphere(c, width)
    return sp

def matching(ds, fvir):
        path = os.path.join(ds.prefix,'matching/{}_{}star'.format(fvir[0], fvir[1]))
        if os.path.exists(path):
            mylog.info('Using {}_{}star for matching'.format(fvir[0],fvir[1]))
        else:
            mylog.info('{}star not found'.format(path))
        for component in ['gal', 'sinks']:
            path_dummy = os.path.join(path, component, str(ds.ids))
            if os.path.exists(path_dummy):
                dummy = pd.read_hdf(path_dummy)
                for c in dummy.columns:
                    if component == 'gal':
                        ds.gal.gal[c] = dummy[c]
                    if component == 'sinks':
                        ds.sink[c] = dummy[c]
        return ds
