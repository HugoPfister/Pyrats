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
         MatchObjects=False, fvir=[1, 90], contam=False,
         old_ramses=False, prefix='./', verbose=True):
    """
    Loads a ramses output

    Parameters
    ----------
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
       If True, match galaxies and BH to halos.
    fvir : 3-tuple, optional
       Fraction of the virial radius to look at when matching objects.
         fvir[0] -> galaxies to halos
         fvir[1] -> sinks to halos
         fvir[2] -> sinks to galaxies
    old_ramses : logical, optional
       DEPRECATED. If true, add a filter to the dataset using the ids
       and age of particles. See note
    prefix : str, optional
       Set this to the relative path to the root folder containing all
       the outputs.

    Note
    ----

    The star and dm filters are now obsolete, as the default behavior
    of yt and RAMSES is to filter based on the particle family. If
    your version of RAMSES is too old, you can still use stars and dm

    """

    if isinstance(files, numbers.Number):
        if files == -1:
            files = os.path.join(prefix,'Outputs')\
              .format(files=files)
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
        path = ds.prefix+'/matching/{}_{}/{}'.format(fvir[0], fvir[1], fvir[2], ds.ids)
        if os.path.exists(path):
            dummy = pd.read_hdf(path+'/gal')
            for c in dummy.columns:
                ds.gal.gal[c] = dummy[c]
            dummy = pd.read_hdf(path+'/sinks')
            for c in dummy.columns:
                ds.sink[c] = dummy[c]

        else: 
            utils._mkdir('./matching/')
            utils._mkdir('./matching/{}_{}_{}'.format(fvir[0],fvir[1],fvir[2]))
            utils._mkdir(path)
            
            L = ds.length_unit.in_units('Mpc')
            mylog.info('Matching galaxies and sinks to haloes')
            #match galaxies and sinks to haloes
            for hid in tqdm(ds.halo.halos.sort_values('level').index):
                h=ds.halo.halos.loc[hid]
                d = np.sqrt((h.x.item() - ds.gal.gal.x)**2 + (h.y.item() - ds.gal.gal.y)** 2 +
                    (h.z.item() - ds.gal.gal.z)**2)
                galID = ds.gal.gal.loc[((d * L) < h.rvir.item()*fvir[0])].index
                haloid = ds.gal.gal.loc[galID].hid.unique()
                haloid = haloid[np.where(haloid != -1)]
                haloid = ds.halo.halos.loc[haloid].galID
                galID = np.setxor1d(galID, haloid)
                if len(galID) != 0:
                    ds.gal.gal.loc[galID, 'mhalo'] = h.m.item()
                    ds.gal.gal.loc[galID, 'hid'] = hid
                    galID = ds.gal.gal.loc[galID].m.idxmax()
                    ds.halo.halos.loc[hid, 'galID'] = galID
                    ds.halo.halos.loc[hid, 'mgal'] = ds.gal.gal.loc[galID].m.item()

                d = np.sqrt((h.x.item() - ds.sink.x)**2 + (h.y.item() - ds.sink.y)** 2 +
                    (h.z.item() - ds.sink.z)**2)
                bhid = ds.sink.loc[((d * L) < h.rvir.item()*fvir[1])].index
                haloid = ds.sink.loc[bhid].hid.unique()
                haloid = haloid[np.where(haloid != -1)]
                haloid = ds.halo.halos.loc[haloid].bhid
                bhid = np.setxor1d(bhid, haloid)
                if len(bhid) != 0:
                    ds.sink.loc[bhid, 'mhalo'] = h.m.item()
                    ds.sink.loc[bhid, 'hid'] = hid
                    bhid = ds.sink.loc[bhid].M.idxmax()
                    bhid = ds.sink.loc[bhid].ID
                    ds.halo.halos.loc[hid, 'bhid'] = bhid
                    ds.halo.halos.loc[hid, 'msink'] = ds.sink.loc[ds.sink.ID == bhid].M.item()

            mylog.info('Matching sinks to galaxies')
            # match sinks to galaxies
            for galID in tqdm(ds.gal.gal.sort_values('level').index):
                g = ds.gal.gal.loc[galID]
                d = np.sqrt((g.x.item() - ds.sink.x)**2 + (g.y.item() - ds.sink.y)** 2 +
                    (g.z.item() - ds.sink.z)**2)
                #bhid = ds.sink.loc[((d * L) < g.r.item()*fvir[2]) & (ds.sink.mgal < g.m.item())].index
                bhid = ds.sink.loc[((d * L) < g.Reff90.item()*fvir[2]) & (ds.sink.mgal < g.m.item())].index
                #bhid = sinks.loc[((d * L) < g.r.item()*fvir[2])].index
                if len(bhid) > 0:
                    ds.sink.loc[bhid, 'mgal'] = g.m.item()
                    ds.sink.loc[bhid, 'galID'] = galID
                    ds.sink.loc[bhid, 'mbulge'] = g.mbulge.item()
                    ds.sink.loc[bhid, 'sigma_bulge'] = g.sigma_bulge.item()
                    bhid = ds.sink.loc[bhid].M.idxmax()
                    bhid = ds.sink.loc[bhid].ID
                    ds.gal.gal.loc[galID, 'bhid'] = bhid
                    ds.gal.gal.loc[galID, 'msink'] = ds.sink.loc[ds.sink.ID == bhid].M.item()
                
            ds.gal.gal[['bhid','hid','msink','mhalo']].to_hdf(
                    path+'/gal', key='hdf5')    
            ds.sink[['hid','mhalo','galID','mgal','mbulge','sigma_bulge']].to_hdf(
                    path+'/sinks', key='hdf5')   

            return
