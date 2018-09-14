import yt
from yt.utilities.logger import ytLogger as mylog
import numpy as np
import os as os
from glob import glob

from . import halos, fields, sink, galaxies


def load(files='',
         haloID=None, Galaxy=False, bhID=None, 
         radius=None, bbox=None,
         MatchObjects=False, fvir=[0.1,0.05,0.5],
         stars=False, dm=False, verbose=True):
    """
    Read __init_.load function for infos
    """

    if type(files) in [int, np.int32, np.int64]:
        if files != -1:
            files = 'output_{files:05}/info_{files:05}.txt'.format(files=files)
        else:
            files = glob('output_*/info_*.txt')[-1]

    yt.funcs.mylog.setLevel(40)
    ds = yt.load(files)
    if verbose:
        yt.funcs.mylog.setLevel(20)
    ids = int(str(ds).split('_')[1])

    # read csv file for sinks
    mylog.info('Reading sinks')
    sinks = sink.get_sinks(ds)

    hp = './'
    p = os.path.join('Halos', str(ids), 'tree_bricks%.3i' % ids)
    halo_ok = os.path.exists(p)
    if not halo_ok & ds.cosmological_simulation == 1:
        mylog.warning('Could not find any Halo directory. Tried %s' % p)

    # load halos and galaxies
    mylog.info('Reading halos and galaxies' )
    halo = halos.HaloList(ds, folder=hp, contam=False)
    gal = galaxies.GalList(ds, folder=hp, contam=False)
    halo.halos['pollution'] = 0
    # read purity of halos
    if os.path.exists('./Halos/'+str(ids)+'/contam_halos{:03}'.format(ids)):
        p = np.loadtxt('./Halos/'+str(ids)+'/contam_halos{:03}'.format(ids))
        if len(p) > 0:
            p = p.T
            halo.halos.loc[p[0], 'pollution'] = p[1]/p[2]

    sinks['hid'] = -1 ; sinks['galID'] = -1
    sinks['mgal'] = 0 ; sinks['mbulge'] = 0
    sinks['sigma_bulge'] = 0 ; sinks['mhalo'] = 0
    halo.halos['bhid'] = -1 ; halo.halos['galID'] = -1
    halo.halos['mgal'] = 0 ; halo.halos['msink'] = 0
    gal.gal['bhid'] = -1 ; gal.gal['hid'] = -1
    gal.gal['msink'] = 0 ; gal.gal['mhalo'] = 0
    if MatchObjects:
        L = ds.length_unit.in_units('Mpc')
        mylog.info('Matching galaxies and sinks to haloes')
        #match galaxies and sinks to haloes
        for hid in halo.halos.sort_values('level').index:
            h=halo.halos.loc[hid]
            d = np.sqrt((h.x.item() - gal.gal.x)**2 + (h.y.item() - gal.gal.y)** 2 + (h.z.item() - gal.gal.z)**2)
            galID = gal.gal.loc[((d * L) < h.rvir.item()*fvir[0]) & (gal.gal.mhalo < h.m.item())].index
            if len(galID) != 0:
                gal.gal.loc[galID, 'mhalo'] = h.m.item()
                gal.gal.loc[galID, 'hid'] = hid
                galID = gal.gal.loc[galID].m.idxmax()
                halo.halos.loc[hid, 'galID'] = galID
                halo.halos.loc[hid, 'mgal'] = gal.gal.loc[galID].m.item()

            d = np.sqrt((h.x.item() - sinks.x)**2 + (h.y.item() - sinks.y)** 2 + (h.z.item() - sinks.z)**2)
            bhid = sinks.loc[((d * L) < h.rvir.item()*fvir[1]) & (sinks.mhalo < h.m.item())].index
            if len(bhid != 0):
                sinks.loc[bhid, 'mhalo'] = h.m.item()
                sinks.loc[bhid, 'hid'] = hid
                bhid = sinks.loc[bhid].M.idxmax()
                bhid = sinks.loc[bhid].ID
                halo.halos.loc[hid, 'bhid'] = bhid
                halo.halos.loc[hid, 'msink'] = sinks.loc[sinks.ID == bhid].M.item()

        mylog.info('Matching sinks to galaxies')
        # match sinks to galaxies
        for galID in gal.gal.sort_values('level').index:
            g = gal.gal.loc[galID]
            d = np.sqrt((g.x.item() - sinks.x)**2 + (g.y.item() - sinks.y)** 2 + (g.z.item() - sinks.z)**2)
            #bhid = sinks.loc[((d * L) < g.r.item()*fvir[2]) & (sinks.mgal < g.m.item())].index
            bhid = sinks.loc[((d * L) < g.r.item()*fvir[2])].index
            if len(bhid) > 0:
                sinks.loc[bhid, 'mgal'] = g.m.item()
                sinks.loc[bhid, 'galID'] = galID
                sinks.loc[bhid, 'mbulge'] = g.mbulge.item()
                sinks.loc[bhid, 'sigma_bulge'] = g.sigma_bulge.item()
                bhid = sinks.loc[bhid].M.idxmax()
                bhid = sinks.loc[bhid].ID
                gal.gal.loc[galID, 'bhid'] = bhid
                gal.gal.loc[galID, 'msink'] = sinks.loc[sinks.ID == bhid].M.item()

    # Load only the relevant part of the simulation
    if haloID is not None:
        if Galaxy:
            h = gal.gal.loc[haloID]
        else:
            h = halo.halos.loc[haloID]
        center = np.copy([h.x.item(), h.y.item(), h.z.item()])
        w = 2*h.r/float(ds.length_unit.in_units('Mpc'))
        if radius is not None:
            w = float(ds.arr(radius[0]*2, radius[1]).in_units('code_length'))
        bbox = [center-w, center+w]

    if bhID is not None:
        h = sinks.loc[sinks.ID == bhID]
        center = np.copy([h.x.item(), h.y.item(), h.z.item()])
        if radius is None:
            print('Please specify a radius, i.e. (10, \'kpc\') for the region')
        else:
            w = float(ds.arr(radius[0], radius[1]).in_units('code_length'))
        bbox = [center-w, center+w]

    if stars or dm:
        ds = yt.load(files, extra_particle_fields=[("particle_birth_time", "d"),
                                                   ("particle_metallicity", "d")],
                     bbox=bbox)
    else:
        ds = yt.load(files, bbox=bbox)

    ds.halo = halo
    ds.gal  = gal
    ds.sink = sinks

    if stars:
                mylog.info('Filtering stars')
                yt.add_particle_filter("stars", function=fields.stars,
                                       filtered_type="io")
                ds.add_particle_filter("stars")

    if dm:
                mylog.info('Filtering dark matter')
                yt.add_particle_filter("dm", function=fields.dm,
                                       filtered_type="io")
                ds.add_particle_filter("dm")

    return ds

def get_sphere(ds, bhid, hnum, Galaxy, width):
    '''
    Create directly a sphere around a BH/halo/galaxy
    '''
    if ((hnum is not None) and (bhid is not None)):
        raise AttributeError('Please specify only hnum or bhid but not both')

    if bhid is not None:
        h = ds.sink.loc[ds.sink.ID == bhid]

    if hnum is not None:
        if Galaxy:
            h = ds.gal.gal.loc[hnum]
        else:
            h = ds.halo.halos.loc[hnum]
    
    c = [h.x.item(), h.y.item(), h.z.item()]
    sp = ds.sphere(c, width)
    return sp
