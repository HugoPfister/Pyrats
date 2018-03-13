import yt
from yt.utilities.logger import ytLogger as mylog
import numpy as np
import pandas as pd
import os as os

from . import halos, fields, visualization, utils, physics, sink, analysis, load_snap, galaxies


def load(files='', stars=False, dm=False, MatchObjects=False, bbox=None, haloID=None):
    """
    Load a RAMSES output, with options to filter stars, DM, BHs, or
    halos (from HaloFinder)
    * files: output/info from ramses
    * stars (False): if True, then add a filter to select star particles
    * dm (False): if True, then add a filter to select dm particles
    * halo (False): if True, load halos, tree_brick must be in ./Halos/ID output/tree_brick and
      computed with HaloFinder
    """

    if type(files) == int:
        files = 'output_{files:05}/info_{files:05}.txt'.format(files=files)
   
    yt.funcs.mylog.setLevel(40) 
    ds = yt.load(files)
    yt.funcs.mylog.setLevel(20)
    ids = int(str(ds).split('_')[1])
   
    #read csv file for sinks 
    mylog.info('Reading sinks')
    sinks = sink.get_sinks(ds)

    #if the location of tree_brick is specified instead of the structure
    #Halos/output_number/tree_brick
    #if type(halo) == str:
    #    # Remove trailing '/'
    #    if halo[-1] == '/':
    #        halo = halo[:-1]
    #    hp = os.path.split(halo)[0]
    #    p = os.path.join(halo, str(ids), 'tree_bricks%.3i' % ids)
    #else:
    hp = './'
    p = os.path.join('Halos', str(ids), 'tree_bricks%.3i' % ids)
    halo_ok = os.path.exists(p)
    if not halo_ok:
        mylog.warning('Halo flag is set yet we could not find any'
                      ' Halo directory. Tried %s' % p)
    
    #load halos and galaxies
    mylog.info('Reading halos and galaxies')
    halo = halos.HaloList(ds, folder=hp, contam=False)
    gal = galaxies.GalList(ds, folder=hp, contam=False)

    sinks['hid'] = -1 ; sinks['galID'] = -1
    sinks['mgal'] = 0 ; sinks['mbulge'] = 0 ; sinks['sigma_bulge'] = 0
    halo.halos['bhid'] = -1 ; halo.halos['galID'] = -1
    gal.gal['bhid'] = -1 ; gal.gal['hid'] = -1
    gal.gal['msink'] = 0 ; gal.gal['mhalo'] = 0
    if MatchObjects:
        L = ds.length_unit.in_units('Mpc')
        mylog.info('Matching sinks to haloes and galaxies')
        #match sinks to haloes and galaxies
        for bhid in sinks.ID:
            bh = sinks[sinks.ID == bhid]
            #haloes
            hid = ((halo.halos.x - bh.x.item())**2 + (halo.halos.y - bh.y.item())** 2 + (halo.halos.z - bh.z.item())**2).argmin()
            d = np.sqrt(((halo.halos.x - bh.x.item())**2 + (halo.halos.y - bh.y.item())** 2 + (halo.halos.z - bh.z.item())**2)[hid])
            if d * L < 0.05 * halo.halos.rvir[hid]:
                oldID = int(halo.halos.bhid[hid])
                if oldID == -1:
                    halo.halos.loc[hid, 'bhid'] = bhid
                else:
                    bhold = sinks.loc[sinks.ID == oldID]
                    oldm = bhold.M.item()
                    if bh.M.item() > oldm:
                        halo.halos.loc[hid, 'bhid'] = bhid
            
            if d * L < halo.halos.rvir[hid]:
                sinks.loc[sinks.ID == bhid, 'hid'] = hid
            
            #Galaxies
            hid = ((gal.gal.x - bh.x.item())**2 + (gal.gal.y - bh.y.item())** 2 + (gal.gal.z - bh.z.item())**2).argmin()
            d = np.sqrt(((gal.gal.x - bh.x.item())**2 + (gal.gal.y - bh.y.item())** 2 + (gal.gal.z - bh.z.item())**2)[hid])
            if d * L < gal.gal.rvir[hid]:
                oldID = int(gal.gal.bhid[hid])
                if oldID == -1:
                    gal.gal.loc[hid, 'bhid'] = bhid
                    gal.gal.loc[hid, 'msink'] = bh.M.item()
                else:
                    oldm = gal.gal.msink[hid].item()
                    if bh.M.item() > oldm:
                        gal.gal.loc[hid, 'bhid'] = bhid
                        gal.gal.loc[hid, 'msink'] = bh.M.item()
                sinks.loc[sinks.ID == bhid, 'galID'] = hid
                sinks.loc[sinks.ID == bhid, 'mgal'] = gal.gal.m[hid].item()
                sinks.loc[sinks.ID == bhid, 'mbulge'] = gal.gal.mbulge[hid].item()
                sinks.loc[sinks.ID == bhid, 'sigma_bulge'] = gal.gal.sigma_bulge[hid].item()
            
        mylog.info('Matching galaxies to haloes')
        #match galaxies to haloes
        for galID in gal.gal.index:
            g=gal.gal.loc[galID]
            hid = ((halo.halos.x - g.x.item())**2 + (halo.halos.y - g.y.item())** 2 + (halo.halos.z - g.z.item())**2).argmin()
            d = np.sqrt(((halo.halos.x - g.x.item())**2 + (halo.halos.y - g.y.item())** 2 + (halo.halos.z - g.z.item())**2)[hid])
            if d * L < 0.1 * halo.halos.rvir[hid]:
                oldID = int(halo.halos.galID[hid])
                if oldID == -1:
                    halo.halos.loc[hid, 'galID'] = galID
                else:
                    oldm = gal.gal.m[oldID].item()
                    if g.m.item() > oldm:
                        halo.halos.loc[hid, 'galID'] = galID
            
            if d * L < halo.halos.rvir[hid]:
                gal.gal.loc[galID, 'hid'] = hid
                gal.gal.loc[galID, 'mhalo'] = halo.halos.m[hid].item()
    
    if (haloID != None):
        h=halo.halos.loc[haloID]
        center=np.copy([h.x,h.y,h.z])
        w=2*h.rvir/float(ds.length_unit.in_units('Mpc'))
        bbox=[center-w, center+w] 
    
    if (stars or dm):
        ds = yt.load(files, extra_particle_fields=[("particle_birth_time", "d"),("particle_metallicity", "d")], bbox=bbox)
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
