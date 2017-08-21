__all__ = ['halos', 'utils', 'trees', 'visualization',
           'fields', 'physics', 'snaplist', 'sink']

import yt
import numpy as np
import pandas as pd
import os as os

from . import *


def load(files='', stars=False, dm=False, bh=False, halo=False):
    """
    Load a RAMSES output, with options to filter stars, DM, BHs, or halos (from HaloFinder)
    files: output/info from ramses
    stars (False): if True, then add a filter to select star particles
    dm (False): if True, then add a filter to select dm particles
    bh (False): if True, load BHs
    halo (False): if True, load halos, they must be in ./Halos and computed with HaloFinder
    """

    if type(files) == int:
        files = 'output_{:05}/info_'.format(files) + '{:05}.txt'.format(files)
    ds = yt.load(files)

    if stars:
        yt.add_particle_filter(
            "stars", function=fields.stars, filtered_type="all", requires=["particle_age"])
        ds.add_particle_filter("stars")

    if dm:
        yt.add_particle_filter("dm", function=fields.dm, filtered_type="all")
        ds.add_particle_filter("dm")

    if bh:
        ds.sink = sink.get_sinks(ds)

    if (halo) & (os.path.exists('./Halos/' + str(int(str(ds)[-5:])) + '/tree_bricks' + str(ds)[-3:])):
        ds.halo = halos.HaloList(ds, contam=False)
        if os.path.exists('./Galaxies/GalProp' + str(str(ds)[-6:]) + '.csv'):
            columns = ['pollution', 'mgal', 'sigma', 'dmdt1_1',
                       'dmdt10_1', 'dmdt50_1', 'dmdt1_10', 'dmdt10_10', 'dmdt50_10']
            tmp = pd.read_csv('./Galaxies/GalProp' +
                              str(str(ds)[-6:]) + '.csv', names=columns)
            tmp.index += 1
            tmp = [ds.halo.halos, tmp]
            ds.halo.halos = pd.concat(tmp, axis=1)
    else:
        halo_keys = ['ID', 'nbpart', 'level', 'min_part_id',
                     'host', 'hostsub', 'nbsub', 'nextsub',
                     'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz',
                     'a', 'b', 'c', 'ek', 'ep', 'et', 'rho0', 'r_c',
                     'spin', 'm', 'r', 'mvir', 'rvir', 'tvir', 'cvel']
        ds.halo.halos = pd.DataFrame(columns=halo_keys)

    if halo & bh:
        L = ds.length_unit.in_units('Mpc')
        ds.sink['hid'] = -1
        ds.halo.halos['bhid'] = -1
        for bhid in ds.sink.ID:
            bh = ds.sink[ds.sink.ID == bhid]
            x = float(bh.x)
            y = float(bh.y)
            z = float(bh.z)
            hid = ((ds.halo.halos.x - x)**2 + (ds.halo.halos.y - y)
                   ** 2 + (ds.halo.halos.z - z)**2).argmin()
            d = np.sqrt(((ds.halo.halos.x - x)**2 + (ds.halo.halos.y - y)
                         ** 2 + (ds.halo.halos.z - z)**2)[hid])
            if d * L < 0.05 * ds.halo.halos.rvir[hid]:
                oldID = int(ds.halo.loc[ds.halo.halos.ID == hid, 'bhid'])
                if oldID == -1:
                    ds.halo.loc[ds.halo.halos.ID == hid, 'bhid'] = bhid
                else:
                    bhold = ds.sink[ds.sink.ID == oldID]
                    oldx = float(bhold.x)
                    oldy = float(bhold.y)
                    oldz = float(bhold.z)
                    oldm = float(bhold.M)
                    if float(bh.M) > oldm:
                        ds.halo.loc[ds.halo.halos.ID == hid, 'bhid'] = bhid

                ds.sink.loc[ds.sink.ID == bhid, 'hid'] = float(
                    ds.halo.halos.ID[hid])

    return ds
