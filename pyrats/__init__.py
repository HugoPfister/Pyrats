__all__ = ['halos', 'utils', 'trees', 'visualization',
           'fields', 'physics', 'snaplist', 'sink', 'analysis']

import yt
from yt.utilities.logger import ytLogger as mylog
import numpy as np
import pandas as pd
import os as os

from . import halos, fields, visualization, utils, physics, sink, analysis


def load(files='', stars=False, dm=False, bh=False, halo=False):
    """
    Load a RAMSES output, with options to filter stars, DM, BHs, or
    halos (from HaloFinder)
    * files: output/info from ramses
    * stars (False): if True, then add a filter to select star particles
    * dm (False): if True, then add a filter to select dm particles
    * bh (False): if True, load BHs
    * halo (False): if True, load halos, tree_brick must be in ./Halos/ID output/tree_brick and
      computed with HaloFinder
    """

    if type(files) == int:
        files = 'output_{files:05}/info_{files:05}.txt'.format(files=files)

    ds = yt.load(files)
    ids = int(str(ds).split('_')[1])

    if stars:
        mylog.info('Filtering stars')
        yt.add_particle_filter("stars", function=fields.stars,
                               filtered_type="all", requires=["particle_age"])
        ds.add_particle_filter("stars")

    if dm:
        mylog.info('Filtering dark matter')
        yt.add_particle_filter("dm", function=fields.dm, filtered_type="all")
        ds.add_particle_filter("dm")

    if bh:
        mylog.info('Reading sinks')
        ds.sink = sink.get_sinks(ds)

    halo_ok = False
    if halo:
        if type(halo) == str:
            # Remove trailing '/'
            if halo[-1] == '/':
                halo = halo[:-1]
            hp = os.path.split(halo)[0]
            p = os.path.join(halo, str(ids), 'tree_bricks%.3i' % ids)
        else:
            hp = './'
            p = os.path.join('Halos', str(ids), 'tree_bricks%.3i' % ids)

        halo_ok = os.path.exists(p)
        if not halo_ok:
            mylog.warning('Halo flag is set yet we could not find any'
                          ' Halo directory. Tried %s' % p)

    if halo_ok:
        mylog.info('Reading halos')
        ds.halo = halos.HaloList(ds, folder=hp, contam=False)
        if os.path.exists('./Galaxies/GalProp' + str(str(ds)[-6:]) + '.csv'):
            columns = ['pollution', 'mgal', 'sigma', 'dmdt1_1',
                       'dmdt10_1', 'dmdt50_1', 'dmdt1_10',
                       'dmdt10_10', 'dmdt50_10']
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

    if halo_ok and bh:
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
                oldID = int(ds.halo.loc[ds.halo.halos.index == hid, 'bhid'])
                if oldID == -1:
                    ds.halo.loc[ds.halo.halos.index == hid, 'bhid'] = bhid
                else:
                    bhold = ds.sink[ds.sink.ID == oldID]
                    oldm = float(bhold.M)
                    if float(bh.M) > oldm:
                        ds.halo.loc[ds.halo.halos.index == hid, 'bhid'] = bhid

                ds.sink.loc[ds.sink.ID == bhid, 'hid'] = float(
                    ds.halo.halos.index[hid-1])

    return ds
