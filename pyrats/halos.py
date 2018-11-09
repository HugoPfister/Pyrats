#!/usr/bin/env python

"""Module to deal with halos, to be used with HaloMaker.

This module is heavily inspired by the set of IDL routines originally
found in the Ramses Analysis ToolSuite (RATS).

TODO: Some more documentation

"""

import numpy as np
import pandas as pd
import yt
from yt.utilities.logger import ytLogger as mylog
import yt.utilities.fortran_utils as fpu
from yt.funcs import get_pbar
import os


class HaloList(object):
    def __init__(self, ds, folder='.', contam=False):
        """
        PandaList with halos and their properties
        """

        self.folder = folder
        self.iout = int(str(ds).split('_')[1])
        self.halos = self._read_halos(data_set=ds, with_contam_option=contam)
        self.ds = ds

    def get_halo(self, hid, fname=None):

        halo = self.halos.loc[hid]
        scale_mpc = float(self.ds.length_unit.in_units('Mpc'))

        halostr = ("Halo {hid:.0f} (level {h.level:.0f}):\n"
                   "\tContains {h.nbpart:.0f} particles and {h.nbsub:.0f} subhalo(s)\n"
                   "\tCenter:\t\t ({h.x}, {h.y}, {h.z}) box units\n"
                   "\tVelocity:\t ({h.vx}, {h.vy}, {h.vz}) km/s\n"
                   "\tL:\t\t ({h.Lx}, {h.Ly}, {h.Lz}) ToCheck\n"
                   "\tMass:\t\t {h.m:.3e} Msun\n"
                   "\tMvir:\t\t {h.mvir:.3e} Msun\n"
                   "\tRadius:\t\t {h.r:.3e} Mpc ({rcodeunits:.3e} box units)\n"
                   "\tRvir:\t\t {h.rvir:.3e} Mpc ({rvcodeunits:.3e} box units)\n"
                   "\tTvir:\t\t {h.tvir:.3e} K".format(hid=hid,
                                                       h=halo,
                                                       rcodeunits=halo.r / scale_mpc,
                                                       rvcodeunits=halo.rvir / scale_mpc))

        if fname is not None:
            with open(fname, 'w') as f:
                f.write(halostr)

        return halostr

    def plot_halo(self, hid, rvir_factor=5, field=('deposit', 'all_density'), folder='./',
                  weight_field=('index', 'ones'), cmap='viridis', slice=False,
                  axis='z', **kwargs):
        '''Plot a given halo.

        Parameters
        ----------
        * hid, integer
            The halo id to plot
        * rvir_factor, float, default=5
            Size of the region to plot in unit of Rvir

        * field, tuple
            The yt field to plot
        * folder, string
            The folder where to save the data
        * weight_field, tuple
            The field to weight the projection by.
        * cmap, string
            The colormap to use
        * slice, boolean
            If true, do a slice plot instead of a projection plot
        * axis, 'x', 'y' or 'z'
            The axis to project onto
        '''
        for k, v in kwargs.items():
            print('%s: %s not supported' % (k, v))

        if hid not in self.halos.index:
            mylog.error('%s not found.' % hid)
            return

        # Get position
        tmp = np.array(self.halos.loc[hid, ['x', 'y', 'z', 'rvir']])
        center = self.ds.arr(tmp[:3], 'code_length')
        radius = self.ds.arr(tmp[3] * rvir_factor, 'Mpc')

        # Get a sphere centered on the halo
        sphere = self.ds.sphere(center, radius)

        # Make a projection plot
        p = yt.ProjectionPlot(self.ds, axis, field, data_source=sphere,
                              weight_field=weight_field)

        p.set_cmap(field=field, cmap=cmap)
        p.annotate_timestamp(corner='upper_left', time=True, redshift=True)
        p.annotate_scale(corner='upper_right')

        # TODO: annotate halos
        # TODO: better name
        p.save(folder)

    # Accessors
    def __getitem__(self, item):
        if str(item) in self.halos:
            return self.halos[item]
        else:
            return self.halos.ix[item]

    # def __getattr__(self, name):
    #     return self.halos.__getattr__(name)  # self.halos[name]

    def __len__(self):
        return len(self.halos)

    def __iter__(self):
        return self.halos.iterrows()

    # Printing functions
    def __str__(self):
        return self.halos.__str__()

    # Convenience functions
    def _read_halos(self, data_set, with_contam_option=False):
        halo_keys = ('ID', 'nbpart', 'level', 'min_part_id',
                     'host', 'hostsub', 'nbsub', 'nextsub',
                     'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz',
                     'a', 'b', 'c', 'ek', 'ep', 'et', 'rho0', 'r_c',
                     'spin', 'm', 'r', 'mvir', 'rvir', 'tvir', 'cvel')
        filename = '{s.folder}/Halos/{s.iout}/tree_bricks{s.iout:03d}'.format(
            s=self)

        data = np.empty(shape=(0, len(halo_keys)), dtype=object)
        yt.funcs.mylog.debug('Reading halo catalog %s (ds=%s)' % (filename, data_set))
        offsets = {}
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                [npart] = fpu.read_vector(f, 'i')
                [massp] = fpu.read_vector(f, 'f')
                [aexp] = fpu.read_vector(f, 'f')
                [omega_t] = fpu.read_vector(f, 'f')
                [age] = fpu.read_vector(f, 'f')
                [nhalos, nsubs] = fpu.read_vector(f, 'i')

                # Save the age/aexp, the mass of the particle,
                # as well as the number of (sub)halos
                self.nhalos = nhalos
                self.nsubs = nsubs
                self.aexp = aexp
                self.age = age
                self.massp = massp
                data = np.empty(shape=(nhalos + nsubs, len(halo_keys)), dtype=object)

                mylog.info('Brick: halos       : %s' % nhalos)
                mylog.info('Brick: sub halos   : %s' % nsubs)
                mylog.info('Brick: aexp        : %s' % aexp)

                pbar = get_pbar('', nhalos+nsubs)

                for ihalo in range(nhalos + nsubs):
                    pos = f.tell()
                    [nbpart] = fpu.read_vector(f, 'i')                 # Number of particles
                    listp = fpu.read_vector(f, 'i')                    # List of the particles IDs
                    [ID] = fpu.read_vector(f, 'i')                     # Halo ID
                    fpu.skip(f, 1)                                     # Skip timestep
                    [level, host, hostsub, nbsub, nextsub] = fpu.read_vector(f, 'i')
                    [m] = fpu.read_vector(f, 'f')                      # Total mass
                    [x, y, z] = fpu.read_vector(f, 'f')                # Center
                    [vx, vy, vz] = fpu.read_vector(f, 'f')             # Velocity
                    [Lx, Ly, Lz] = fpu.read_vector(f, 'f')             # Angular momentum
                    [r, a, b, c] = fpu.read_vector(f, 'f')             # Shape (ellipticity)
                    [ek, ep, et] = fpu.read_vector(f, 'f')             # Energetics
                    [spin] = fpu.read_vector(f, 'f')                   # Total angular momentum
                    [rvir, mvir, tvir, cvel] = fpu.read_vector(f, 'f') # Virial parameters
                    [rho0, r_c] = fpu.read_vector(f, 'f')              # NFW params

                    if with_contam_option:
                        [contam] = fpu.read_vector(f, 'i')  # Contamination

                    # Add the halo to the list
                    # halos.loc[ihalo] = [ID, nbpart, level, listp.min(),
                    #                     host, hostsub, nbsub, nextsub,
                    #                     x, y, z, vx, vy, vz, Lx, Ly, Lz,
                    #                     a, b, c, ek, ep, et, rho0, r_c,
                    #                     spin, m, r, mvir, rvir, tvir, cvel]
                    data[ihalo] = [ID, nbpart, level, listp.min(),
                                   host, hostsub, nbsub, nextsub,
                                   x, y, z, vx, vy, vz, Lx, Ly, Lz,
                                   a, b, c, ek, ep, et, rho0, r_c,
                                   spin, m, r, mvir, rvir, tvir, cvel]
                    pbar.update()
                    offsets[ID] = pos

        print('')
        types = {}
        for k in ('ID', 'nbpart', 'level', 'min_part_id',
                  'host', 'hostsub', 'nbsub', 'nextsub'):
            types[k] = np.int64
        for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz',
                  'a', 'b', 'c', 'ek', 'ep', 'et', 'rho0', 'r_c',
                  'spin', 'm', 'r', 'mvir', 'rvir', 'tvir', 'cvel'):
            types[k] = np.float64
        dd = {k: data[:, i].astype(types[k])
              for i, k in enumerate(halo_keys)}

        halos = pd.DataFrame(dd)

        # Get properties in the right units
        # Masses
        halos.m *= 1e11
        halos.mvir *= 1e11
        # Positions and distances
        scale_mpc = float(data_set.length_unit.in_units('cm') / 3.08e24)
        halos.x = halos.x / scale_mpc + .5
        halos.y = halos.y / scale_mpc + .5
        halos.z = halos.z / scale_mpc + .5

        self.offsets = offsets

        return halos.set_index('ID')

    def get_halo_parts(self, hid):
        filename = '{s.folder}/Halos/{s.iout}/tree_bricks{s.iout:03d}'.format(
            s=self)
        with open(filename, 'br') as fd:
            fd.seek(self.offsets[hid])
            fpu.skip(fd, 1)
            listp = fpu.read_vector(fd, 'i')

        return listp
