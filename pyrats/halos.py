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
from scipy.io import FortranFile as FF

from . import fields, sink


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

        halostr = ("Halo {h.ID:.0f} (level {h.level:.0f}):\n"
                   "\tContains {h.nbpart:.0f} particles and {h.nbsub:.0f} subhalo(s)\n"
                   "\tCenter:\t\t ({h.x}, {h.y}, {h.z}) box units\n"
                   "\tVelocity:\t ({h.vx}, {h.vy}, {h.vz}) km/s\n"
                   "\tL:\t\t ({h.Lx}, {h.Ly}, {h.Lz}) ToCheck\n"
                   "\tMass:\t\t {h.m:.3e} Msun\n"
                   "\tMvir:\t\t {h.mvir:.3e} Msun\n"
                   "\tRadius:\t\t {h.r:.3e} Mpc ({rcodeunits:.3e} box units)\n"
                   "\tRvir:\t\t {h.rvir:.3e} Mpc ({rvcodeunits:.3e} box units)\n"
                   "\tTvir:\t\t {h.tvir:.3e} K".format(h=halo,
                                                       rcodeunits=halo.r / scale_mpc,
                                                       rvcodeunits=halo.rvir / scale_mpc))

        if fname is not None:
            with open(fname, 'w') as f:
                f.write(halostr)

        return halostr

    def show_halos(self, hid=[], axis='z', folder='./',
                   field=('deposit', 'all_density'),
                   weight_field=('index', 'ones')):
        """
        Plot a density map of the whole box with a circle around halos
        (pretty useless, documentation TBW)
        Parameters
        ----------
        hid : list of int
            contains the ID of the halo to be plotted

        axis : 'x', 'y' or 'z' is the projection axis
        """
        p = yt.ProjectionPlot(self.ds, axis=axis, fields=field, axes_unit=(
            'Mpccm'), weight_field=weight_field)
        hid = list(hid)
        if hid == []:
            hlist = self.halos['ID']
        else:
            hlist = hid

        for ID in hlist:
            p.annotate_sphere(
                [self.halos['x'][ID], self.halos['y'][ID], self.halos['z'][ID]],
                radius=(self.halos['rvir'][ID], 'Mpc'),
                circle_args={'color': 'yellow'})
        p.annotate_timestamp(corner='upper_left', time=False, redshift=True)
        p.set_cmap(field=field, cmap='viridis')
        p.annotate_scale(corner='upper_right')
        # p.hide_axes()
        p.save(folder + str(self.ds) + '_halos')
        return

    def plot_halo(self, hid, axis='z', folder='./',
                  field=('deposit', 'all_density'), r=None,
                  slice=False, weight_field=('index', 'ones'),
                  cmap='viridis', limits=[0, 0], plotsinks=False,
                  units=None, plothalos=False, masshalomin=1e10):
        """
        Plot a map centered on halo with ID hid
        Parameters
        ---------
        * hid: ID of the halo you want to center the map on
        * axis ('z'): Axis to do the projection
        * folder ('./'): where to save the map
        * field (('deposit','all_density')): field you want to project
        * r (None): width of the map e.g. (10, 'Mpc')
        * slice (False): If True then slice instead of projection
        * weight_field ('index','ones'): field used to weight the
          projection, default value is here to divide by the length of
          the LOS
        * cmap ('viridis'): colormap used for the map
        * limits ([0,0]): Min and Max limits for the colorbar, if
          Max/Min > 50 then logscale is used
        * plotsinks (True): Plot black dots at the position of BHs with
          their ID aside
        * units (None): Units for the width e.g. ('Mpccm')
        * plothalos (False): add black circles to show halos of mass
          greater than masshalomin, the radius of the circles is the
          virial radius of the halo
        * masshalomin (1e10): see above

        /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\
        This routines must be updated if filtered quantities want to be
        shown (except for stars and dm which is already implemented)
        """
        c = [self.halos['x'][hid], self.halos['y'][hid], self.halos['z'][hid]]

        if 'stars' in field[1]:
            yt.add_particle_filter(
                "stars", function=fields.stars, filtered_type="all",
                requires=["particle_age"])
            self.ds.add_particle_filter("stars")
        if 'dm' in field[1]:
            yt.add_particle_filter(
                "dm", function=fields.dm, filtered_type="all")
            self.ds.add_particle_filter("dm")

        if r is None:
            r = self.ds.arr(2 * self.halos['rvir'][hid], 'Mpc')
            dd = self.ds.sphere(c, r)
        else:
            dd = self.ds.sphere(c, r)

        if slice:
            p = yt.SlicePlot(self.ds, data_source=dd,
                             axis=axis, fields=field, center=c)
        else:
            p = yt.ProjectionPlot(self.ds, data_source=dd, axis=axis,
                                  fields=field, center=c,
                                  weight_field=weight_field)

        p.set_width((float(dd.radius.in_units('kpccm')), str('kpccm')))
        if limits != [0, 0]:
            p.set_zlim(field, limits[0], limits[1])
            if limits[1] / limits[0] > 50:
                p.set_log(field, log=True)

        if units != None:
            p.set_unit(field=field, new_unit=units)

        if plotsinks:
            h = self.halos.loc[hid]
            self.ds.sink = sink.get_sinks(self.ds)
            for bhid in self.ds.sink.ID:
                ch = self.ds.sink.loc[self.ds.sink.ID == bhid]
                if (((h.x.item() - ch.x.item())**2 +
                     (h.y.item() - ch.y.item())**2 +
                     (h.z.item() - ch.z.item())**2) <
                    ((dd.radius.in_units('code_length') / 2)**2)):

                    p.annotate_marker(
                        [ch.x.item(), ch.y.item(), ch.z.item()],
                        marker='.', plot_args={'color': 'black', 's': 100})

                    p.annotate_text([ch.x.item(), ch.y.item(), ch.z.item()],
                                    text=str(ch.ID.item()),
                                    text_args={'color': 'black'})

        if plothalos:
            h = self.halos.loc[hid]
            for hid in self.ID:
                ch = self.loc[hid]
                if ((ch.m > masshalomin) &
                    (((h.x.item() - ch.x.item())**2 +
                      (h.y.item() - ch.y.item())**2 +
                      (h.z.item() - ch.z.item())**2) <
                     ((dd.radius.in_units('code_length') / 2)**2))):

                    p.annotate_sphere([ch.x.item(), ch.y.item(), ch.z.item()],
                                      (ch.rvir.item(), 'Mpc'),
                                      circle_args={'color': 'black'})

                    p.annotate_text([ch.x.item(), ch.y.item(),
                                     ch.z.item()], text=str(int(ch.ID.item())))

        p.annotate_timestamp(corner='upper_left', time=True, redshift=True)
        p.set_cmap(field=field, cmap=cmap)
        p.annotate_scale(corner='upper_right')
        p.save(folder + '/' + str(self.ds) + '_halo' + str(hid))
        return

    # Accessors
    def __getitem__(self, item):
        if str(item) in self.halos:
            return self.halos[item]
        else:
            return self.halos.ix[item]

    def __getattr__(self, name):
        return self.halos.__getattr__(name)  # self.halos[name]

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
                pbar.update()
                [nbpart] = fpu.read_vector(f, 'i')  # Number of particles
                listp = fpu.read_vector(f, 'i')  # List of the particles IDs
                [ID] = fpu.read_vector(f, 'i')  # Halo ID
                fpu.skip(f, 1) # Skip timestep
                [level, host, hostsub, nbsub, nextsub] = fpu.read_vector(f, 'i')
                [m] = fpu.read_vector(f, 'f')  # Total mass
                [x, y, z] = fpu.read_vector(f, 'f')  # Center
                [vx, vy, vz] = fpu.read_vector(f, 'f')  # Velocity
                [Lx, Ly, Lz] = fpu.read_vector(f, 'f')  # Angular momentum
                [r, a, b, c] = fpu.read_vector(f, 'f')  # Shape (ellipticity)
                [ek, ep, et] = fpu.read_vector(f, 'f')  # Energetics
                [spin] = fpu.read_vector(f, 'f')  # Total angular momentum
                [rvir, mvir, tvir, cvel] = fpu.read_vector(f, 'f')  # Virial parameters
                [rho0, r_c] = fpu.read_vector(f, 'f')  # ?

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

            return halos.set_index('ID')


