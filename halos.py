#!/usr/bin/env python

"""Module to deal with halos, to be used with HaloMaker.

This module is heavily inspired by the set of IDL routines originally found
in the Ramses Analysis ToolSuite (RATS).

TODO: Some more documentation
"""

import numpy as np
import pandas as pd
import utils
import yt

__author__ = "Maxime Trebitsch"
__copyright__ = "Copyright 2015, Maxime Trebitsch"
__credits__ = ["Maxime Trebitsch", "Jeremy Blaizot",
               "Leo Michel-Dansac"]
__license__ = "BSD"
__version__ = "0.2"
__maintainer__ = "Maxime Trebitsch"
__email__ = "maxime.trebitsch@ens-lyon.org"
__status__ = "Beta"


class HaloList(object):
    def __init__(self, ds, contam=False):
        """Some documentation, list of useful function, etc."""
        
        self.folder = '.'
        self.iout = int(str(ds)[-5:])
        self.halos = self._read_halos(data_set=ds, with_contam_option=contam)
        self.ds= ds

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
                   "\tRadius:\t\t {h.r:.3e} Mpc ({rmpc:.3e} box units)\n"
                   "\tRvir:\t\t {h.rvir:.3e} Mpc ({rvmpc:.3e} box units)\n"
                   "\tTvir:\t\t {h.tvir:.3e} K".format(h=halo,
                                               rmpc=halo.r/scale_mpc,
                                               rvmpc=halo.rvir/scale_mpc))

        if fname is not None:
            with open(fname, 'w') as f:
                f.write(halostr)

        return halostr

    def show_halos(self, hid=[], axis='z', folder='./'):
        """Plot a density map with a circle around halos
        Parameters
        ----------
        hid : list of int
            contains the ID of the halo to be plotted
        
        axis : 'x', 'y' or 'z' is the projection axis
        """
        p=yt.ProjectionPlot(self.ds, axis=axis, fields=('deposit', 'all_density'), weight_field= ('deposit', 'all_density'), axes_unit=('Mpccm'))
        hid=list(hid)
        if hid==[]: hlist=self.halos['ID']
        else: hlist=hid

        for ID in hlist:
            p.annotate_sphere([self.halos['x'][ID],self.halos['y'][ID],self.halos['z'][ID]], radius=(self.halos['rvir'][ID], 'Mpc'), circle_args={'color':'red'})
        p.annotate_timestamp(corner='upper_left', time=False, redshift=True)
        p.annotate_scale(corner='upper_right')
        #p.hide_axes()
        p.save(str(self.ds)+'_halos')
        return

    def plot_halo(self, hid, axis='z', folder='./', qty=('deposit', 'all_density')):
        """Plot a density map of halos hid
        """
        c=[self.halos['x'][hid],self.halos['y'][hid],self.halos['z'][hid]]
        r=self.halos['rvir'][hid]

        dd=self.ds.sphere(c,(3*r, ('Mpc')))
        p=yt.ProjectionPlot(self.ds, data_source=dd,axis='z', fields=qty\
        #p=yt.SlicePlot(self.ds, data_source=dd,axis='z', fields=('deposit', 'all_density')\
            ,weight_field=qty, axes_unit='Mpc', center=c, width=(4*r, 'Mpc'))
        p.annotate_timestamp(corner='upper_left', time=False, redshift=True)
        p.save(folder+str(self.ds)+'_halo'+str(hid))
        return

    ### Accessors ###
    def __getitem__(self, item):
        if str(item) in self.halos:
            return self.halos[item]
        else:
            return self.halos.ix[item]

    def __getattr__(self, name):
        return self.halos.__getattr__(name)  #self.halos[name]

    def __len__(self):
        return len(self.halos)

    def __iter__(self):
        return self.halos.iterrows()

    ### Printing functions ###
    def __str__(self):
        return self.halos.__str__()

    
    ### Convenience functions ###
    def _read_halos(self, data_set, with_contam_option=False):
        import fortranfile as ff

        halo_keys = ('ID', 'nbpart', 'level', 'min_part_id',
                     'host', 'hostsub', 'nbsub', 'nextsub',
                     'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz',
                     'a', 'b', 'c', 'ek', 'ep', 'et', 'rho0', 'r_c',
                     'spin', 'm', 'r', 'mvir', 'rvir', 'tvir', 'cvel')
        filename = '{s.folder}/Halos/{s.iout}/tree_bricks{s.iout:03d}'.format(s=self)


        with ff.FortranFile(filename) as tb:
            [npart] = tb.readInts()
            [massp] = tb.readReals()
            [aexp] = tb.readReals()
            [omega_t] = tb.readReals()
            [age] = tb.readReals()
            [nhalos, nsubs] = tb.readInts()

            # Save the age/aexp, the mass of the particle,
            # as well as the number of (sub)halos
            self.nhalos = nhalos
            self.nsubs = nsubs
            self.aexp = aexp
            self.age = age
            self.massp = massp
            data = np.empty(shape=(nhalos+nsubs, len(halo_keys)))
            #halos = pd.DataFrame(columns=halo_keys, data=np.empty(shape=(nhalos+nsubs, len(halo_keys))))

            for ihalo in range(nhalos+nsubs):
                [nbpart] = tb.readInts()  # Number of particles
                listp = np.array(tb.readInts())  # List of the particles IDs
                [ID] = tb.readInts()  # Halo ID
                [__] = tb.readInts()  # skip timestep
                [level, host, hostsub, nbsub, nextsub] = tb.readInts()
                [m] = tb.readReals()  # Total mass
                [x, y, z] = tb.readReals()  # Center
                [vx, vy, vz] = tb.readReals()  # Velocity
                [Lx, Ly, Lz] = tb.readReals()  # Angular momentum
                [r, a, b, c] = tb.readReals()  # Shape (ellipticity)
                [ek, ep, et] = tb.readReals()  # Energetics
                [spin] = tb.readReals()  # Total angular momentum
                [rvir, mvir, tvir, cvel] = tb.readReals()  # Virial parameters
                [rho0, r_c] = tb.readReals()  # ?

                if with_contam_option:
                    [contam] = tb.readInts()  # Contamination                


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
            halos = pd.DataFrame(columns=halo_keys, data=data)

            # Get properties in the right units
            # Masses
            halos.m *= 1e11
            halos.mvir *= 1e11
            # Positions and distances
            scale_mpc = float(data_set.length_unit.in_units('Mpc'))
            halos.x = halos.x/scale_mpc + .5
            halos.y = halos.y/scale_mpc + .5
            halos.z = halos.z/scale_mpc + .5

            return halos.set_index(halos.ID)
