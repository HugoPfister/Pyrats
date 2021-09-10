#!/usr/bin/env python

"""Module to deal with trees

Some more documentation

For now (and for some time), this module assumes that in the Trees folder,
there is only one set of tree_bricks (i.e. TreeMaker was run for only one set
of tree_bricks).

TODO:
    - everything
    - pep8

"""
import matplotlib
import numpy as np
#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import FortranFile as FF
import yt
from glob import glob
import os
import yt.utilities.fortran_utils as fpu
from tqdm import tqdm

from . import physics
from .utils import find_outputs


class Forest(object):
    """
    Read the outputs from TreeMaker and gather in a dataframe
    in self.trees
    units are:
    x,y,z -> Mpccm
    vx,vy,xz -> km/s (to check)
    m,mvir -> 1e11Msun
    r,rvir -> Mpc
    """

    def __init__(self, Galaxy=False, path='.', big_run=False):
        paths = find_outputs(path+'/Outputs')
        self.prefix = path
        yt.funcs.mylog.setLevel(40)
        ds = yt.load(paths[-1])

        _sim = {}
        _sim['h'] = float(ds.cosmology.hubble_constant)
        _sim['Om'] = ds.omega_matter
        _sim['Ol'] = ds.omega_lambda
        _sim['Lbox'] = float(ds.length_unit.in_units('Mpccm'))

        treefolder = os.path.join(self.prefix, 'Trees')
        self.galaxies = False
        if Galaxy:
            treefolder = os.path.join(self.prefix, 'TreeStars')
            self.galaxies = True

        self.treefolder = treefolder
        self.sim = _sim
        self.ds = ds
        self.folder = treefolder
        self.snap = self._get_timestep_number()

        self.read_tree(big_run)

    def read_tree(self, big_run=False):
        """
        """

        if self.galaxies:
            tree_file = os.path.join(self.folder, 'tree.dat')
            self = self._read_tree(tree_file, 'd', big_run)
        else:
            tree_file = glob('{}/tree_file_???.001'.format(self.folder))[0]
            self = self._read_tree(tree_file, 'f', big_run)

        return

    def map_halo_ts_to_output(self, timestep):
        timestep = timestep.astype('int64')
        bricks = sorted(glob(os.path.join(self.treefolder, 'tree_bricks???')))
        mapping = np.empty(timestep.max(), dtype=np.int64)
        for istep in sorted(timestep.unique()):
            #ioutput = int(brick.split('bricks')[1])
            ioutput = int(bricks[istep-1].split('bricks')[1])
            mapping[istep-1] = ioutput

        return mapping[timestep-1]


    def get_all_progenitors(self, hnum, timestep=None):
        """Return the reduced tree containing ONLY the progenitors of halo hid
        ==========
        * hid: ID of the halo at output timestep given by the halo/galaxy finder
        * timestep (None): timestep at which the ID must be taken, default is last timestep
        """

        if timestep is None:
            ts = self.halo_ts.max()
        print('Care, this can be very long for galaxies. The get_main_progenitor is much faster')
        print('It is OK for Haloes')
        progenitors = self._get_progenitors(hnum, timestep=timestep)

        return progenitors

    def get_main_progenitor(self, hnum, timestep=None):
        """Return the reduced tree containing ONLY the main progenitors of halo hid
        ==========
        * hid: ID of the halo at output timestep given by the halo/galaxy finder
        * timestep (None): timestep at which the ID must be taken, default is last timestep
        """
        id_main = {}
        if timestep is None:
            ts = self.trees.halo_ts.max()
        else:
            ts = timestep

        list_progenitors = [(hnum, ts)]
        fatherMass = self.trees.loc[hnum,timestep].fatherMass
        fathersID = self.trees.loc[hnum,timestep].fathersID
        while (fathersID.size > 1):
            #update ts
            ts -= 1
            #remove background
            fatherMass = fatherMass[fathersID > 0]
            fathersID = fathersID[fathersID > 0]
            #main father is the one that contains most mass
            main_father = fathersID[fatherMass.argmax()]
            list_progenitors += [(main_father, ts)]
            fatherMass = self.trees.loc[main_father, ts].fatherMass
            fathersID = self.trees.loc[main_father, ts].fathersID
        main_progs = self.trees.loc[list_progenitors]

        return main_progs

    def get_main_children(self, hnum, timestep=None):
        """Return the reduced tree containing ONLY the main children of halo hid
        ==========
        * hid: ID of the halo at output timestep given by the halo/galaxy finder
        * timestep (None): timestep at which the ID must be taken, default is last timestep
        """
        # Get current timestep
        max_ts = self.trees.halo_ts.max()
        if timestep is None:
            ts = max_ts 
        else:
            ts = timestep

        list_sons = [(hnum, ts)]
        while (ts < max_ts):
            main_son = self.trees.loc[hnum,ts].mainSon
            #update ts
            ts += 1
            list_sons += [(main_son, ts)]
        children = self.trees.loc[list_sons]

        return children

    def get_family(self, hnum, timestep=None):
        """Return the reduced tree containing ONLY the main progenitors/children of halo hid
        ==========
        * hid: ID of the halo at output timestep given by the halo/galaxy finder
        * timestep (None): timestep at which the ID must be taken, default is last timestep
        """
        # Get current timestep
        if timestep is None:
            ts = self.trees.halo_ts.max()
        else:
            ts = timestep

        child = self.get_main_children(hnum, timestep)
        fathers = self.get_main_progenitor(hnum, timestep)
        fathers = fathers.loc[fathers.halo_ts != ts]
        family = pd.concat((child, fathers))
        family = family.sort_index(level=1)

        return family

    def plot_all_trees(self, minmass=-1, maxmass=1e99, radius=1.0,
                       output=None, loc='./'):
        """
        See plot_halo_tree.
        Compute the above function for all halos with virial mass
        between `minmass` and `maxmass`. The width in Mpccm for the
        dynamics of the BH is radius. `loc` is the location to save the
        PDF.
        """

        if output is None:
            output = str(minmass) + 'To' + str(maxmass) + '_r' + str(radius)

        # Normalize min/max mass to 1e11
        minmass /= 1.e11
        maxmass /= 1.e11

        # Select halos at the last timestep of the tree, with the right mass
        # Also selects halos with at least 1 progenitor...
        mask = ((self.trees.halo_ts == self.snap) &
                (self.trees.mvir > minmass) &
                (self.trees.mvir < maxmass) &
                (self.trees.first_prog != -1))

        tid = self.trees[mask].halo_id
        print('Total: {} halos'.format(len(tid)))

        pdf = PdfPages(loc + 'trees{}.pdf'.format(output))
        try:
            fig = self.fig
        except AttributeError:
            self.fig = plt.figure(figsize=(12, 12))
        self.fig.savefig(pdf, format='pdf', dpi=200)

        for ihalo in tid:
            self.plot_halo_tree(hid=int(ihalo), radius=radius, pdffile=pdf)
            plt.close()
        pdf.close()
        plt.close('all')

        return 'OK'

    def plot_halo_tree(self, hid=None, hnum=None, hts=None, radius=1.0,
                       pdffile=None, loc='./'):
        """
        Plot Mass/Merger history of a halo
        Parameters
        ----------
        * hid (None): halo_id of the halo you want to have the merger/mass history
        * hnum (None): halo_num of the halo you want to have the
          merger history, if used then the timestep of this halo must
          be given with hts
        * radius (1): width of the window to plot the dynamics of the halo
        * loc ('./'): where to save the plot
        """

        # Define the selected halo
        if hid is None:
            hid = (self.trees.loc[(self.trees.halo_num == hnum) & (
                self.trees.halo_ts == hts)].halo_id).astype(int)
        halo = self.trees.ix[hid]
        sim = self.sim

        xh = halo['x']  # - sim['Lbox']/2 #/sim['Lbox'] + .5
        yh = halo['y']  # - sim['Lbox']/2   #/sim['Lbox'] + .5
        zh = halo['z']  # - sim['Lbox']/2   #/sim['Lbox'] + .5

        # Get progenitors
        progs = self._get_progenitors(hid)
        progs.loc[hid].descendent_id = -1
        main_progs = self.get_main_progenitor(hid)
        main_progs.loc[hid].descendent_id = -1

        # Recenter on selected halo and correct for periodic boundaries
        progs['x'] -= xh
        progs['y'] -= yh
        progs['z'] -= zh
        progs['x'].where(progs.x <= sim['Lbox'] / 2.,
                         progs.x - sim['Lbox'], inplace=True)
        progs['y'].where(progs.y <= sim['Lbox'] / 2.,
                         progs.y - sim['Lbox'], inplace=True)
        progs['z'].where(progs.z <= sim['Lbox'] / 2.,
                         progs.z - sim['Lbox'], inplace=True)
        progs['x'].where(progs.x >= -sim['Lbox'] / 2.,
                         progs.x + sim['Lbox'], inplace=True)
        progs['y'].where(progs.y >= -sim['Lbox'] / 2.,
                         progs.y + sim['Lbox'], inplace=True)
        progs['z'].where(progs.z >= -sim['Lbox'] / 2.,
                         progs.z + sim['Lbox'], inplace=True)
        main_progs['x'] -= xh
        main_progs['y'] -= yh
        main_progs['z'] -= zh
        main_progs['x'].where(main_progs.x <= sim['Lbox'] / 2.,
                              main_progs.x - sim['Lbox'], inplace=True)
        main_progs['y'].where(main_progs.y <= sim['Lbox'] / 2.,
                              main_progs.y - sim['Lbox'], inplace=True)
        main_progs['z'].where(main_progs.z <= sim['Lbox'] / 2.,
                              main_progs.z - sim['Lbox'], inplace=True)
        main_progs['x'].where(main_progs.x >= -sim['Lbox'] / 2.,
                              main_progs.x + sim['Lbox'], inplace=True)
        main_progs['y'].where(main_progs.y >= -sim['Lbox'] / 2.,
                              main_progs.y + sim['Lbox'], inplace=True)
        main_progs['z'].where(main_progs.z >= -sim['Lbox'] / 2.,
                              main_progs.z + sim['Lbox'], inplace=True)

        # Define plot range
        xc = (main_progs['x'].min() + main_progs['x'].max()) * .5
        yc = (main_progs['y'].min() + main_progs['y'].max()) * .5
        zc = (main_progs['z'].min() + main_progs['z'].max()) * .5
        xmin = ymin = zmin = -radius
        xmax = ymax = zmax = radius
        progs[['x', 'y', 'z']] -= [xc, yc, zc]
        main_progs[['x', 'y', 'z']] -= [xc, yc, zc]

        # Recenter other halos, and correct for periodic boundaries
        x = self.trees.x - (xc + xh)
        y = self.trees.y - (yc + yh)
        z = self.trees.z - (zc + zh)
        x.where(x <= sim['Lbox'] / 2., x - sim['Lbox'], inplace=True)
        y.where(y <= sim['Lbox'] / 2., y - sim['Lbox'], inplace=True)
        z.where(z <= sim['Lbox'] / 2., z - sim['Lbox'], inplace=True)
        x.where(x >= -sim['Lbox'] / 2., x + sim['Lbox'], inplace=True)
        y.where(y >= -sim['Lbox'] / 2., y + sim['Lbox'], inplace=True)
        z.where(z >= -sim['Lbox'] / 2., z + sim['Lbox'], inplace=True)

        # Select other halos within plot range
        others = ((x < radius) & (-radius < x) &
                  (y < radius) & (-radius < y) &
                  (z < radius) & (-radius < z))

        # Scalings
        scc = np.log10(progs.m)
        mainscc = np.log10(main_progs.m)
        sc = (scc - scc.min()) / (scc.max() - scc.min()) * 500.
        mainsc = (mainscc - scc.min()) / (mainscc.max() - scc.min()) * 500.

        osc = np.log10(self.trees[others].m)
        osc = (osc - scc.min()) / (scc.max() - scc.min()) * 500.
        ocol = self.trees[others].halo_ts
        ocol = .15 + .7 * (ocol - ocol.min()) / (ocol.max() - ocol.min())
        edg = np.where(self.trees[others].bush_id ==
                       halo['bush_id'], 'orange', 'k')

        # Plot halos dynamics in the comoving space
        # try:
        #    fig = self.fig
        # except AttributeError:
        self.fig = plt.figure(figsize=(12, 12))
        fig = self.fig
        # try:
        #    ax = self.ax
        # except AttributeError:
        ax10 = self.fig.add_axes([0.07, 0.05, 0.4, 0.4])
        ax00 = self.fig.add_axes([0.07, 0.5, 0.4, 0.4])
        ax01 = self.fig.add_axes([0.55, 0.5, 0.4, 0.4])
        ax11 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4])
        self.ax = [ax00, ax01, ax10, ax11]
        ax = self.ax

        self.ax[0].scatter(x[others], y[others], c=ocol, s=osc, cmap='Greys',
                           vmin=0, vmax=1., edgecolor=edg,
                           rasterized=True)
        self.ax[0].scatter(progs.x, progs.y, c=progs.halo_ts, s=sc,
                           cmap='summer', rasterized=True)
        self.ax[0].scatter(main_progs.x, main_progs.y,
                           c=main_progs.halo_ts, s=mainsc,
                           cmap='magma', rasterized=True)
        self.ax[0].set_xlabel(r'$x$ (cMpc)')
        self.ax[0].set_ylabel(r'$y$ (cMpc)')
        self.ax[0].set_xlim(xmin, xmax)
        self.ax[0].set_ylim(ymin, ymax)

        self.ax[1].scatter(z[others], y[others], c=ocol, s=osc, cmap='Greys',
                           vmin=0, vmax=1., edgecolor=edg,
                           rasterized=True)
        self.ax[1].scatter(progs.z, progs.y, c=progs.halo_ts, s=sc,
                           cmap='summer', rasterized=True)
        self.ax[1].scatter(main_progs.z, main_progs.y,
                           c=main_progs.halo_ts, s=mainsc,
                           cmap='magma', rasterized=True)
        self.ax[1].set_xlabel(r'$z$ (cMpc)')
        self.ax[1].set_ylabel(r'$y$ (cMpc)')
        self.ax[1].set_xlim(zmin, zmax)
        self.ax[1].set_ylim(ymin, ymax)

        self.ax[2].scatter(x[others], z[others], c=ocol, s=osc, cmap='Greys',
                           vmin=0, vmax=1., edgecolor=edg,
                           rasterized=True)
        self.ax[2].scatter(progs.x, progs.z, c=progs.halo_ts, s=sc,
                           cmap='summer', rasterized=True)
        self.ax[2].scatter(main_progs.x, main_progs.z,
                           c=main_progs.halo_ts, s=mainsc,
                           cmap='magma', rasterized=True)
        self.ax[2].set_xlabel(r'$x$ (cMpc)')
        self.ax[2].set_ylabel(r'$z$ (cMpc)')
        self.ax[2].set_xlim(xmin, xmax)
        self.ax[2].set_ylim(zmin, zmax)

        # Draw some lines
        for progid in progs.halo_id:
            descid = progs.ix[progid].descendent_id
            if descid > 0:
                progenitor = progs.ix[progid]
                descendent = progs.ix[descid]
                px, py, pz = progenitor[['x', 'y', 'z']]
                dx, dy, dz = descendent[['x', 'y', 'z']]
                l0 = self.ax[0].plot([px, dx], [py, dy],
                                     lw=1, c='k', zorder=-1)
                l1 = self.ax[1].plot([pz, dz], [py, dy], lw=1, c='k',
                                     zorder=-1)
                l2 = self.ax[2].plot([px, dx], [pz, dz], lw=1, c='k',
                                     zorder=-1)

        title = "Halo #{t:d}\n(x, y, z) = ({x:3.3f}, {y:3.3f}, {z:3.3f})"
        title = title.format(t=int(halo.halo_num),
                             x=xh / sim['Lbox'] + .5,
                             y=yh / sim['Lbox'] + .5,
                             z=zh / sim['Lbox'] + .5)
        suptitle = self.fig.suptitle(title, fontsize=18)

        if not self.fig.texts:
            self.fig.texts.append(suptitle)
        if pdffile:
            self.fig.savefig(pdffile, format='pdf', dpi=200)
        else:
            plt.savefig(loc + 'halo_{:d}dynamics.png'.format(int(halo.halo_num)), dpi=100, format='png')

        # A bit of cleaning
        # for axx in self.ax:
        #    axx.clear()
        #self.fig.texts = []
        fig.clf()
        # try:
        #    fig = self.fig
        # except AttributeError:
        #    self.fig = plt.figure(figsize=(12, 12))
        # try:
        #    ax = self.ax
        # except AttributeError:
        ax10 = self.fig.add_axes([0.07, 0.05, 0.4, 0.4])
        ax00 = self.fig.add_axes([0.07, 0.5, 0.4, 0.4])
        ax01 = self.fig.add_axes([0.55, 0.5, 0.4, 0.4])
        ax11 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4])
        self.ax = [ax00, ax01, ax10, ax11]
        ax = self.ax

        # Plot halo mass/merger history

        time = np.array([self.timestep['age'][int(hts) - 1]
                         for hts in progs.halo_ts])
        maintime = np.array([self.timestep['age'][int(hts) - 1]
                             for hts in main_progs.halo_ts])
        self.ax[0].scatter(time, progs.m * 1e11, c=progs.halo_ts,
                           s=sc, cmap='summer', rasterized=True)
        self.ax[0].semilogy(
            [self.ds.cosmology.t_from_z(z) / (3600 * 24 * 365 * 1e9)
             for z in np.logspace(-10, 2, 1000)],
            1e12 * physics.MofZ(halo.m * 1e11 / 1e12,
                                np.logspace(-10, 2, 1000),
                                1 / self.timestep['aexp'][progs.loc[hid].halo_ts - 1] - 1),
            'k')
        self.ax[0].scatter(maintime, main_progs.m * 1e11,
                           c=main_progs.halo_ts, s=mainsc,
                           cmap='magma', rasterized=True)
        self.ax[0].set_xlabel(r'$t$ (Gyr)')
        self.ax[0].set_ylabel(r'$M$ ($\mathrm{M}_{\odot}$)')
        self.ax[0].set_ylim(1e11 * progs.m.min() / 3.,
                            1e11 * progs.m.max() * 3.)
        self.ax[0].set_xlim(time.min(), time.max())
        self.ax[0].set_yscale('log')
        z = np.unique([int((1 / self.timestep['aexp'][int(hts) - 1] - 1) * 2) / 2.
                       for hts in main_progs.halo_ts])
        z = z[1:-1]
        ax_z = self.ax[0].twiny()
        ax_z.set_xlim(self.ax[0].get_xlim())
        ax_z.set_xticks([float(self.ds.cosmology.t_from_z(
            zz) / (1e9 * 365 * 24 * 3600)) for zz in z])
        ax_z.set_xticklabels(z)
        ax_z.set_xlabel("redshift")

        main_prog = self.get_main_progenitor(hid)
        minor_mergers = []
        major_mergers = []
        for current_id in main_prog.halo_id:
            if main_prog.halo_ts[current_id] != main_prog.halo_ts.min():
                halo_ts = main_prog.halo_ts[current_id]
                progs = self.get_all_progenitors(
                    current_id, timestep=halo_ts - 1)
                mask_minor = ((progs.m / main_prog.m[current_id + 1] > 1. / 20) &
                              (progs.m / main_prog.m[current_id + 1] < 1. / 4))
                minor_mergers += [len(progs.m[mask_minor])]
                mask_major = (progs.m / main_prog.m[current_id + 1] >= 1. / 4)
                major_mergers += [len(progs.m[mask_major]) - 1]

        scc = np.log10(main_prog.m)
        sc = (scc - scc.min()) / (scc.max() - scc.min()) * 500.
        time = np.array([self.timestep['age'][hts - 1]
                         for hts in main_prog[main_prog.halo_ts !=
                                              self.trees.halo_ts[hid]].halo_ts])
        self.ax[2].plot(time, # np.cumsum(major_mergers)[-1] -
                        np.cumsum(major_mergers),
                        color='b', label='1:1 > mass ratio > 1:4 (sim)')

        self.ax[2].plot(time,
                        [physics.N_merger_until_z(
                            0.25,
                            halo.m * 1e11 / 1e12,
                            self.ds.cosmology.z_from_t(t * 1e9 * 365 * 24 * 3600),
                            self.ds.cosmology.z_from_t(time.max() * 1e9 * 365 * 24 * 3600))
                         for t in time],
                        color='b', linestyle='--',
                        label='1:1 > mass ratio > 1:4 (theory)')

        self.ax[2].plot(time, # np.cumsum(minor_mergers)[-1] -
                        np.cumsum(minor_mergers), color='r',
                        label='1:4 > mass ratio > 1:20 (sim)')

        self.ax[2].plot(time,
                        [physics.N_merger_until_z(
                            1. / 20,
                            halo.m * 1e11 / 1e12,
                            self.ds.cosmology.z_from_t(t * 1e9 * 365 * 24 * 3600),
                            self.ds.cosmology.z_from_t(time.max() * 1e9 * 365 * 24 * 3600))
                         - physics.N_merger_until_z(
                             1. / 4,
                             halo.m * 1e11 / 1e12,
                             self.ds.cosmology.z_from_t(t * 1e9 * 365 * 24 * 3600),
                             self.ds.cosmology.z_from_t(time.max() * 1e9 * 365 * 24 * 3600))
                         for t in time],
                        color='r', linestyle='--',
                        label='1:4 > mass ratio > 1:20 (theory)')
        # self.ax[2].scatter(time, [minor_mergers[m] for m in major_mergers],
        #    c=main_prog[main_prog.halo_ts != main_prog.halo_ts.max()].halo_ts,
        #    s=sc, cmap='summer', rasterized=True)
        # self.ax[2].scatter(time, [major_mergers[m] for m in major_mergers],
        #    c=main_prog[main_prog.halo_ts != main_prog.halo_ts.max()].halo_ts,
        #    s=sc, cmap='magma', rasterized=True, marker='*')
        self.ax[2].legend(loc='best')
        self.ax[2].set_ylabel('#merger left before t$_{max}$')
        self.ax[2].set_xlabel(r'$t$ (Gyr)')
        self.ax[2].set_xlim(time.min(), time.max())
        self.ax[2].set_ylim(ymin=0)
        z = np.unique([int(1 / self.timestep['aexp'][int(hts) - 1] - 1)
                       for hts in main_progs.halo_ts])
        z = z[1:-1]
        ax_z = self.ax[2].twiny()
        ax_z.set_xlim(self.ax[0].get_xlim())
        ax_z.set_xticks([np.copy(self.ds.cosmology.t_from_z(
            zz) / (1e9 * 365 * 24 * 3600)) for zz in z])
        ax_z.set_xticklabels(z)

        title = "Halo #{t:d}\nFinal virial mass = {x:.2e} M$_\odot$"
        title = title.format(t=int(halo.halo_num),
                             x=halo['mvir'] * 1e11)
        suptitle = self.fig.suptitle(title, fontsize=18)

        if not self.fig.texts:
            self.fig.texts.append(suptitle)
        if pdffile:
            self.fig.savefig(pdffile, format='pdf', dpi=200)
        else:
            plt.savefig('halo_{:d}.png'.format(
                int(halo.halo_num)), dpi=100, format='png')

        # A bit of cleaning
        # for axx in self.ax:
        #    axx.clear()
        #self.fig.texts = []
        fig.clf()

        return hid

    def plot_accretion_history(self, hid, scale='mass'):
        """Plot halo accretion history.

        Blah
        """
        # Define the selected halo
        halo = self.trees.ix[hid]
        sim = self.sim

        # Get progenitors
        progs = self._get_progenitors(hid)

        # Scalings
        scc = np.log10(progs.m)
        sc = (scc - scc.min()) / (scc.max() - scc.min()) * 500.

        try:
            fig = self.fig
        except AttributeError:
            self.fig = plt.figure(figsize=(12, 12))
        try:
            ax = self.ax
        except AttributeError:
            self.ax = self.fig.add_subplot(111)

        time = np.array([self.timestep['age'][hts - 1]
                         for hts in progs.halo_ts])
        self.ax.scatter(time, progs.m * 1e11, c=progs.halo_ts, s=sc,
                        cmap='summer', rasterized=True)
        self.ax.set_xlabel(r'$t$ (Gyr)')
        self.ax.set_ylabel(r'$M$ ($\mathrm{M}_{\odot}$)')
        self.ax.set_ylim(1e11 * progs.m.min() / 3., 1e11 * progs.m.max() * 3.)
        self.ax.set_xlim(time.min(), time.max())
        self.ax.set_yscale('log')

        self.fig.tight_layout()
        self.fig.savefig('halo_{:d}_history.pdf'.format(
            int(halo.halo_num)), dpi=200, format='pdf')

        return hid

    # CONVENIENCE FUNCTIONS
    def _get_timestep_number(self):
        l = len(glob(os.path.join(self.folder, 'halos_results.???')))
        return l

    def get_ioutlist(self, inputfile='input_TreeMaker.dat'):
        """Return the number of treebricks files from an input file

        The input_TreeMaker.dat file must be a list of:
            nsnaps some_number
            '/path/to/tree_bricks{n:03d}'
            '/path/to/tree_bricks{n+1:03d}'
            ...
            '/path/to/tree_bricks{n+nsnaps:03d}'
            ...

        What should be done:
            * Read all the /path/to/tree_bricks lines
            * For each line, get the tree_bricks number
            * Return the mapping between [1, nsnaps] and the tree_bricks list.
        """
        import os.path

        with open(self.folder + inputfile, 'r') as ifile:
            lines = ifile.readlines()

        outputs = []

        # We know that the name of the tree_bricks file should be
        # "tree_bricksXXX"
        tbbasename = 'tree_bricks'
        # We can skip the first line, which should be "nsnaps 1"
        for path in lines[1:]:
            # Remove those nasty "'" and \n
            tbname = os.path.basename(path[1:-2])
            assert tbname.startswith(tbbasename)
            outputs.append(int(tbname[len(tbbasename):]))

        return outputs

    def _get_progenitors(self, hid, timestep):
        if self.galaxies:
            progenitors = self.trees.loc[(self.trees.halo_num == hid) & (self.trees.halo_ts == timestep)]
            if len(progenitors) == 0:
                return
            else:
                if len(progenitors.fathersID.item()) == 1:
                    return progenitors
                else:
                    for fatherID in progenitors.fathersID.item():
                        progenitors = pd.concat([progenitors, self._get_progenitors(fatherID, timestep-1)])
                    return progenitors

        else:
            target = self.trees.loc[(self.trees.halo_num == hid) & (self.trees.halo_ts == timestep)].index
            print(target)
            mask = ((self.trees.halo_id >= self.trees.loc[target].halo_id.item()) &
                    (self.trees.halo_id <= self.trees.loc[target].last_prog.item()))

            progenitors = self.trees.loc[mask].copy()
            return progenitors

    def _read_halo(self, F, precision, big_run=False):
        [ID] = fpu.read_vector(F, 'i')
        [BushID] = fpu.read_vector(F, 'i')
        [timestep] = fpu.read_vector(F, 'i')
        [level, hosthaloID, hostsubID, nbsub, nextsub] = fpu.read_vector(F, 'i')
        [m] = fpu.read_vector(F, precision)
        [dmacc] = fpu.read_vector(F, 'd')
        [x, y, z] = fpu.read_vector(F, precision)
        [vx, vy, vz] = fpu.read_vector(F, precision)
        [Lx, Ly, Lz] = fpu.read_vector(F, precision)
        [r, a, b, c] = fpu.read_vector(F, precision)
        [ek, ep, et] = fpu.read_vector(F, precision)
        [spin] = fpu.read_vector(F, precision)
        [nbfather] = fpu.read_vector(F, 'i')
        if nbfather == 0:
            fatherID = []
            fatherMass = []
        else:
            fatherID = fpu.read_vector(F, 'i')
            fatherMass = fpu.read_vector(F, precision)
        [nbsons] = fpu.read_vector(F, 'i')
        if nbsons == 0:
            sonsID = []
            mainSon = []
        else:
            sonsID = fpu.read_vector(F, 'i')
        mainSon = fpu.read_vector(F, 'i')[0]
        if self.galaxies:
            [rvir, mvir, tvir, cvel,Reff] = fpu.read_vector(F, precision)
        else:
            [rvir, mvir, tvir, cvel] = fpu.read_vector(F, precision)
        [rho_0, r_c] = fpu.read_vector(F, precision)
        if not big_run:
            fpu.read_vector(F, 'i')  # ncont

        return (ID, timestep, level, hosthaloID, hostsubID, m,
                           dmacc, x, y, z, vx, vy, vz, Lx, Ly, Lz, r, a, b,
                           c, ek, ep, et, spin, rvir, mvir, tvir, cvel,
                           fatherID, fatherMass, sonsID, mainSon, Reff)


    def _read_tree(self, tree_file, precision, big_run=False):
        Key_tree = [
            'halo_num', 'tree_step', 'level', 'host_halo_id', 'host_sub_id', 'm',
            'dmacc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz', 'r',
            'a', 'b', 'c', 'ek', 'ep', 'et', 'spin', 'rvir', 'mvir', 'tvir',
            'cvel', 'fathersID', 'fatherMass', 'sonsID', 'mainSon']
        if self.galaxies:
            Key_tree += ['Reff']

        with open(tree_file, 'rb') as F:
            self.timestep = {}
            [self.timestep['nsteps']] = fpu.read_vector(F, 'i')
            n_halo_tot = fpu.read_vector(F, 'i')
            self.timestep['nhalos'] = n_halo_tot[:self.timestep['nsteps']]
            self.timestep['aexp'] = fpu.read_vector(F, precision)
            fpu.read_vector(F, precision)
            self.timestep['age'] = fpu.read_vector(F, precision)

            data = np.empty(shape=(n_halo_tot.sum(), len(Key_tree)), dtype=object)
            if (os.path.exists('{}/tree.hdf'.format(self.folder))):
              self.trees = pd.read_hdf('{}/tree.hdf'.format(self.folder))
            else:
              print('Reading the tree for the first time... be patient!')
              j = 0
              for istep in tqdm(range(self.timestep['nsteps'])):
                for ihalo in range(n_halo_tot[istep]+n_halo_tot[istep+self.timestep['nsteps']]):
                    data[j] = self._read_halo(F, precision, big_run) 
                    j += 1

              dd = {k: data[:, i]
                  for i, k in enumerate(Key_tree)}
              self.trees = pd.DataFrame(dd)
              for k in ['halo_num', 'tree_step', 'level', 'host_halo_id', 'host_sub_id']:
                    self.trees[k] = self.trees[k].astype(np.int32)
              for k in ['m', 'dmacc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly',
                  'Lz', 'r', 'a', 'b', 'c', 'ek', 'ep', 'et', 'spin', 'rvir',
                  'mvir', 'tvir', 'cvel']:
                self.trees[k] = self.trees[k].astype(np.float32)


              #process a bit the data
              self.timestep['aexp'] = 1/(1+self.ds.cosmology.z_from_t(self.ds.arr(self.timestep['age'],'Gyr'))) 
              ####### Create halo_ts from the step in the tree
              self.trees['halo_ts'] = self.map_halo_ts_to_output(self.trees.tree_step)
              iout = self.trees.tree_step.values.astype(int)-1
              self.trees['aexp'] = self.timestep['aexp'][iout]
              aexp = self.trees['aexp']
              aexp_last = self.timestep['aexp'].max() 
      
              factor = float(self.ds.length_unit.in_units('cm') / 3.08e24) * aexp / aexp_last 
              self.trees['x'] = self.trees['x'] / factor + 0.5
              self.trees['y'] = self.trees['y'] / factor + 0.5
              self.trees['z'] = self.trees['z'] / factor + 0.5
      
              # Find outputs given halo_ts
              paths = find_outputs(self.prefix+'/Outputs')
              self.outputs = paths[-int(self.trees.halo_ts.max()):]
              step_first_gal = len(paths) - self.trees.halo_ts.max()
              self.trees.halo_ts += step_first_gal

              self.trees.m *= 1e11
              self.trees.dmacc *= 1e11

              self.trees.set_index(['halo_num','halo_ts'], drop=False, inplace=True)
              self.trees.to_hdf('{}/tree.hdf'.format(self.folder), 'hdf5')

        return self
