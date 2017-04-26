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
matplotlib.use('PDF')
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import fortranfile as ff
import halos
import utils
from time import sleep
from tqdm import tqdm


class Forest(object):
    def __init__(self, ds):
                 # sim={'Lbox': 10.0, 'h': 0.6711, 'Om': 0.3175, 'Ol': 0.6825}):
        sim=ds
        _sim = {}
        _sim['h'] = float(ds.cosmology.hubble_constant)
        _sim['Om'] = ds.omega_matter
        _sim['Ol'] = ds.omega_lambda
        _sim['Lbox'] = float(ds.length_unit.in_units('Mpccm')) 

        treefolder = './Trees/'
        
        self.sim = _sim
        self.folder = treefolder
        self.snap = self._get_timestep_number()

        self.read_tree()


    def read_tree(self):
        """Creates a tree.
        
        TODO: create Tree class, read cosmo from a Cosmology class (?)
        """
        
        tstep_file = '{}/tstep_file_{:03d}.001'.format(self.folder, self.snap)
        tree_file = '{}/tree_file_{:03d}.001'.format(self.folder, self.snap)
        props_file = '{}/props_{:03d}.001'.format(self.folder, self.snap)
        
        self.timestep = self._read_timesteps_props(tstep_file)
        self.struct = self._read_tree_struct(tree_file)
        self.prop = self._read_halo_props(props_file)
        
        self.struct.set_index(self.struct.halo_id, inplace=True)
        self.prop.set_index(self.struct.halo_id, inplace=True)
        self.trees = pd.concat([self.prop, self.struct], axis=1)
        
        st = self.struct['halo_ts'] - 1
        aexp = np.array([self.timestep['aexp'][int(i)] for i in st])
        self.trees['x'] = self.trees['x'] / aexp #+ self.sim['Lbox']/2
        self.trees['y'] = self.trees['y'] / aexp #+ self.sim['Lbox']/2
        self.trees['z'] = self.trees['z'] / aexp #+ self.sim['Lbox']/2


        return

    def get_main_progenitor(self, hnum, timestep=None):
        """Return the main progenitors of a halo with their timestep."""

        # Define the timestep
        if not timestep:
            timestep = self.snap

        # Define the selected halo
        target_id = (hnum-1) * 1000000  # FIXME: is this robust?
        target = self.struct.ix[target_id]
        progenitors = self._get_progenitors(target_id)

        id_main = {}
        for ts in xrange(timestep+1):
            prog_mass = progenitors[progenitors.halo_ts == ts].m
            if len(prog_mass):
                id_main[ts] = int(prog_mass.argmax())

        main_progs = pd.concat([self.trees.ix[id_main[ts]] for ts in id_main], axis=1, join='inner').T
        return main_progs

    def get_all_progenitors(self, hnum, timestep=None):
        """Return the main progenitors of a halo with their timestep."""

        # Define the timestep
        if not timestep:
            timestep = self.snap

        # Define the selected halo
        target_id = (hnum-1) * 1000000  # FIXME: is this robust?
        target = self.struct.ix[target_id]
        progenitors = self._get_progenitors(target_id)

        return progenitors
                
    def plot_all_trees(self, minmass=-1, maxmass=1e15, radius=1.0, output=None, loc='./'):
        """Blah

        Blah
        """

        if output==None:
            output=str(minmass)+'To'+str(maxmass)+'_r'+str(radius)

        # Normalize min/max mass to 1e11
        minmass /= 1.e11
        maxmass /= 1.e11

        # Select halos at the last timestep of the tree, with the right mass (in M, not Mvir)
        mask = ((self.trees.halo_ts == self.snap) &
                (self.trees.m > minmass) &
                (self.trees.m < maxmass))

        tid = self.trees[mask].halo_id
        print 'Total: {} halos'.format(len(tid))

        pdf = PdfPages(loc+'trees{}.pdf'.format(output))
        for ihalo in tqdm(tid):
            self.plot_halo_tree(int(ihalo), radius=radius, pdffile=pdf)
        pdf.close()

        return 'OK'
                

    def plot_halo_tree(self, hid, scale='mass', radius=1.0, pdffile=None):
        """Plot halo tree...
        
        Blah
        """

        from matplotlib import cm

        # Define the selected halo
        halo = self.trees.ix[hid]
        sim = self.sim

        xh = halo['x'] #- sim['Lbox']/2 #/sim['Lbox'] + .5
        yh = halo['y'] #- sim['Lbox']/2   #/sim['Lbox'] + .5
        zh = halo['z'] #- sim['Lbox']/2   #/sim['Lbox'] + .5

        # Get progenitors
        progs = self._get_progenitors(hid)

        # Recenter on selected halo and correct for periodic boundaries
        progs['x'] -= xh
        progs['y'] -= yh
        progs['z'] -= zh
        progs['x'].where(progs.x <= sim['Lbox']/2.,
                          progs.x - sim['Lbox'], inplace=True)
        progs['y'].where(progs.y <= sim['Lbox']/2.,
                          progs.y - sim['Lbox'], inplace=True)
        progs['z'].where(progs.z <= sim['Lbox']/2.,
                          progs.z - sim['Lbox'], inplace=True)
        progs['x'].where(progs.x >= -sim['Lbox']/2.,
                          progs.x + sim['Lbox'], inplace=True)
        progs['y'].where(progs.y >= -sim['Lbox']/2.,
                          progs.y + sim['Lbox'], inplace=True)
        progs['z'].where(progs.z >= -sim['Lbox']/2.,
                          progs.z + sim['Lbox'], inplace=True)

        # Define plot range
        xc = (progs['x'].min() + progs['x'].max())*.5
        yc = (progs['y'].min() + progs['y'].max())*.5
        zc = (progs['z'].min() + progs['z'].max())*.5
        xmin = ymin = zmin = -radius
        xmax = ymax = zmax = radius
        progs[['x', 'y', 'z']] -= [xc, yc, zc]

        # Recenter other halos, and correct for periodic boundaries
        x = self.trees.x - (xc + xh)
        y = self.trees.y - (yc + yh)
        z = self.trees.z - (zc + zh)
        x.where(x <= sim['Lbox']/2., x - sim['Lbox'], inplace=True)
        y.where(y <= sim['Lbox']/2., y - sim['Lbox'], inplace=True)
        z.where(z <= sim['Lbox']/2., z - sim['Lbox'], inplace=True)
        x.where(x >= -sim['Lbox']/2., x + sim['Lbox'], inplace=True)
        y.where(y >= -sim['Lbox']/2., y + sim['Lbox'], inplace=True)
        z.where(z >= -sim['Lbox']/2., z + sim['Lbox'], inplace=True)

        # Select other halos within plot range
        others = ((x < radius) & (-radius < x) &
                  (y < radius) & (-radius < y) &
                  (z < radius) & (-radius < z))

        # Scalings
        scc = np.log10(progs.m)
        sc = (scc - scc.min())/(scc.max() - scc.min()) * 500.

        osc = np.log10(self.trees[others].m)
        osc = (osc - scc.min())/(scc.max() - scc.min()) * 500.
        ocol = self.trees[others].halo_ts
        ocol = .15 + .7 * (ocol - ocol.min()) / (ocol.max() - ocol.min())
        edg = np.where(self.trees[others].bush_id == halo['bush_id'],'orange', 'k')



        try:
            fig = self.fig
        except AttributeError:
            self.fig = plt.figure(figsize=(12, 12))
        try:
            ax = self.ax
        except AttributeError:
            ax10 = self.fig.add_axes([0.07, 0.05, 0.4, 0.4])
            ax00 = self.fig.add_axes([0.07, 0.5, 0.4, 0.4])
            ax01 = self.fig.add_axes([0.55, 0.5, 0.4, 0.4])
            ax11 = self.fig.add_axes([0.55, 0.05, 0.4, 0.4])
            self.ax = [ax00, ax01, ax10, ax11]

        self.ax[0].scatter(x[others], y[others], c=ocol, s=osc, cmap='Greys',
                           vmin=0, vmax=1., edgecolor=edg,
                           rasterized=True)
        self.ax[0].scatter(progs.x, progs.y, c=progs.halo_ts, s=sc, cmap='summer',
                           rasterized=True)
        
        self.ax[0].set_xlabel(r'$x$ (cMpc)')
        self.ax[0].set_ylabel(r'$y$ (cMpc)')
        self.ax[0].set_xlim(xmin, xmax)
        self.ax[0].set_ylim(ymin, ymax)
        
        self.ax[1].scatter(z[others], y[others], c=ocol, s=osc, cmap='Greys',
                                      vmin=0, vmax=1., edgecolor=edg,
                           rasterized=True)
        self.ax[1].scatter(progs.z, progs.y, c=progs.halo_ts, s=sc, cmap='summer',
                           rasterized=True)
        self.ax[1].set_xlabel(r'$z$ (cMpc)')
        self.ax[1].set_ylabel(r'$y$ (cMpc)')
        self.ax[1].set_xlim(zmin, zmax)
        self.ax[1].set_ylim(ymin, ymax)
        
        self.ax[2].scatter(x[others], z[others], c=ocol, s=osc, cmap='Greys',
                                      vmin=0, vmax=1., edgecolor=edg,
                           rasterized=True)
        self.ax[2].scatter(progs.x, progs.z, c=progs.halo_ts, s=sc, cmap='summer',
                           rasterized=True)
        self.ax[2].set_xlabel(r'$x$ (cMpc)')
        self.ax[2].set_ylabel(r'$z$ (cMpc)')
        self.ax[2].set_xlim(xmin, xmax)
        self.ax[2].set_ylim(zmin, zmax)


        time = np.array([self.timestep['age'][int(hts)-1] for hts in progs.halo_ts])
        self.ax[3].scatter(time, progs.m*1e11, c=progs.halo_ts, s=sc, cmap='summer',
                           rasterized=True)
        self.ax[3].set_xlabel(r'$t$ (Gyr)')
        self.ax[3].set_ylabel(r'$M$ ($\mathrm{M}_{\odot}$)')
        self.ax[3].set_ylim(1e11*progs.m.min()/3., 1e11*progs.m.max()*3.)
        self.ax[3].set_xlim(time.min(), time.max())
        self.ax[3].set_yscale('log')


        # Draw some lines
        for progid in progs.halo_id:
            descid = progs.ix[progid].descendent_id
            if descid > 0:
                progenitor = progs.ix[progid]
                descendent = progs.ix[descid]
                px, py, pz = progenitor[['x', 'y', 'z']]
                dx, dy, dz = descendent[['x', 'y', 'z']]
                l0 = self.ax[0].plot([px, dx], [py, dy], lw=1, c='k', zorder=-1)
                l1 = self.ax[1].plot([pz, dz], [py, dy], lw=1, c='k', zorder=-1)
                l2 = self.ax[2].plot([px, dx], [pz, dz], lw=1, c='k', zorder=-1)





        title = "Halo #{t:d}\n(x, y, z) = ({x:3.3f}, {y:3.3f}, {z:3.3f})".format(t=int(halo.halo_num),
                                                                                 x=xh/sim['Lbox']+.5,
                                                                                 y=yh/sim['Lbox']+.5,
                                                                                 z=zh/sim['Lbox']+.5)

        suptitle = self.fig.suptitle(title, fontsize=18)
        if not self.fig.texts:
            self.fig.texts.append(suptitle)
        if pdffile:
            self.fig.savefig(pdffile, format='pdf', dpi=200)
        else:
            plt.savefig('halo_{:d}.png'.format(int(halo.halo_num)), dpi=100, format='png')

        # A bit of cleaning
        for axx in self.ax:
            axx.clear()
        self.fig.texts = []

        return hid

    def plot_accretion_history(self, hid, scale='mass'):
        """Plot halo accretion history.
        
        Blah
        """

        from matplotlib import cm

        # Define the selected halo
        halo = self.trees.ix[hid]
        sim = self.sim

        # Get progenitors
        progs = self._get_progenitors(hid)

        # Scalings
        scc = np.log10(progs.m)
        sc = (scc - scc.min())/(scc.max() - scc.min()) * 500.

        try:
            fig = self.fig
        except AttributeError:
            self.fig = plt.figure(figsize=(12, 12))
        try:
            ax = self.ax
        except AttributeError:
            self.ax = self.fig.add_subplot(111)


        time = np.array([self.timestep['age'][hts-1] for hts in progs.halo_ts])
        self.ax.scatter(time, progs.m*1e11, c=progs.halo_ts, s=sc, cmap='summer',
                           rasterized=True)
        self.ax.set_xlabel(r'$t$ (Gyr)')
        self.ax.set_ylabel(r'$M$ ($\mathrm{M}_{\odot}$)')
        self.ax.set_ylim(1e11*progs.m.min()/3., 1e11*progs.m.max()*3.)
        self.ax.set_xlim(time.min(), time.max())
        self.ax.set_yscale('log')

        self.fig.tight_layout()
        self.fig.savefig('halo_{:d}_history.pdf'.format(int(halo.halo_num)), dpi=200, format='pdf')

        return hid



    ### CONVENIENCE FUNCTIONS
    def _get_timestep_number(self):
        import glob, os.path
        halos_results = [os.path.basename(path)
                        for path in glob.glob(self.folder+'halos_results.*')]
        return len(halos_results)


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

        with open(self.folder+inputfile, 'r') as ifile:
            lines = ifile.readlines()

        outputs = []

        # We know that the name of the tree_bricks file should be "tree_bricksXXX"
        tbbasename = 'tree_bricks'
        # We can skip the first line, which should be "nsnaps 1"
        for path in lines[1:]:
            tbname = os.path.basename(path[1:-2])  # Remove those nasty "'" and \n
            assert tbname.startswith(tbbasename)
            outputs.append(int(tbname[len(tbbasename):]))

        return outputs

    def _read_timesteps_props(self, tsfile):
        with ff.FortranFile(tsfile) as ts:
            nsteps = ts.readInts()
            nhalos = ts.readInts()
            aexp = ts.readReals()
            age_univ = ts.readReals()

        return dict(nsteps=nsteps,
                    nhalos=nhalos,
                    aexp=aexp,
                    age=age_univ)

    def _read_tree_struct(self, tfile):
        ID_keys = ('bush_id', 'tree_id', 'halo_id', 'halo_num', 'halo_ts',
                   'first_prog', 'next_prog', 'descendent_id', 'last_prog',
                   'host_halo_id', 'host_sub_id', 'next_sub_id')
        IDs = pd.DataFrame(columns=ID_keys)
        with ff.FortranFile(tfile) as t:
            [nsteps, nIDs, nIndex] = t.readInts()
            nhalos = t.readInts()
            
            for ts in xrange(nsteps):
                if nhalos[ts] > 0:
                    IDs_raw = t.readInts('l').reshape((nhalos[ts], nIDs))
                    id_df = pd.DataFrame(IDs_raw, columns=ID_keys)
                    IDs = pd.concat((IDs, id_df))
                    t.readInts()  # Skip indexes
        return IDs

    def _read_halo_props(self, pfile):
        p_keys = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'r', 'spin',
                  'rvir', 'mvir', 'tvir', 'cvel', 'dmacc', 'frag',
                  'Lx', 'Ly', 'Lz', 'ep', 'ek', 'et')
        props = pd.DataFrame(columns=p_keys)
        
        with ff.FortranFile(pfile) as p:
            [nsteps, nprops] = p.readInts()
            nhalos = p.readInts()
            for ts in xrange(nsteps):
                if nhalos[ts]>0:
                    p_raw = p.readReals().reshape((nhalos[ts], nprops))
                    p_df = pd.DataFrame(p_raw, columns=p_keys)
                    props = pd.concat((props, p_df))
        return props


    def _get_progenitors(self, hid):
        target = self.trees.ix[hid]
        mask = ((self.trees.halo_id >= hid) &
                (self.trees.halo_id <= target['last_prog']))
        progenitors = self.trees.loc[mask].copy()
        return progenitors


    


