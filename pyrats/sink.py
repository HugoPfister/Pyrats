#!/usr/bin/env python

"""Module to deal with sinks 
"""
import matplotlib
# matplotlib.use('PDF')
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import yt
import glob as glob
import os as os
from yt.utilities.lib.cosmology_time import get_ramses_ages
class Sinks(object):
    """
    Read the sinks outputs from RAMSES and put them in a PandaFrame
    """

    def __init__(self):

        folder = glob.glob('output*/info*')
        folder.sort()
        ds = yt.load(folder[0])
        sim = ds
        _sim = {}
        _sim['h'] = float(ds.cosmology.hubble_constant)
        _sim['Om'] = ds.omega_matter
        _sim['Ol'] = ds.omega_lambda
        _sim['Lbox'] = float(ds.length_unit.in_units('Mpccm'))

        self.sim = _sim
        self.ds = ds

        files = glob.glob('./sinks/BH*')
        files.sort()

        self.sink = []
        self.sink += [pd.read_csv(f) for f in files]

        return

    def plot_sink_accretion(self, bhid=0, loc='./', limM=None,
                            limrho=None, limv=None, limdM=None):
        """
        Put sink Mass, sound speed, relative velocity, bondi,
        Eddington, simulation accretion as a function of time in a PDF
        Parameters
        ----------
        * bhid (0): 0 for all sinks
                    [i,j...] for BH #i #j #...
        * loc ('./'): Where to save those PDFs
        * limM rho v dM (None): boundy for the plots e.g. [1e-1, 1e3]
        """
        plt.rcParams.update({'font.size': 7})
        # plt.rcParams.update({'lines.linewidth':2})
        # plt.rcParams.update({'lines.markersize':5})
        os.system('mkdir ' + loc + '/BH')
        if bhid == 0:
            tmp = range(len(self.sink) - 1)
        else:
            tmp = np.copy(bhid) - 1

        for i in tqdm(tmp):
            sink = self.sink[i + 1]
            r = sink.M.max() / sink.M.min()

            plt.figure()
            plt.subplot(221)
            if r > 10:
                plt.semilogy(sink.t, sink.M)
            else:
                plt.plot(sink.t, sink.M)
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('M$_\\bullet$ [M$_\\odot$]')
            if limM is not None:
                plt.ylim(limM[0], limM[1])

            plt.subplot(222)
            plt.semilogy(sink.t, sink.rho)
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('$\\rho$ [part cc$^{-1}$]')
            if limrho is not None:
                plt.ylim(limrho[0], limrho[1])

            plt.subplot(223)
            plt.semilogy(sink.t, sink.cs, label='c$_s$')
            plt.semilogy(sink.t, sink.dv, label='$\Delta$v')
            plt.legend(loc='lower left')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('c$_s$ & $\Delta$v [km s$^{-1}$]')
            if limv is not None:
                plt.ylim(limv[0], limv[1])

            plt.subplot(224)
            plt.semilogy(sink.t, sink.dME, label='Eddington')
            plt.semilogy(sink.t, sink.dMB, label='Bondi')
            plt.semilogy(sink.t, sink.dM, linestyle='--', label='Simulation')
            plt.legend(loc='lower right')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('dM/dt [M$_\\odot$ yr$^{-1}$]')
            if limdM is not None:
                plt.ylim(limdM[0], limdM[1])

            plt.suptitle('BH #{:03}'.format(i + 1))

            plt.savefig(loc + '/BH/BH{:03}'.format(i + 1) + '.pdf')
        plt.rcParams.update({'font.size': 10})
        return


def get_sinks(ds):
    columns_name = ['ID', 'M', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'age', 'Mdot']
    sink = pd.read_csv(
        'output' + str(ds)[-6:] + '/sink' + str(ds)[-6:] + '.csv',
        names=columns_name)
    if len(sink.ID) > 0:
        sink['M'] = sink.M * (ds.arr(1, 'code_mass').in_units('Msun'))
        sink['Mdot'] = sink.M * \
            (ds.arr(1, 'code_mass/code_time').in_units('Msun/yr'))
        sink.age = np.copy(
            ds.arr(get_ramses_ages(
                ds.t_frw, ds.tau_frw, ds.dtau,
                ds.time_simu,
                1. / (ds.hubble_constant * 100 * 1e5 / 3.08e24)
                / ds['unit_t'],
                ds['time'] - np.copy(sink.age),
                ds.n_frw / 2, len(sink.age)), 'code_time').in_units('Myr'))
        sink.vx = sink.vx * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.vy = sink.vy * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.vz = sink.vz * (ds.arr(1, 'code_velocity').in_units('km / s'))
    return sink
