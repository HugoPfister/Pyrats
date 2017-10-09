#!/usr/bin/env python

"""
Module to deal with sinks
TODO: get boxlen to rescale and use in a similar way cosmo/ideal sim
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
            plt.clf()
            sink = self.sink[i + 1]
            r = sink.M.max() / sink.M.min()

            if len(sink.t) > 1000:
                sink=sink.loc[::len(sink.t)//1000]

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
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('c$_s$ & $\Delta$v [km s$^{-1}$]')
            if limv is not None:
                plt.ylim(limv[0], limv[1])

            plt.subplot(224)
            plt.semilogy(sink.t, sink.dME, label='Eddington')
            plt.semilogy(sink.t, sink.dMB, label='Bondi')
            plt.semilogy(sink.t, sink.dM, linestyle='--', label='Simulation')
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('dM/dt [M$_\\odot$ yr$^{-1}$]')
            if limdM is not None:
                plt.ylim(limdM[0], limdM[1])

            plt.suptitle('BH #{:03}'.format(i + 1))

            plt.savefig(loc + '/BH/BH{:03}'.format(i + 1) + '.pdf')
            plt.clf()
        plt.rcParams.update({'font.size': 10})
        return

    def plot_sink_dynamics(self, bhid=0, loc='./', center=[0.5,0.5,0.5],
                limrho=None, limv=None, limf=None):
        """
        """
        files=glob.glob('output_*/info*')
        ds=yt.load(files[0])
        d=ds.all_data()
        dx=float(d[('index', 'dx')].min().in_units('pc'))

        plt.rcParams.update({'font.size': 7})
        os.system('mkdir ' + loc + '/BHdynamics')
        if bhid == 0:
            tmp = range(len(self.sink) - 1)
        else:
            tmp = np.copy(bhid) - 1

        for i in tqdm(tmp):
            plt.clf()
            sink = self.sink[i + 1]

            if len(sink.t) > 1000:
                sink=sink.loc[::len(sink.t)//1000]

            plt.figure()
            plt.subplot(221)
            d=ds.arr(np.copy(np.sqrt((sink.x-center[0])**2+(sink.y-center[1])**2+(sink.z-center[2])**2)), 'code_length')
            if d.max()/d.min() > 50 :
                plt.semilogy(sink.t, d.in_units('kpc'))
            else:    
                plt.plot(sink.t, d.in_units('kpc'))
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('Distance [kpc]')

            plt.subplot(222)
            plt.semilogy(sink.t, sink.rho, label='Gas')
            plt.semilogy(sink.t, sink.rho_part, label='DM+stellar density')
            plt.semilogy(sink.t, sink.rho_part*sink.frac_star, label='Stellar DF density')
            plt.semilogy(sink.t, sink.rho_part*sink.frac_dm, label='DM DF density')
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('$\\rho$ [part cc$^{-1}$]')
            if limrho is not None:
                plt.ylim(limrho[0], limrho[1])

            plt.subplot(223)
            plt.semilogy(sink.t, sink.cs, label='c$_s$')
            plt.semilogy(sink.t, sink.dv, label='$\Delta$v gas')
            plt.semilogy(sink.t, sink.v_part, label='$\Delta$v DM+stars')
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('velocity [km s$^{-1}$]')
            if limv is not None:
                plt.ylim(limv[0], limv[1])

            plt.subplot(224)
            if sink.v_part.max() > 3e5 : print('fix PYRATS and compute Lambda w/ max(Rsh, b90)')
            a_stars=4*np.pi*(6.67e-8)**2*sink.M*2e33*sink.frac_star*sink.rho_part*1.67e-24*np.log(dx/(6.67e-8*sink.M*2e33/(sink.v_part*1e5)**2/3.08e18))/(sink.v_part*1e5)**2*(3600*24*365*1e6)/1e5
            a_dm=4*np.pi*(6.67e-8)**2*sink.M*2e33*sink.frac_dm*sink.rho_part*1.67e-24*np.log(dx/(6.67e-8*sink.M*2e33/(sink.v_part*1e5)**2/3.08e18))/(sink.v_part*1e5)**2*(3600*24*365*1e6)/1e5
            
            fudge=[]
            for isink in sink.index:
                M=sink.loc[isink].dv/sink.loc[isink].cs
                if M < 0.95 : fudge+=[1/M**2*(0.5*np.log((1+M)/(1-M)) - M)]
                if ((M >= 0.95) & (M <= 1.007)) : fudge+=[1]
                if (M > 1.007) : fudge+=[1/M**2*(0.5*np.log(M**2-1) + 3.2)]

            a_gas=4*np.pi*(6.67e-8)**2*sink.M*2e33*sink.rho*1.67e-24/(sink.cs*1e5)**2*fudge*(3600*24*365*1e6)/1e5
            plt.semilogy(sink.t, a_stars, label='DF stars')
            plt.semilogy(sink.t, a_dm, label='DF DM')
            plt.semilogy(sink.t, a_gas, label='DF gas')
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('Force [km/s Myr$^{-1}$]')
            if limf is not None:
                plt.ylim(limf[0], limf[1])



            plt.savefig(loc + '/BHdynamics/BH{:03}'.format(i + 1) + '.pdf')
            plt.clf()
        plt.rcParams.update({'font.size': 10})
        return


def get_sinks(ds):
    columns_name = ['ID', 'M', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'age', 'Mdot']

    SinkFile = 'output' + str(ds)[-6:] + '/sink' + str(ds)[-6:] + '.csv'

    if os.path.isfile(SinkFile):
        sink = pd.read_csv(SinkFile, names=columns_name)
    else:
        sink = pd.DataFrame(columns = columns_name)

    if len(sink.ID) > 0:
        sink['M'] = sink.M * (ds.arr(1, 'code_mass').in_units('Msun'))
        sink['Mdot'] = sink.Mdot * \
            (ds.arr(1, 'code_mass/code_time').in_units('Msun/yr'))
        if ds.cosmological_simulation==1:
            sink.age = np.copy(
            ds.arr(get_ramses_ages(
                ds.t_frw, ds.tau_frw, ds.dtau,
                ds.time_simu,
                1. / (ds.hubble_constant * 100 * 1e5 / 3.08e24)
                / ds['unit_t'],
                ds['time'] - np.copy(sink.age),
                ds.n_frw / 2, len(sink.age)), 'code_time').in_units('Myr'))
        else:
            sink.age =sink.age*ds.arr(1,'code_time').in_units('Myr')
        sink.vx = sink.vx * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.vy = sink.vy * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.vz = sink.vz * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.x = sink.x / ds['boxlen'] 
        sink.y = sink.y / ds['boxlen'] 
        sink.z = sink.z / ds['boxlen'] 
    return sink
