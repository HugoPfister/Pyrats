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
from scipy.interpolate import interp1d

from . import analysis 
class Sinks(object):
    """
    Read the sinks outputs from RAMSES and put them in a PandaFrame
    The sink output has to be computed with BH.py (explanation of this
    works is given in BH.py)
    """

    def __init__(self, ExtraProps=False, center='none'):

        files = glob.glob('./sinks/BH*')
        files.sort()

        snaps=glob.glob('output_*/info*')
        ds=yt.load(snaps[0])
        d=ds.all_data()
        dx=float(ds.length_unit.in_units('pc')/2**ds.max_level*(1+ds.current_redshift))

        self.sink = [pd.read_csv(files[0])]
        j=1
        for f in tqdm(files[1:]):
            self.sink += [pd.read_csv(f)]

            if 'fact_stars' in self.sink[j].columns:
                self.sink[j]['v_part'] = np.sqrt(self.sink[j].vx_part**2+self.sink[j].vy_part**2+self.sink[j].vz_part**2)
                self.sink[j]['vsink_rel'] = np.sqrt((self.sink[j].vx_part-self.sink[j].vx)**2+(self.sink[j].vy_part-self.sink[j].vy)**2+(self.sink[j].vz_part-self.sink[j].vz)**2)
                self.sink[j]['rinf'] = (self.sink[j].M/1e7)/(self.sink[j].vsink_rel/200)**2
                CoulombLog = np.maximum(np.zeros(len(self.sink[j].t)), np.log(4*dx/self.sink[j].rinf))
                self.sink[j]['a_stars_slow']=4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].frac_stars*self.sink[j].rho_stars*1.67e-24*CoulombLog/(self.sink[j].vsink_rel*1e5)**2*(3600*24*365*1e6)/1e5
                self.sink[j]['a_dm_slow']  =4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].frac_dm*self.sink[j].rho_dm*1.67e-24*CoulombLog/(self.sink[j].vsink_rel*1e5)**2*(3600*24*365*1e6)/1e5
                CoulombLog = np.minimum(np.zeros(len(self.sink[j].rinf)), self.sink[j].rinf - 4*dx) / (self.sink[j].rinf - 4*dx)  
                self.sink[j]['a_stars_fast']=4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].fact_stars*1.67e-24*CoulombLog/(self.sink[j].vsink_rel*1e5)**2*(3600*24*365*1e6)/1e5
                self.sink[j]['a_dm_fast']=4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].fact_dm*1.67e-24*CoulombLog/(self.sink[j].vsink_rel*1e5)**2*(3600*24*365*1e6)/1e5
                self.sink[j]['a_dm'] = self.sink[j]['a_dm_slow']+self.sink[j]['a_dm_fast']
                self.sink[j]['a_stars'] = self.sink[j]['a_stars_slow']+self.sink[j]['a_stars_fast']

            if 'dv_x' in self.sink[j].columns:
                self.sink[j]['vsink_rel'] = np.sqrt(self.sink[j].vx_part**2+self.sink[j].vy_part**2+self.sink[j].vz_part**2)
                self.sink[j]['dv_drag'] = np.sqrt(self.sink[j].dv_x**2+self.sink[j].dv_y**2+self.sink[j].dv_z**2)
                self.sink[j]['rinf'] = (self.sink[j].M/1e7)/(self.sink[j].vsink_rel/200)**2
                CoulombLog = np.maximum(np.zeros(len(self.sink[j].t)), np.log(4*dx/self.sink[j].rinf))
                self.sink[j]['a_stars_slow']=4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].frac_stars*self.sink[j].rho_stars*1.67e-24*CoulombLog/(self.sink[j].vsink_rel*1e5)**2*(3600*24*365*1e6)/1e5
                self.sink[j]['a_dm_slow']=4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].frac_dm*self.sink[j].rho_dm*1.67e-24*CoulombLog/(self.sink[j].vsink_rel*1e5)**2*(3600*24*365*1e6)/1e5

            M=self.sink[j].dv/self.sink[j].cs
            self.sink[j]['rinf_gas'] = (self.sink[j].M/1e7)/((self.sink[j].dv**2+self.sink[j].cs**2)/200**2)
            CoulombLog = np.minimum(np.zeros(len(self.sink[j].rinf_gas)), self.sink[j].rinf_gas - 4*dx) / (self.sink[j].rinf_gas - 4*dx)
            fudge = M
            fudge.loc[M < 0.95] = 1/M**2*(0.5*np.log((1+M)/(1-M)) - M)
            fudge.loc[(M >= 0.95) & (M <= 1.007)] = 1
            fudge.loc[M > 1.007] = 1/M**2*(0.5*np.log(M**2-1) + 3.2)
            self.sink[j]['a_gas']=4*np.pi*(6.67e-8)**2*self.sink[j].M*2e33*self.sink[j].rho*1.67e-24/(self.sink[j].cs*1e5)**2*fudge*(3600*24*365*1e6)/1e5*CoulombLog
       
            j+=1

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
            plt.semilogy(sink.t, sink.dv, label='$\Delta$v', alpha=0.9)
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('c$_s$ & $\Delta$v [km s$^{-1}$]')
            if limv is not None:
                plt.ylim(limv[0], limv[1])

            plt.subplot(224)
            plt.semilogy(sink.t, sink.dMB, label='Bondi')
            plt.semilogy(sink.t, sink.dM, linestyle='--', label='Simulation', alpha=0.9)
            plt.semilogy(sink.t, sink.dME, label='Eddington')
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('dM/dt [M$_\\odot$ yr$^{-1}$]')
            if limdM is not None:
                plt.ylim(limdM[0], limdM[1])

            plt.suptitle('BH #{:03}'.format(i + 1))

            plt.savefig(loc + '/BH/BH{:03}'.format(i + 1) + '.pdf')
            plt.close('all')
        plt.rcParams.update({'font.size': 10})
        return

    def plot_sink_dynamics(self, bhid=[0], loc='./', IDhalos=[0],
                limrho=None, limv=None, limf=None, logDist=True):
        """
        Show on a same PDF distance, surrounding gas/stars/dm density, relative velocity and magnitude
        of the drag force
        
        bhid : list of BHs to analyse, [0] is all
        loc : folder where to save the pdf
        IDhalos : list of halos to give to pyrats.analysis.dist_sink_to_halo
        limrho v f : y limits for the plot
        logDist : force log plot or not for the distance

        """
        files=glob.glob('output_*/info*')
        ds=yt.load(files[0])
        d=ds.all_data()
        dx=float(ds.length_unit.in_units('pc')/2**ds.max_level*(1+ds.current_redshift))

        plt.rcParams.update({'font.size': 7})
        os.system('mkdir ' + loc + '/BHdynamics')
        if bhid == [0]:
            tmp = range(len(self.sink) - 1)
        else:
            tmp = np.copy(bhid) - 1
        
        if IDhalos == [0]:
            if ds.cosmological_simulation:
                print('Please fill a halo for each BHs you have put')
                return 
            else:
                IDhalos=len(bhid)*[-1]

        j=-1
        for i in tqdm(tmp):
            j +=1
            plt.clf()
            sink = self.sink[i + 1]

            if len(sink.t) > 1000:
                sink=sink.loc[::len(sink.t)//1000]

            plt.figure()
            plt.subplot(221)

            d, t=analysis.dist_sink_to_halo(IDsink=[i+1], IDhalos=[IDhalos[j]])
            if len(t[0]) > 1000:
                d=d[0][::len(t[0])//1000]
                t=t[0][::len(t[0])//1000]
            if (logDist):
                plt.semilogy(t, d)
            else:    
                plt.plot(t, d)
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('Distance [kpc]')

            plt.subplot(222)
            plt.semilogy(sink.t, sink.rho, label='Gas')
            if 'rho_stars' in sink.columns:
                plt.semilogy(sink.t, sink.rho_stars, label='Stellar density', color='g')
                plt.semilogy(sink.t, sink.rho_dm, label='DM density', color='orange')
                plt.semilogy(sink.t, sink.rho_stars*sink.frac_stars, color='g', linestyle=':', alpha=0.7)
                plt.semilogy(sink.t, sink.rho_dm*sink.frac_dm, linestyle=':', color='orange', alpha=0.7)
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('$\\rho$ [part cc$^{-1}$]')
            if limrho is not None:
                plt.ylim(limrho[0], limrho[1])

            plt.subplot(223)
            plt.semilogy(sink.t, sink.dv, label='$\Delta$v gas', alpha=0.8)
            plt.semilogy(sink.t, sink.cs, label='c$_s$', color='red')
            if 'vsink_rel' in sink.columns:
                plt.semilogy(sink.t, sink.vsink_rel, label='$\Delta$v DM+stars', alpha=0.8, color='orange')
            
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('velocity [km s$^{-1}$]')
            if limv is not None:
                plt.ylim(limv[0], limv[1])

            plt.subplot(224)

            plt.semilogy(sink.t, sink.a_gas, label='Gas')
            if 'vsink_rel' in sink.columns:    
                plt.semilogy(sink.t, sink.a_stars_slow, label='Stars slow', alpha=0.8, color='green')
                plt.semilogy(sink.t, sink.a_dm_slow, alpha=0.8, color='orange', label='DM slow')
            if 'a_stars_fast' in sink.columns:
                plt.semilogy(sink.t, np.copy(sink.a_stars_fast), linestyle =':', color='green', alpha=1)
                plt.semilogy(sink.t, np.copy(sink.a_dm_fast), alpha=1, linestyle=':', color='orange')
            plt.legend(loc='best')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('Acceleration [km/s Myr$^{-1}$]')
            plt.suptitle('BH #{:03}'.format(i+1)+' Halo #{:04}'.format(IDhalos[j]))
            if limf is not None:
                plt.ylim(limf[0], limf[1])



            plt.savefig(loc + '/BHdynamics/BH{:03}'.format(i + 1) + '.pdf')
            plt.clf()
            plt.close('all')
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
