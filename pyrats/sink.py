#!/usr/bin/env pythn tqdm(files): n

"""
Module to deal with sinks
TODO: get boxlen to rescale and use in a similar way cosmo/ideal sim
"""
import pandas as pd
from tqdm import tqdm
import yt
import glob as glob
import os as os
from yt.frontends.ramses.io import convert_ramses_ages
from yt.utilities.logger import ytLogger as mylog

from . import analysis 

class Sinks(object):
    """
    Read the sinks outputs from RAMSES and put them in a PandaFrame
    The sink output has to be computed with BH.py (explanation of this
    works is given in BH.py)
    ID : [-1] read all sinks, [i,j] read sinks i and j only
    """

    def __init__(self, ID=[-1]):

        files = glob.glob('./sinks/BH?????.csv')
        files.sort()

        #snaps=glob.glob('output_*/info*')
        #ds=yt.load(snaps[0])
        #d=ds.all_data()
        #dx=float(ds.length_unit.in_units('pc')/2**ds.max_level*(1+ds.current_redshift))
        
        self.sink = [[] for _ in range(len(files))]
        self.sink[0] = [pd.read_csv(files[0])]
        if ID == [-1]:
            files = files[1:]
            ID = range(1, len(files)+1) 
        else:
            files = [files[i] for i in ID]

        j=0
        for f in files:
            self.sink[ID[j]] = pd.read_csv(f)
            j+=1

        return

    def distance(self, bhID1, bhID2):
        bh1 = self.sink[bhID1]
        bh2 = self.sink[bhID2]

        print('CARE, HAS TO BE TESTED IN COSMO SIMS')
        d = np.sqrt((bh1.x-bh2.x)**2 +
                 (bh1.y-bh2.y)**2 +
                 (bh1.z-bh2.z)**2)
        return d

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

    def plot_sink_dynamics(self, bhid=[0], loc='./', IDhalos=[0], Galaxy=False, timestep=None,
                limrho=None, limv=None, limf=None, limt=None,logDist=True, average=5):
        """
        Show on a same PDF distance, surrounding gas/stars/dm density, relative velocity and magnitude
        of the drag force
        
        bhid : list of BHs to analyse, [0] is all
        loc : folder where to save the pdf
        IDhalos : list of halos to give to pyrats.analysis.dist_sink_to_halo
        Galaxy : if true, consider ID of galaxies instead of halos
        timestep : timestep at which consider halos/galaxies
        limrho v f : y limits for the plot
        logDist : force log plot or not for the distance
        average (5 Myr default) : window (in Myr) to average rho / v / a
        
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
                timestep = 0

        j=-1
        for i in tqdm(tmp):
            j +=1
            plt.clf()
            sink = self.sink[i + 1]

            dt_mean = np.diff(sink.t).mean()
            n_av = int(average // (dt_mean*1000))
            n_max = len(sink.t) // n_av *n_av-1

            plt.figure()
            plt.subplot(221)

            d, t=analysis.dist_sink_to_halo(IDsink=[i+1], IDhalos=[IDhalos[j]], Galaxy=Galaxy, timestep=timestep)
            if len(t[0]) > 1000:
                d=[d[0][::len(t[0])//1000]]
                t=[t[0][::len(t[0])//1000]]
            if (logDist):
                plt.semilogy(t[0], d[0])
            else:    
                plt.plot(t, d)
            plt.plot([t[0].min()],[d[0].min()], color='C0', label='Gas')
            plt.plot([t[0].min()],[d[0].min()], color='green', label='Stars')
            plt.plot([t[0].min()],[d[0].min()], color='orange', label='DM')
            plt.legend(loc='best')
            #plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('D$_\\mathrm{halo}$ [kpc]')
            if Galaxy:
                plt.ylabel('D$_\\mathrm{galaxy}$ [kpc]')
            if limt is not None:
                plt.xlim(limt[0], limt[1])

            plt.subplot(222)
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.rho.loc[:n_max]).reshape(-1, n_av)).mean(-1))
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.rho_stars.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                label='total', color='g', linestyle='--')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.rho_lowspeed_stars.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                label='low speed', color='g')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.rho_dm.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                 color='orange', linestyle='--')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.rho_lowspeed_dm.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                color='orange')
            #plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('$\\rho$ [part cc$^{-1}$]')
            plt.legend()
            if limrho is not None:
                plt.ylim(limrho[0], limrho[1])
            if limt is not None:
                plt.xlim(limt[0], limt[1])

            plt.subplot(223)
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.dv.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                label='$\Delta$v gas')
            #plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
            #    (np.array(sink.cs.loc[:n_max]).reshape(-1, n_av)).mean(-1),
            #    label='$c_s$', color='red')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.vsink_rel_stars.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                label='$\\Delta$v$_\\star$', color='g')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.vsink_rel_dm.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                label='$\\Delta$v$_\\mathrm{DM}$', color='orange')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('v [km s$^{-1}$]')
            if limv is not None:
                plt.ylim(limv[0], limv[1])
            if limt is not None:
                plt.xlim(limt[0], limt[1])

            plt.subplot(224)

            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.a_gas.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                )
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.a_stars_slow.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                color='g')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.a_stars_fast.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                color='lightgreen', label='Stars high speed')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.a_dm_slow.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                color='orange')
            plt.semilogy((np.array(sink.t.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                (np.array(sink.a_dm_fast.loc[:n_max]).reshape(-1, n_av)).mean(-1),
                color='red', label='DM high speed')
            plt.xlabel('Age of the universe [Gyr]')
            plt.ylabel('|$\\vec{a}$| [km/s Myr$^{-1}$]')
            plt.legend()
            plt.suptitle('BH #{:03}'.format(i+1)+' Halo #{:04} in output_{:05}'.format(IDhalos[j], timestep))
            if Galaxy:
                plt.suptitle('BH #{:03}'.format(i+1)+' Galaxy #{:04} in output_{:05}'.format(IDhalos[j], timestep))
            if limf is not None:
                plt.ylim(limf[0], limf[1])
            if limt is not None:
                plt.xlim(limt[0], limt[1])

            if Galaxy:
                plt.savefig(loc + '/BHdynamics/BH{:03}_Galaxy{:04}_ts{:03}.pdf'.format(i+1, IDhalos[j], timestep))
            else:
                plt.savefig(loc + '/BHdynamics/BH{:03}_Halo{:04}_ts{:03}.pdf'.format(i+1, IDhalos[j], timestep))
            plt.clf()
            plt.close('all')
        plt.rcParams.update({'font.size': 10})
        return


def get_sinks(ds):
    columns_name = ['ID', 'M', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'age', 'Mdot']

    sink_files = os.path.join(ds.files,'sink_{:05}'.format(ds.ids))
    if os.path.isfile(sink_file):
        sink = pd.read_csv(sink_file, names=columns_name)
    else:
        sink = pd.DataFrame(columns = columns_name)
        mylog.info('Did not find any sink file')

    if len(sink.ID) > 0:
        sink['M'] = sink.M * (ds.arr(1, 'code_mass').in_units('Msun'))
        sink['Mdot'] = sink.Mdot*(ds.arr(1, 'code_mass/code_time').in_units('Msun/yr'))
        if ds.cosmological_simulation==1:
            sink.age = np.copy(
                ds.arr(convert_ramses_ages(
                ds ,np.copy(sink.age))))
        else:
            sink.age =sink.age*ds.arr(1,'code_time').in_units('Myr')
        sink.vx = sink.vx * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.vy = sink.vy * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.vz = sink.vz * (ds.arr(1, 'code_velocity').in_units('km / s'))
        sink.x = sink.x / ds['boxlen'] 
        sink.y = sink.y / ds['boxlen'] 
        sink.z = sink.z / ds['boxlen'] 

        sink_file = os.path.isfile(ds.prefix , 'matching', 'BH_average', '{}'.format(ds.ids))
        if os.path.isfile(sink_file):
            dummy = pd.read_hdf(sink_file)
            sink = pd.concat([dummy, sink], axis=1)
        else:
            mylog.info('Did not found averaged quantities, keep going anyway')

    else:
        mylog.info('Sink file found,but it is empty')

    sink['galID'] = -1
    sink['mgal'] = 0 ; sink['mbulge'] = 0
    sink['sigma_bulge'] = 0 ; sink['mhalo'] = 0

    return sink
