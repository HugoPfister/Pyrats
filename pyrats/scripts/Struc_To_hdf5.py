# This script converts files from the HaloFinder
# into hdf5 readable files.

#It also computes the Sersic index of all structures

# Not all quantities are dumped, should be modified depending on what is needed
# and the version of the HaloFinder

import numpy as np
import pandas as pd
import yt.utilities.fortran_utils as fpu
from yt.utilities.logger import ytLogger as mylog
from yt.funcs import get_pbar
import yt
import os
from scipy.optimize import curve_fit
import pyrats

def main():
    output_folder = '../Outputs'
    tree_brick_folder = './AdaptaHOP'


    if not os.path.exists('hdf5'):
        os.system('mkdir hdf5')
        mylog.info('Making the folder hdf5')
    
    files = pyrats.utils.find_outputs(output_folder)
    files.sort()
    mylog.info('Found {} outputs from {} to {}'.format(len(files), files[0], files[-1]))

    for f in files:
        GalList(int(f[-9:-4]), contam=False, tree_brick_folder=tree_brick_folder)
    #f = files[130]
    #GalList(int(f[-9:-4]), contam=False, tree_brick_folder=tree_brick_folder)

    return

class GalList(object):
    def __init__(self, iout, tree_brick_folder, contam=False):
        self.iout = iout
        filename = os.path.join(tree_brick_folder,'tree_bricks{:03}'.format(self.iout))
        print(filename)
        self.gal = self._read_halos(contam,filename)
        if self.gal.index.size > 0:    
            self.gal.to_hdf(
                    './hdf5/tree_bricks{:03d}.hdf'.format(self.iout), 'hdf5')


    # Convenience functions
    def _read_halos(self, contam, filename, prec='d', longint=True):
        halo_keys = ['ID', 'nbpart', 'level', 'min_part_id',
                     'host', 'hostsub', 'nbsub', 'nextsub',
                     'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz',
                     'a', 'b', 'c', 'ek', 'ep', 'et', 'rho0', 'r_c',
                     'spin', 'm', 'ntot', 'mtot',
                     'r', 'mvir', 'rvir', 'tvir', 'cvel',
                     'rvmax', 'vmax', 'cNFW',
                     'r200','m200',
                     'r50', 'r90', 'sigma',
                     'nDM', 'nstar', 'mDM', 'mstar',
                     'ntotDM', 'ntotstar',
                     'mtotDM', 'mtotstar',
                     'xDM', 'yDM', 'zDM', 'xstar', 'ystar', 'zstar',
                     'vxDM', 'vyDM', 'vzDM', 'vxstar', 'vystar', 'vzstar',
                     'rDM', 'rstar',
                     'aDM', 'bDM', 'cDM', 'astar', 'bstar', 'cstar',
                     'sigmaDM', 'sigmastar',
                     'reff', 'Zstar', 'tstar',
                     'sfr10', 'sfr100', 'sfr1000',
                     'r50DM', 'r90DM', 'r50star', 'r90star',
                     'Vsigma', 'sigma1D',
                     'Vsigma_disc', 'sigma1D_disc',
                     'sigma_bulge', 'mbulge',
                     'n_sersic']
                     #WE DO NOT STORE THE PROFILES
                     #'rr', 'rho', 'rr3D', 'rho3D',
                     #'rr3DDM', 'rho3DDM', 'rr3Dstar', 'rho3Dstar',
                     #]
    
        if contam:
            halo_keys.append('contam')
            halo_keys.append('mcontam')
            halo_keys.append('mtotcontam')
            halo_keys.append('ncontam')
            halo_keys.append('ntotcontam')
        iprec = 'q' if longint else 'i'
        
        data = np.empty(shape=(0, len(halo_keys)), dtype=object)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                [npart] = fpu.read_vector(f, iprec)
                [massp] = fpu.read_vector(f, prec)
                [aexp] = fpu.read_vector(f, prec)
                [omega_t] = fpu.read_vector(f, prec)
                [age] = fpu.read_vector(f, prec)
                [nhalos, nsubs] = fpu.read_vector(f, 'i')
    
                # Save the age/aexp, the mass of the particle,
                # as well as the number of (sub)halos
                data = np.empty(shape=(nhalos + nsubs, len(halo_keys)), dtype=object)
    
                mylog.info('Brick: groups       : %s' % nhalos)
                mylog.info('Brick: sub group   : %s' % nsubs)
                mylog.info('Brick: aexp        : %s' % aexp)
    
                pbar = get_pbar('', nhalos+nsubs)
                for ihalo in range(nhalos + nsubs):
                    [nbpart] = fpu.read_vector(f, iprec)  # Number of particles
                    listp = fpu.read_vector(f, iprec)  # List of particles IDs
                    listm = fpu.read_vector(f, prec)  # List of particles masses
                    listf = fpu.read_vector(f, 'b')  # List of particles families
                    [ID] = fpu.read_vector(f, 'i')  # Halo ID
                    fpu.skip(f, 1) # Skip timestep
                    [level, host, hostsub, nbsub, nextsub] = fpu.read_vector(f, 'i')
                    [m] = fpu.read_vector(f, prec)  # Total mass
                    [ntot] = fpu.read_vector(f, iprec)  # Total number of particles
                    [mtot] = fpu.read_vector(f, prec)  # Total mass + subs
                    [x, y, z] = fpu.read_vector(f, prec)  # Center
                    [vx, vy, vz] = fpu.read_vector(f, prec)  # Velocity
                    [Lx, Ly, Lz] = fpu.read_vector(f, prec)  # Angular momentum
                    [r, a, b, c] = fpu.read_vector(f, prec)  # Shape (ellipticity)
                    [ek, ep, et] = fpu.read_vector(f, prec)  # Energetics
                    [spin] = fpu.read_vector(f, prec)  # Total angular momentum
                    [sigma] = fpu.read_vector(f, prec) # 3D velocity dispersion
                    [rvir, mvir, tvir, cvel] = fpu.read_vector(f, prec)  # Virial parameters
                    [rvmax, vmax] = fpu.read_vector(f, prec)  # RVmax and Vmax
                    [cNFW] = fpu.read_vector(f, prec)  # NFW concentration from Prada+2012
                    [r200, m200] = fpu.read_vector(f, prec)  # R200 and M200
                    [r50, r90] = fpu.read_vector(f, prec)  # R50 and R90
                    rr3D = fpu.read_vector(f, prec)  # Radial bins
                    rho3D = fpu.read_vector(f, prec)  # 3D density profile
                    [rho0, r_c] = fpu.read_vector(f, prec)  # ?
                    # Stellar-only properties
                    [reff] = fpu.read_vector(f, prec)  # Effective radius
                    [Zstar] = fpu.read_vector(f, prec)  # Metallicity
                    [tstar] = fpu.read_vector(f, prec)  # Age
                    [sfr10, sfr100, sfr1000] = fpu.read_vector(f, prec)  # SFR
                    [Vsigma, sigma1D] = fpu.read_vector(f, prec)  # V/sigma and sigma1D
                    [Vsigma_disc, sigma1D_disc] = fpu.read_vector(f, prec)  # V/sigma and sigma1D for the disc
                    [sigma_bulge, mbulge] = fpu.read_vector(f, prec)  # Bulge properties
                    # Stellar surface density profile
                    fpu.skip(f, 1) # number of bins
                    rr = fpu.read_vector(f, prec)  # Radial bins
                    rho = fpu.read_vector(f, prec)  # Surface density profile
                    # fpu.skip(f, 1)
                    # fpu.skip(f, 1)
    
                    # DM vs stars quantities
                    [ndm, nstar] = fpu.read_vector(f, iprec)  # Nb of particles
                    [mdm, mstar] = fpu.read_vector(f, prec)  # Masses
                    [ntotdm, ntotstar] = fpu.read_vector(f, iprec)  # Nb of particles with substructures
                    [mtotdm, mtotstar] = fpu.read_vector(f, prec)  # Masses with substructures
                    [xdm, ydm, zdm] = fpu.read_vector(f, prec)  # Halo centre (DM)
                    [xstar, ystar, zstar] = fpu.read_vector(f, prec)  # Halo centre (stars)
                    [vxdm, vydm, vzdm] = fpu.read_vector(f, prec)  # Halo velocity (DM)
                    [vxstar, vystar, vzstar] = fpu.read_vector(f, prec)  # Halo velocity (stars)
                    [Lxdm, Lydm, Lzdm] = fpu.read_vector(f, prec)  # Angular momentum (DM)
                    [Lxstar, Lystar, Lzstar] = fpu.read_vector(f, prec)  # Angular momentum (stars)
                    [rdm, adm, bdm, cdm] = fpu.read_vector(f, prec)  # Shape (DM)
                    [rstar, astar, bstar, cstar] = fpu.read_vector(f, prec)  # Shape (stars)
                    [r50dm, r90dm] = fpu.read_vector(f, prec)  # R50 and R90 (DM)
                    [r50star, r90star] = fpu.read_vector(f, prec)  # R50 and R90 (stars)
                    #rr3Ddm = fpu.read_vector(f, prec)  # Radial bins
                    #rho3Ddm = fpu.read_vector(f, prec)  # 3D density profile
                    #rr3Dstar = fpu.read_vector(f, prec)  # Radial bins
                    #rho3Dstar = fpu.read_vector(f, prec)  # 3D density profile
                    fpu.read_vector(f, prec)  #dummy 
                    fpu.read_vector(f, prec)  #dummy 
                    fpu.read_vector(f, prec)  #dummy 
                    fpu.read_vector(f, prec)  #dummy 
                    [sigmadm, sigmastar] = fpu.read_vector(f, prec)  # Velocity dispersions
    
                    if contam:
                        [contamlevel] = fpu.read_vector(f, 'i')  # Contamination
                        [mcontam, mtotcontam] = fpu.read_vector(f, prec)  # Mass of contaminated particles
                        [ncontam, ntotcontam] = fpu.read_vector(f, iprec)  # Mass of contaminated particles
                    
                    n_sersic = 0
                    arg = rho > 0 
                    if True in arg:
                        #fit with a Sersic profile
                        (_,n_sersic),pcov = curve_fit(lambda r,I0,n: I0-(2*n-1/3)*(r/reff/1e3)**(1/n),
                            rr[arg], np.log(rho[arg]),
                            p0 = [np.log(mstar/reff**2*1e6), 1],
                            bounds = ([-np.inf,0.5], [np.inf,20]), maxfev = 10000)
                    
                    halodata = [ID, nbpart, level, listp.min(),
                                host, hostsub, nbsub, nextsub,
                                x, y, z, vx, vy, vz, Lx, Ly, Lz,
                                a, b, c, ek, ep, et, rho0, r_c,
                                spin, m, ntot, mtot, r, mvir, rvir, tvir, cvel,
                                rvmax, vmax, cNFW,
                                r200, m200,
                                r50, r90, sigma,
                                ndm, nstar, mdm, mstar, ntotdm, ntotstar, mtotdm, mtotstar,
                                xdm, ydm, zdm, xstar, ystar, zstar,
                                vxdm, vydm, vzdm, vxstar, vystar, vzstar,
                                rdm, rstar,
                                adm, bdm, cdm, astar, bstar, cstar,
                                sigmadm, sigmastar,
                                reff, Zstar, tstar,
                                sfr10, sfr100, sfr1000,
                                r50dm, r90dm, r50star, r90star,
                                Vsigma, sigma1D,
                                Vsigma_disc, sigma1D_disc,
                                sigma_bulge, mbulge,
                                n_sersic]
                                #rr, rho,
                                #rr3D, rho3D,
                                #rr3Ddm, rho3Ddm, rr3Dstar, rho3Dstar]
                    if contam:
                        halodata.append(contamlevel)
                        halodata.append(mcontam)
                        halodata.append(mtotcontam)
                        halodata.append(ncontam)
                        halodata.append(ntotcontam)
    
                    data[ihalo] = halodata
    
                    pbar.update()
    
        types = {}
        for k in ('ID', 'nbpart', 'level', 'min_part_id',
                  'host', 'hostsub', 'nbsub', 'nextsub', 'contam',
                  'nDM', 'nstar', 'ntot', 'ntotDM', 'ntotstar',
                  'ncontam', 'ntotcontam'):
            types[k] = np.int64
        for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz',
                  'a', 'b', 'c', 'ek', 'ep', 'et', 'rho0', 'r_c',
                  'spin', 'm', 'mtot', 'r', 'mvir', 'rvir', 'tvir', 'cvel',
                  'rvmax', 'vmax', 'cNFW',
                  'r200', 'm200',
                  'r50', 'r90',
                  'mDM', 'mstar', 'mtotDM', 'mtotstar',
                  'xDM', 'yDM', 'zDM', 'xstar', 'ystar', 'zstar',
                  'vxDM', 'vyDM', 'vzDM', 'vxstar', 'vystar', 'vzstar',
                  'rDM', 'rstar', 'sigmaDM', 'sigmastar',
                  'aDM', 'bDM', 'cDM', 'astar', 'bstar', 'cstar',
                  'reff', 'Zstar', 'tstar',
                  'sfr10', 'sfr100', 'sfr1000',
                  'r50DM', 'r90DM', 'r50star', 'r90star',
                  'Vsigma', 'sigma1D',
                  'Vsigma_disc', 'sigma1D_disc',
                  'sigma', 'sigma_bulge', 'mbulge',
                  'mcontam', 'mtotcontam',
                  'n_sersic'):
            types[k] = np.float64
            #for k in ('rr', 'rho',
            #          'rr3D', 'rr3DDM', 'rr3Dstar',
            #          'rho3D', 'rho3DDM', 'rho3Dstar'
            #          ):
            #    types[k] = 'object'
        dd = {k: data[:, i].astype(types[k])
              for i, k in enumerate(halo_keys)}
    
        halos = pd.DataFrame(dd)
        halos.set_index('ID', inplace=True)
    
        # Get properties in theright units
        # Masses
        halos.m *= 1e11
        halos.mvir *= 1e11
        halos.m200 *= 1e11
        halos.mbulge *= 1e11
        halos.mstar *= 1e11
        halos.mDM *= 1e11
        halos.mtot *= 1e11
        halos.mtotDM *= 1e11
        halos.mtotstar *= 1e11
        if contam:
            halos.mcontam *= 1e11
            halos.mtotcontam *= 1e11
        # SFR
        halos.sfr10 *= 1e11
        halos.sfr100 *= 1e11
        halos.sfr1000 *= 1e11
        # Positions and distances
        data_set = yt.load('../Outputs/output_{:05}/info_{:05}.txt'.format(self.iout, self.iout))
        scale_mpc = float(data_set.length_unit.in_units('cm') / 3.08e24)
        halos.x = halos.x / scale_mpc + .5
        halos.y = halos.y / scale_mpc + .5
        halos.z = halos.z / scale_mpc + .5
        halos.xDM = halos.xDM / scale_mpc + .5
        halos.yDM = halos.yDM / scale_mpc + .5
        halos.zDM = halos.zDM / scale_mpc + .5
        halos.xstar = halos.xstar / scale_mpc + .5
        halos.ystar = halos.ystar / scale_mpc + .5
        halos.zstar = halos.zstar / scale_mpc + .5
    
        # Some cheap derived quantitites
        halos['fstar'] = halos.mstar/halos.m
        halos['fstartot'] = halos.mtotstar/halos.mtot
        # Contamination fraction
        if contam:
            # Mass fractions
            halos['fcontam_mass'] = halos.mcontam / halos.mDM
            halos['fcontam_mass_tot'] = halos.mtotcontam / halos.mtotDM
            halos['fcontam_nb'] = halos.ncontam / halos.nDM
            halos['fcontam_nb_tot'] = halos.ntotcontam / halos.ntotDM
            # Propagate values from the host
            halos['ncontam_host'] = halos.ntotcontam.loc[halos.host].values
            halos['mcontam_host'] = halos.mtotcontam.loc[halos.host].values
            halos['fcontam_nb_host'] = halos.fcontam_nb_tot.loc[halos.host].values
            halos['fcontam_mass_host'] = halos.fcontam_mass_tot.loc[halos.host].values
            # Max contam
            halos['fcontam_mass_max'] = np.maximum.reduce([halos.fcontam_mass.values,
                                                           halos.fcontam_mass_tot.values,
                                                           halos.fcontam_mass_host.values])
            halos['fcontam_nb_max'] = np.maximum.reduce([halos.fcontam_nb.values,
                                                           halos.fcontam_nb_tot.values,
                                                           halos.fcontam_nb_host.values])
        # For subs, check if within rvir
        dist_to_host = np.sqrt((halos.x.values - halos.x.loc[halos.host].values)**2 +
                               (halos.y.values - halos.y.loc[halos.host].values)**2 +
                               (halos.z.values - halos.z.loc[halos.host].values)**2) * scale_mpc
        dist_to_host_rvir = dist_to_host / halos.rvir.loc[halos.host].values
        halos['within_host'] = np.logical_and(dist_to_host_rvir <= 1, halos.level.values > 1)
        
        return halos 

main()
