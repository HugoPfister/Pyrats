import yt
import glob
from tqdm import tqdm
from time import time
import numpy as np
from scipy import integrate


def MofZ(M0, z, z0):
    """
    Return the typical mass (in units 1e12 Msun), at redshift z, of a halo of mass M0 (in units 1e12 Msun) at redshift z0
    This is the integration from eq.(2) (the mean) from Fakhouri+10 
    """
    return M0 * (1 - M0**0.1 * 0.0627 * ((0.11 * np.log(1 + z) - 1.11 * z) - (0.11 * np.log(1 + z0) - 1.11 * z0)))**(-10)


def N_merger_until_z(ximin, M0, z, z0):
    """
    Return the number of merger with a mass ratio greater than xmin
    for a halo of mass M0 at z0
    between z and z0 < z
    """
    [intxi, residual] = integrate.quad(
        lambda x: x**-1.995 * np.exp((x / 0.00972)**0.263), ximin, 1)
    [result, residual] = integrate.quad(
        lambda zz: 0.0104 * (1 + zz)**0.0993 * intxi * MofZ(M0, zz, z0)**0.133, z0, z)
    return result


def Mstar(Mh, z):
    """
    Give the stellar mass as a function of the halo mass (Mh) at a given redshift z following Behroozi et al - 2012 formula
    """
    a = 1. / (1 + z)
    nu = np.exp(-4 * a**2)
    log_eps = -1.777 + (-0.006 * (a - 1)) * nu - 0.119 * (a - 1)
    log_M1 = 11.514 + (-1.793 * (a - 1) + (-0.251) * z) * nu
    alpha = -1.412 + (0.731) * (a - 1) * nu
    delta = 3.508 + (2.608 * (a - 1) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1) + 0.279 * z) * nu
    log_MhOverM1 = np.log10(Mh) - log_M1
    f_1 = -np.log10(10**(alpha * log_MhOverM1) + 1) + (delta * np.log10(1 +
                                                                        np.exp(log_MhOverM1))**gamma) / (1 + np.exp(10**(-log_MhOverM1)))
    return 10**(log_eps + log_M1 + f_1 - (-np.log10(2) + delta * np.log10(2)**gamma / (1 + np.exp(1))))


def Mhalo(mStar, z):
    """
    Give the halo mass as a function of the stellar mass (mStar) at a given redshift z following Behroozi et al - 2012 formula
    """
    mh = np.logspace(3, 15, 1500)
    mstar = np.copy([Mstar(mmh, z) for mmh in mh])
    diffm = np.abs(mstar - mStar)
    arg = diffm.argmin()
    mhalo = mh[arg]
    if diffm[arg] / mhalo > 1e-2:
        print 'there might be something wrong in the estimation of mHalo'
    return mhalo
