# This file contains the different filters one may
# want to use for particles (dm, stars etc...)
#
#/!\/!\/!\/!\/!\/!\/!\/!\
# Be carefull with the definition of stars in RAMSES
# Here we assume:
#-DM with a particle age very negative and a positive ID
#-Stars with a particle age positive
#-Using this only remains VERY young stars (negligible) and cloud particles from BH


import yt
import numpy as np


def stars(pfilter, data):
    filter = np.logical_and(data["particle_age"] >
                            0, data["particle_age"] != None)
    return filter


def dm(pfilter, data):
    try:
        data['particle_age']
        filter = np.logical_and((data["particle_identifier"] > 0) & (
            data["particle_age"] == data['particle_age'].min()), data["particle_age"] != None)
    except:
        filter = data["particle_identifier"] > 0
    return filter


def young_stars(pfilter, data):
    filter = np.logical_and((data["particle_age"] > 0) & (
        data["particle_age"] < d.ds.arr(10, 'Myr')), data["particle_age"] != None)
    return filter
