# This file contains the different filters one may
# want to use for particles (dm, stars etc...)
#
#/!\/!\/!\/!\/!\/!\/!\/!\
# Be carefull with the definition of stars in RAMSES
# Here we assume:
# - DM with a particle age very negative and a positive ID
# - Stars with a particle age positive
# - Using this only remains VERY young stars (negligible) and cloud particles from BH
import yt

def _star(pfilter, data):
    '''
    Select stars particles
    '''
    if data.ds.cosmological_simulation==1:
        filter = (data['io','particle_birth_time'] != 0) & (data['io','particle_birth_time'] != None)
    else:
        filter = (data['io','particle_birth_time'] != 0)
    return filter


def _DM(pfilter, data):
    '''
    Select DM particles
    '''
    if data.ds.cosmological_simulation==1:    
     if data['io','particle_birth_time'] != None:
         filter = (data['io','particle_birth_time'] == 0) & (data['io','particle_identity'] > 0)
     else:
         filter= (data['io','particle_identity'] >0)
    else:
         filter = ((data['io','particle_birth_time'] == 0 ) & (data['io','particle_identity'] > 0))
    return filter


def young_star(ds):
    yt.add_particle_filter("young_star", function=_young_star,
                  filtered_type="star")
    ds.add_particle_filter("young_star")  
    return ds

def _young_star(pfilter, data):
    '''
    Select particles created after the beginning of the simulation,
    that are younger than 10Myr.'''
    filter = data[('star','star_age')] < data.ds.arr(10,'Myr')
    return filter
