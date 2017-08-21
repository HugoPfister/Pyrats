# This file contains the different filters one may
# want to use for particles (dm, stars etc...)
#
#/!\/!\/!\/!\/!\/!\/!\/!\
# Be carefull with the definition of stars in RAMSES
# Here we assume:
# - DM with a particle age very negative and a positive ID
# - Stars with a particle age positive
# - Using this only remains VERY young stars (negligible) and cloud particles from BH


def stars(pfilter, data):
    filter = (data['particle_age'] > 0) & (data['particle_age'] != None)
    return filter


def dm(pfilter, data):
    '''
    Select DM particles'''
    if 'particle_age' in data:
        filter = ((data['particle_identifier'] > 0) &
                  (data['particle_age'] == data['particle_age'].min()) &
                  (data['particle_age'] != None))
    else:
        filter = data['particle_identifier'] > 0
    return filter


def young_stars(pfilter, data):
    '''
    Select particles created after the beginning of the simulation,
    that are younger than 10Myr.'''
    filter = ((data['particle_age'] > 0) &
              (data['particle_age'] < data.ds.arr(10, 'Myr')) &
              (data['particle_age'] != None))
    return filter
