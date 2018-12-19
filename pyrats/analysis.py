import yt
import glob
import os as os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from . import halos, trees, sink, fields, utils, load_snap

def profiles(ds, center=None,
        rbound=[(0.01,'kpc'),(10, 'kpc')],
        n_bins=100, log=True,
        qtty=[('gas','density')],
        weight_field=('index','cell_volume'), bin_fields=('index', 'radius'),
        hnum=None, Galaxy=False, bhid=None, 
        accumulation=False, filter=None):
    """
    This routine plot the profile for a given snapshots

    center : center of the sphere for the plot, useless if hnum/bhid

    rbound : min/max radius for the profile

    n_bins : number of bin for the radius
    log : if True then loglog plot
    filter (None) : add a particular filter (for instance a temperature floor for gas) CARE WITH UNITS
    example syntax for filter: "obj[('gas','temperature')] < 1e4]" (" and obj are mandatory)

    qtty : list of fields to be profiled
    weight_field : weight field for the profile

    hnum : center on the center of the halo
    bhid : center on a particular BH

    accumulation : sum between 0 and r (cumulative profile)
    ds : dataset to profile 
    """

    yt.funcs.mylog.setLevel(40)
    if center != None:
        sp = ds.sphere(center, width)
    else:
        sp = load_snap.get_sphere(ds, rbound[1], bhid, hnum, Galaxy)

    if filter != None:
        sp=ds.cut_region(sp, [filter])

    p=yt.create_profile(data_source=sp, bin_fields=bin_fields, weight_field=weight_field,
            fields=qtty,
            accumulation=accumulation,
            n_bins=n_bins, deposition='cic')
    yt.funcs.mylog.setLevel(20)

    return p

def dist_sink_to_halo(IDsink, IDhalos, timestep=None, Galaxy=False):
    """
    Use the tree from HaloFinder/TreeMaker and the sink files to
    compute the distance, as a function of time, between sinks and halos
    work also for non Cosmo sims, use GalFinder.py (explanation in GalFinder.py)

    CARE :  
    IDsink = list of id sinks
    IDhalos = list of halos ids (given by HaloFinder), use -1 for GalFinder output

    The first element of IDsink is associated with the first of hid etc...
    """
    if np.copy(IDhalos).max() > 0:
        tree = trees.Forest(Galaxy=Galaxy)

    ds=load_snap.load(-1, verbose = False)    
    Lbox = float(ds.length_unit.in_units('kpc')*(1+ds.current_redshift))
    
    d=[]; t=[]    
    if len(IDsink) != len(IDhalos):
        print('Please put the same number of sinks and halos')
        return

    for i in range(len(IDsink)):
        bh = pd.read_csv('./sinks/BH{:05}.csv'.format(IDsink[i]))
        bhid = IDsink[i]
        hid = IDhalos[i]

        if hid == -1:
                Gal=pd.read_csv('GalCenter.csv')
                xh = interp1d(Gal.t, Gal.cx, kind='cubic')
                yh = interp1d(Gal.t, Gal.cy, kind='cubic')
                zh = interp1d(Gal.t, Gal.cz, kind='cubic')
                tmin = max(Gal.t.min(),  bh.t.min())
                tmax = min(Gal.t.max(),  bh.t.max())

        if hid > 0:
            prog = tree.get_family(hid, timestep=timestep)

            time = prog.aexp.apply(lambda x: ds.cosmology.t_from_z(1/x-1).to('Gyr'))
            #magic line to make it work better...
            time -= time.tolist()[-1] - ds.cosmology.t_from_z(ds.current_redshift).to('Gyr').value
            xh = interp1d(time, prog.x, kind='cubic')
            yh = interp1d(time, prog.y, kind='cubic')
            zh = interp1d(time, prog.z, kind='cubic')

            tmin=max(time.min(),bh.t.min())
            tmax=min(time.max(),bh.t.max())
            
        bh = bh.loc[(bh.t > tmin) & (bh.t < tmax)]
        dx=(xh(bh.t)-bh.x) 
        dy=(yh(bh.t)-bh.y) 
        dz=(zh(bh.t)-bh.z) 
        
        d+=[np.sqrt(dx**2+dy**2+dz**2)*Lbox]
        t+=[bh.t]
        if ds.cosmological_simulation == 1:
            d[-1] = d[-1] / (1+ds.cosmology.z_from_t(ds.arr(t[-1].tolist(), 'Gyr')))

    return d, t

def mean_density(snap=[-1], hnum=None, timestep=None, Galaxy=False, bhid=None, radius=1):
    '''
    This routine measure the mean density (gas/stars) in a sphere of radius 'radius' around the defined object
    the output is a pandaframe with time and mean density.

    radius : if int then in units of resolution, else in the form (10, 'pc')
    hnum : ID of the halo, at timestep, you want to center the images
    timestep : see above, default is the last one
    Galaxy : True if galaxy, false if halo
    bhid : ID of the sink you want to center the images
    snap : list of snapshots you want to show, default ALL (-1)
    '''
    files = utils.find_outputs()
    ToConsider, hid = utils.filter_outputs(snap=snap,hnum=hnum,timestep=timestep,Galaxy=Galaxy,bhid=bhid)
    
    ds = load_snap.load(1, verbose=False)
    if ((type(radius) is int) | (type(radius) is float)):
        width = (float(radius*(ds.length_unit/ds.parameters['aexp']).to('pc')/2**ds.parameters['levelmax']), 'pc')
    elif (type(radius) is tuple):
        if ((type(radius[0]) is int) | (type(radius) is float)) & (type(radius[1]) is str):
            width = radius
        else:
            raise TypeError('Please give the radius with the form (10,\'pc\')')
    else:
        raise TypeError('Please give an int or a tuple like (10, \'pc\') for the radius')
 
    res = np.array(Parallel(n_jobs=utils._get_ncpus())(delayed(_mean_density)(i, ToConsider, hid, width, bhid, files, Galaxy) for i in tqdm(range(len(files)))))
    res = res[res>-1]
    res = res.reshape(len(res) // 3, 3)
    res=pd.DataFrame(res, columns=['t', 'rho_star', 'rho_gas'])
    
    res.to_csv('Density_'+utils._get_extension(hnum,timestep,Galaxy,bhid,radius), index=False)

    return res

def _mean_density(i, ToConsider, hid, width, bhid, files, Galaxy):
    if ToConsider[i]:       
        ds = load_snap.load(files[i], haloID=hid[i], Galaxy=Galaxy, bhID=bhid, radius=width, stars=True, verbose=False)
        sp = load_snap.get_sphere(ds, bhid, hid[i], Galaxy, width)

        t = ds.current_time.to('Myr')
        
        rho_star = sp[('stars','particle_mass')].sum() / (4/3*np.pi*ds.arr(width[0], width[1])**3)
        rho_star = rho_star.to('Msun/pc**3')

        rho_gas = sp[('gas','cell_mass')].sum() / (4/3*np.pi*ds.arr(width[0], width[1])**3)
        rho_gas = rho_gas.to('Msun/pc**3')
        
        return [t, rho_star, rho_gas]
    else:
        return [-1,-1,-1]

