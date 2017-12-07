import yt
import glob
from tqdm import tqdm
import os as os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


from . import halos, trees, sink, fields

def profiles(ds, center=[0.5,0.5,0.5],
        rbound=[(0.01,'kpc'),(10, 'kpc')],
        n_bins=128, log=True,
        qtty=[('gas','density'),('deposit','stars_cic'),('deposit','dm_cic')],
        weight_field=('index','cell_volume'), bin_fields=('index', 'radius'),
        hnum=None, bhid=None, accumulation=False, filter=None):
    """
    This routine plot the profile for all snapshots

    center : center of the sphere for the plot, useless if hnum/bhid

    rbound : min/max radius for the profile

    n_bins : number of bin for the radius
    log : if True then loglog plot
    filter (None) : add a particular filter (for instance a temperature floor for gas) CARE WITH UNITS
    example syntax for filter: "obj[('gas','temperature')] < 1e4]" (" and obj are mandatory)

    qtty : list qqty to be profiled, must have the same dimension
    weight_field : weight field for the profile

    hnum : center on the center of the halo
    bhid : center on a particular BH

    accumulation : sum between 0 and r (cumulative profile)
    ds : dataset to profile 
    """

    c = center
    if hnum != None:
        h = halos.HaloList(ds)
        hh = h.halos.loc[hnum]
        c = [hh.x.item(), hh.y.item(), hh.z.item()]

    if bhid != None:
        ds.sink = sink.get_sinks(ds)
        bh = ds.sink.loc[ds.sink.ID == bhid]
        c = [bh.x.item(), bh.y.item(), bh.z.item()]

    sp=ds.sphere(c, (rbound[1][0]*2, rbound[1][1]))
    if filter != None:
        sp=ds.cut_region(sp, [filter])

    p=yt.create_profile(data_source=sp, bin_fields=bin_fields, weight_field=weight_field,
            fields=qtty,
            accumulation=False,
            n_bins=n_bins)

    return p

def dist_sink_to_halo(IDsink, IDhalos):
    """
    Use the tree from HaloFinder/TreeMaker and the sink files to
    compute the distance, as a function of time, between sinks and halos
    work also for non Cosmo sims, use GalFinder.py (explanation in GalFinder.py)

    CARE : ALL IDs are set for the last output available
    IDsink = list of id sinks
    IDhalos = list of halos ids (given by HaloFinder), use -1 for GalFinder output

    The first element of IDsink is associated with the first of hid etc...
    """
    if np.copy(IDhalos).max() > 0:
        tree = trees.Forest(LoadGal=False)

    files = glob.glob('output*/info*')
    files.sort()
    ds=yt.load(files[-1])    
    Lbox = float(ds.length_unit.in_units('kpc')*(1+ds.current_redshift))
    
    files = glob.glob('./sinks/BH*')
    files.sort()

    d=[]; t=[]    
    if len(IDsink) != len(IDhalos):
        print('Please put the same number of sinks and halos')
        return

    for i in range(len(IDsink)):
        bh = pd.read_csv(files[IDsink[i]])
        bhid = IDsink[i]
        hid = IDhalos[i]

        if hid == -1:
                Gal=pd.read_csv('GalCenter.csv')
                xh = interp1d(Gal.t, (Gal.cx-0.5)*Lbox, kind='cubic')
                yh = interp1d(Gal.t, (Gal.cy-0.5)*Lbox, kind='cubic')
                zh = interp1d(Gal.t, (Gal.cz-0.5)*Lbox, kind='cubic')
                tmin = max(Gal.t.min(), bh.t.min())
                tmax = min(Gal.t.max(), bh.t.max())

        if hid > 0:
            prog = tree.get_main_progenitor(int(tree.trees.loc[(tree.trees.halo_num == hid) & (tree.trees.halo_ts == tree.trees.halo_ts.max())].halo_id))
        
            xh = interp1d(tree.timestep['age'][prog.halo_ts.min()-1:prog.halo_ts.max()], prog.x[::-1]*1000, kind='cubic')
            yh = interp1d(tree.timestep['age'][prog.halo_ts.min()-1:prog.halo_ts.max()], prog.y[::-1]*1000, kind='cubic')
            zh = interp1d(tree.timestep['age'][prog.halo_ts.min()-1:prog.halo_ts.max()], prog.z[::-1]*1000, kind='cubic')

            tmin=max(tree.timestep['age'][prog.halo_ts.min()-1:prog.halo_ts.max()].min(),bh.t.min())
            tmax=min(tree.timestep['age'][prog.halo_ts.min()-1:prog.halo_ts.max()].max(),bh.t.max())
            
        bh = bh.loc[(bh.t > tmin) & (bh.t < tmax)]
        dx=xh(bh.t)-(bh.x-0.5)*Lbox 
        dy=yh(bh.t)-(bh.y-0.5)*Lbox 
        dz=zh(bh.t)-(bh.z-0.5)*Lbox 
        
        z=np.copy([1]*len(bh.t))
        if ds.cosmological_simulation == 1:
            z=ds.cosmology.z_from_t(ds.arr(list(bh.t), 'Gyr'))
        d+=[np.sqrt(dx**2+dy**2+dz**2)/z]
        t+=[bh.t]

    return d, t
