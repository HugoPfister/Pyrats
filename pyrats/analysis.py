import yt
import glob
import os as os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


from . import halos, trees, sink, fields

def profiles(ds, center=[0.5,0.5,0.5],
        rbound=[(0.01,'kpc'),(10, 'kpc')],
        n_bins=128, log=True,
        qtty=[("gas","density"),("deposit","stars_cic"),("deposit","dm_cic")],
        weight_field=('index','cell_volume'), bin_fields=('index', 'radius'),
        hnum=None, Galaxy=False, bhid=None, 
        accumulation=False, filter=None):
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

    yt.funcs.mylog.setLevel(40)
    c = center
    if hnum != None:
      if hnum > 0:
        if Galaxy:
            h = ds.gal.gal.loc[hnum]
        else:
            h = ds.halo.halos.loc[hnum]
        c = [h.x.item(), h.y.item(), h.z.item()]
      else:
            print('Considering ideal simulation, GalCenter.csv must be present')
            Gal=pd.read_csv('GalCenter.csv')
            arg = np.abs(Gal.t - float(ds.current_time.in_units('Gyr'))).argmin()
            h = Gal.loc[arg]
            c = [h.cx.item(), h.cy.item(), h.cz.item()]

    if bhid != None:
        bh = ds.sink.loc[ds.sink.ID == bhid]
        c = [bh.x.item(), bh.y.item(), bh.z.item()]

    sp=ds.sphere(c, (rbound[1][0], rbound[1][1]))
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
            FirstOutput = tree.trees.halo_ts.min()
            prog = tree.get_family(hid, timestep=timestep)
            #prog = tree.get_main_progenitor(hid, timestep=timestep)
            prog['halo_ts'] -= FirstOutput
            prog = prog.loc[prog.halo_ts < 45]

            xh = interp1d(tree.timestep['age'][prog.halo_ts.min():prog.halo_ts.max()+1], prog.x*1000, kind='cubic')
            yh = interp1d(tree.timestep['age'][prog.halo_ts.min():prog.halo_ts.max()+1], prog.y*1000, kind='cubic')
            zh = interp1d(tree.timestep['age'][prog.halo_ts.min():prog.halo_ts.max()+1], prog.z*1000, kind='cubic')

            tmin=max(tree.timestep['age'][prog.halo_ts.min():prog.halo_ts.max()+1].min(),bh.t.min())
            tmax=min(tree.timestep['age'][prog.halo_ts.min():prog.halo_ts.max()+1].max(),bh.t.max())
            
        bh = bh.loc[(bh.t > tmin) & (bh.t < tmax)]
        dx=xh(bh.t)-(bh.x-0.5)*Lbox 
        dy=yh(bh.t)-(bh.y-0.5)*Lbox 
        dz=zh(bh.t)-(bh.z-0.5)*Lbox 
        
        z=np.copy([0]*len(bh.t))
        if ds.cosmological_simulation == 1:
            z=ds.cosmology.z_from_t(ds.arr(list(bh.t), 'Gyr'))
        d+=[np.sqrt(dx**2+dy**2+dz**2)/(1+z)]
        t+=[bh.t]

    return d, t
