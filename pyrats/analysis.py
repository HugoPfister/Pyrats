import yt
import glob
from tqdm import tqdm
import os as os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np


from . import halos, trees, sink, fields

def profile(folder='./', center=[0.5,0.5,0.5],
        rbound=[(0.01,'kpc'),(10, 'kpc')], ybound=None, units=None,
        n_bins=64, log=True,
        qtty=[('gas','density'),('deposit','stars_cic'),('deposit','dm_cic')],
        weight_field=('index','cell_volume'),
        hnum=None, bhid=None, accumulation=False, snap=-1):
    """
    This routine plot the profile for all snapshots

    folder : location to save plots
    center : center of the sphere for the plot, useless if hnum/bhid

    rbound : min/max radius for the profile
    ybound : min/max value to show, in units 'units'
    units : unit to show the profile

    n_bins : number of bin for the radius
    log : if True then loglog plot

    qtty : list qqty to be profiled, must have the same dimension
    weight_field : weight field for the profile

    hnum : center on the center of the halo
    bhid : center on a particular BH

    accumulation : sum between 0 and r (cumulative profile)
    snap : list of snapshots to profile
    """

    files = glob.glob('output_*/info*')
    files.sort()

    path=folder + '/profiles'
    os.system('mkdir ' + path)

    istart=0
    if hnum != None:
        t = trees.Forest(LoadGal=False)
        hid = int(t.trees[(t.trees.halo_ts == t.trees.halo_ts.max())
                      & (t.trees.halo_num == hnum)].halo_id)
        prog_id = [_ for _ in t.get_main_progenitor(hid).halo_num]
        prog_id = prog_id[::-1]
        istart=len(files) - len(prog_id)
        files=files[istart:]

        path = path + '/Halo' + str(hnum)
        os.system('mkdir ' + path)

    if bhid != None:
        path = path + '/BH' + str(bhid)
        os.system('mkdir ' + path)
        s=sink.Sinks()
        tform=s.sink[bhid].t.min()
        imin=0
        for f in files:
            ds=yt.load(f)
            if ds.current_time < ds.arr(tform, 'Gyr'):
                imin+=1
        files=files[imin:]

    path = path + '/'
    for f in qtty:
        path = path + f[0]+f[1]
    os.system('mkdir ' + path)


    if snap != -1:
        files = [files[i-istart-1] for i in snap]
        if hnum != None:
            prog_id = [prog_id[i-istart-1] for i in snap]

    part=False
    for field in qtty:
        if (('stars' in field[1]) or ('dm' in field[1])):
            part=True 

    for fn in yt.parallel_objects(files):
        plt.clf()
        if part: 
            ds = yt.load(fn, extra_particle_fields=[("particle_age", "d"),("particle_metallicity", "d")])
        else:
            ds = yt.load(fn)
        i = files.index(fn)
        
        for field in qtty:
          if 'stars' in field[1]:
            yt.add_particle_filter(
                "stars", function=fields.stars, filtered_type="io",
                requires=["particle_age"])
            ds.add_particle_filter("stars")
          if 'dm' in field[1]:
            yt.add_particle_filter(
                "dm", function=fields.dm, filtered_type="io")
            ds.add_particle_filter("dm")
        
        c = center
        if hnum != None:
            h = halos.HaloList(ds)
            hid = prog_id[i]
            hh = h.halos.loc[hid]
            c = [h.halos['x'][hid], h.halos['y'][hid], h.halos['z'][hid]]

        if bhid != None:
            ds.sink = sink.get_sinks(ds)
            bh = ds.sink.loc[ds.sink.ID == bhid]
            c = [bh.x.item(), bh.y.item(), bh.z.item()]

        sp=ds.sphere(c, rbound[1])
        
        p=yt.create_profile(data_source=sp, bin_fields=('index', 'radius'), weight_field=weight_field,
            fields=qtty,
            accumulation=False,
            n_bins=64)

        for field in qtty:
            plt.plot(p.x.in_units(rbound[0][1]),
                p[field].in_units(units) if units!=None else p[field],
                label=field[0]+' '+field[1])

        if log:
            plt.loglog()

        plt.legend()

        plt.xlim(rbound[0][0], float(ds.arr(rbound[1][0],rbound[1][1]).in_units(rbound[0][1])))
        if ybound != None:
            plt.ylim(ybound[0], ybound[1])

        plt.xlabel('Radius ['+rbound[0][1]+']')
        plt.ylabel(qtty[0][1]+' ['+(units if units!=None else str(p[qtty[0]].units))+']')
        plt.title('t={:.3f} Gyr'.format(float(ds.current_time.in_units('Gyr'))))

        plt.savefig(path+'/profile{:03}'.format(i+1))
        plt.clf()
    return

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
        tree = trees.Forest()

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
