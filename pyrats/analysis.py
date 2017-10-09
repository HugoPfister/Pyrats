import yt
import glob
from tqdm import tqdm
import os as os
import matplotlib.pyplot as plt

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

    for fn in yt.parallel_objects(files):
        plt.clf()
        ds = yt.load(fn)
        i = files.index(fn)
        
        for field in qtty:
          if 'stars' in field[1]:
            yt.add_particle_filter(
                "stars", function=fields.stars, filtered_type="all",
                requires=["particle_age"])
            ds.add_particle_filter("stars")
          if 'dm' in field[1]:
            yt.add_particle_filter(
                "dm", function=fields.dm, filtered_type="all")
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
