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
        hnum=None, bhid=None, accumulation=False):

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

    for i in tqdm(range(len(files))):
        ds = yt.load(files[i])
        
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
        
        p=yt.create_profile(data_source=sp, bin_fields=('index', 'radius'), weight_field=('index','cell_volume'),
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
        plt.ylabel(qtty[0][1]+' ['+(units if units!=None else p[field].units)+']')
        plt.title('t={:.3f} Gyr'.format(float(ds.current_time.in_units('Gyr'))))

        plt.savefig(path+'/profile{:03}'.format(i+1))
        plt.clf()
    return
