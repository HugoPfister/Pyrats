import yt
import glob
from tqdm import tqdm
import os
import subprocess

from . import halos, trees, sink

def plot_snapshots(axis='z', center=[0.5,0.5,0.5],
        field=('deposit','all_density'), weight_field=('index','ones'), slice=False,
        width=(10, 'kpc'), axis_units='kpc', folder='./',
        cbarunits=None, cbarbounds=None, cmap='viridis',
        hnum=None, plothalos=False, masshalomin=1e10,
        bhid=None, plotsinks=False, sinkdynamics=0,
        snap=-1):

    files = glob.glob('output_*/info*')
    files.sort()

    path=folder + '/snapshots'
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


    if slice:
        path = path + '/Slice'
        os.system('mkdir ' + path)
    else:
        path = path + '/Proj'
        os.system('mkdir ' + path)

    if width != None:
        path = path + '/' + str(width[0]) + width[1]
        os.system('mkdir ' + path)

    path = path + '/' + field[0] + field[1]
    os.system('mkdir ' + path)
    path = path + '/' + 'Axis_' + axis
    os.system('mkdir ' + path)

    if snap != -1:
        files=[files[i-istart-1] for i in snap]
        if hnum != None:
            prog_id=[prog_id[i-istart-1] for i in snap]

    for i in tqdm(range(len(files))):
        ds = yt.load(files[i])
        
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

        sp=ds.sphere(c, width)
        
        if slice:
            p = yt.SlicePlot(ds, data_source=sp, axis=axis, fields=field, center=sp.center, width=width)
        else:
            p= yt.ProjectionPlot(ds, data_source=sp, axis=axis, fields=field, weight_field=weight_field,
                 center=sp.center, width=width)

        if plotsinks:
            ds.sink = sink.get_sinks(ds)
            for bhnum in ds.sink.ID:
                ch = ds.sink.loc[ds.sink.ID == bhnum]
                if (((c[0] - ch.x.item())**2 +
                    (c[1] - ch.y.item())**2 +
                    (c[2] - ch.z.item())**2) <
                    ((sp.radius.in_units('code_length') / 2)**2)):

                    p.annotate_marker([ch.x.item(), ch.y.item(), ch.z.item()],
                                  marker='.', plot_args={'color':
                                  'black', 's': 100})

                    p.annotate_text([ch.x.item(), ch.y.item(), ch.z.item()],
                                text=str(ch.ID.item()),
                                text_args={'color': 'black'})

                    if sinkdynamics > 0:
                        ch=s.sink[bhnum]
                        ch = ch.loc[
                         (ch.t>float((ds.current_time-
                            ds.arr(sinkdynamics, 'Myr')).in_units('Gyr'))) &
                         (ch.t<float((ds.current_time+
                            ds.arr(sinkdynamics, 'Myr')).in_units('Gyr')))]
                        x=list(ch.x)
                        y=list(ch.y)
                        z=list(ch.z)
                        for i in range(len(x)-1):
                            p.annotate_line([x[i],y[i],z[i]],
                                [x[i+1],y[i+1],z[i+1]],
                                coord_system='data', plot_args={'color':'black'})

        if plothalos:
            for hid in h.ID:
                ch = h.loc[hid]
                if ((ch.m > masshalomin) &
                    (((c[0] - ch.x.item())**2 +
                      (c[1] - ch.y.item())**2 +
                      (c[2] - ch.z.item())**2) <
                     ((dd.radius.in_units('code_length') / 2)**2))):

                    p.annotate_sphere([ch.x.item(), ch.y.item(), ch.z.item()],
                                      (ch.rvir.item(), 'Mpc'),
                                      circle_args={'color': 'black'})

                    p.annotate_text([ch.x.item(), ch.y.item(),
                                     ch.z.item()], text=str(int(ch.ID.item())))

        p.annotate_timestamp(corner='upper_left', time=True, redshift=True)
        p.annotate_scale(corner='upper_right')

        p.set_cmap(field=field, cmap=cmap)
        if cbarunits !=None:
            p.set_unit(field=field, new_unit=cbarunits)
        if cbarbounds !=None:
            p.set_zlim(field=field, zmin=cbarbounds[0], zmax=cbarbounds[1])
            if cbarbounds[1] / cbarbounds[0] > 50:
                p.set_log(field, log=True)

        p.set_axes_unit(width[1])
        p.set_width(width)

        print(path)
        p.save(path)

    return


    



def plot_all_snapshots(axis='z', field=('deposit', 'all_density'),
                       folder='./', cbarmin=None, cbarmax=None,
                       weight_field=('index', 'ones'), width=None, axis_units='kpc'):
    """
    Plot a map of all the snapshots for this simulations
    Parameters
    ----------
    * axis ('z') : Projection axis
    * field ('deposit','all_density') : Field that will be projected
    * folder='./' : Folder to put the output (a subfolder 'snapshots'
      will be created with the outputs inside)
    * cbarmin/cbarmax (None/None) : Set limits to the colorbar
    * weight_field ('index','ones'): Field used to weight the map
    """
    files = glob.glob('output_*/info*')
    for dd in tqdm(files):
        ds = yt.load(dd)
        if width != None:
            sp=ds.sphere([0.5,0.5,0.5], width)
            p = yt.ProjectionPlot(ds, axis=axis, fields=field,
                              weight_field=weight_field, axes_unit=axis_units,
                               data_source=sp)
            p.set_width(width)
        else:
            p = yt.ProjectionPlot(ds, axis=axis, fields=field,
                              weight_field=weight_field, axes_unit=axis_units)
        p.set_cmap(field=field, cmap='viridis')
        p.annotate_timestamp(corner='upper_left', time=True, redshift=True)
        p.annotate_scale(corner='upper_right')
        p.set_zlim(field=field, zmin=cbarmin, zmax=cbarmax)
        os.system('mkdir ' + folder + '/snapshots')
        os.system('mkdir ' + folder + '/snapshots/'+field[0]+field[1])
        if width != None:
            os.system('mkdir ' + folder + '/snapshots/'+field[0]+field[1]+'/'+str(width[0])+width[1])
            path = folder + '/snapshots/'+field[0]+field[1]+'/'+str(width[0])+width[1]
        else:
            os.system('mkdir ' + folder + '/snapshots/'+field[0]+field[1]+'/all')
            path = folder + '/snapshots/'+field[0]+field[1]+'/all'
            

        
        p.save(path)
    return


def plot_halo_history(hnum, axis='z', 
                      field=('deposit', 'all_density'), folder='./', 
                      weight_field=('index', 'ones'), slice=False,
                      size=None, units=None, 
                      cmap='viridis', limits=[0, 0],
                      plotsinks=False, SinkDynamicsTimeScale = -1, 
                      plothalos=False, masshalomin=1e10):
    """
    TODO but not urgent: gather this and plot_bh_history in a function
    with a switch for BH/halos

    Plot a map, at eauch output of the halo with ID hnum at the last
    output of the simulation (computed with HaloMaker)
    Parameters
    ----------
    * hnum: ID, for the last output, of the halo you want to plot all
      the progenitors
    Other parame ters are the one used in halo.plot_halo
    """

    files = glob.glob('output_*/info*')
    files.sort()
    ds = yt.load(files[-1])
    t = trees.Forest(LoadGal=False)
    hid = int(t.trees[(t.trees.halo_ts == t.trees.halo_ts.max())
                      & (t.trees.halo_num == hnum)].halo_id)
    path = os.path.join(folder, 'Halo%s' % (hnum))
    if not os.path.exists(path):
        subprocess.call(['mkdir', path])
    if size is not None:
        path = os.path.join(path, '%s%s' % (size[0], size[1]))
        subprocess.call(['mkdir', path])
    path = os.path.join(path, '%s%s' % (field[0], field[1]))
    subprocess.call(['mkdir', path])
    path = os.path.join(path, 'Axis_%s' % axis)
    subprocess.call(['mkdir', path])

    prog_id = [_ for _ in t.get_main_progenitor(hid).halo_num]
    h = halos.HaloList(ds)
    if size is None:
        rvirfin = (h.halos.rvir[prog_id[-1]] *
                   (1 + ds.current_redshift), 'Mpccm')
    else:
        rvirfin = size

    for i in tqdm(range(len(files))):
        ds = yt.load(files[-i - 1])
        h = halos.HaloList(ds)
        h.plot_halo(prog_id[-i - 1], axis=axis, folder=path,
                    field=field, r=rvirfin, weight_field=weight_field,
                    cmap=cmap, limits=limits, plotsinks=plotsinks,
                    units=units, SinkDynamicsTimeScale=SinkDynamicsTimeScale)
    return
