import yt
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from . import halos, trees, sink, analysis, load_snap, utils

def _mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def plot_snapshots(axis='z', center=[0.5,0.5,0.5],
                   field=('gas', 'density'),
                   weight_field=('index','ones'), slice=False,
                   width=(10, 'kpc'), axis_units='kpc', folder='./',
                   cbarunits=None, cbarbounds=None, cmap='viridis', LogScale=True,
                   hnum=None, timestep=None, Galaxy=False, bhid=None,
                   plothalos=False, masshalomin=1e5,
                   plotsinks=[0], plotparticles=False, sinkdynamics=0, BHcolor='black',
                   snap=[-1], extension='pdf', method='integrate'):
    """
    Visualization function, by default it is applied to ALL snapshots

    axis : the projection axis
    center : center (in code_length units) of the region to show, useless if hnum/bhid

    width : width of the window, default (10, 'kpc'), can be 'Rvir'
        in that case the size is set to the virial radius of each snapshots
    axis_units : units for the x/y axis, if None then no axis
    folder : folder to save the images

    field : yt field to show
    weight_field : yt field to use to weight, default ('index', 'ones')
    slice : True/False slice/projection, if True weight_field useless

    cbarunits : units for the colorbar, defaut units of field
    cbarbounds : limits for the cbar, in units of cbarunits
    cmap : cmap to use
    LogScale (True) : plot in log

    hnum : ID of the halo, in the last output, you want to center the images
    plothalos : circles around halos with a mass larger than masshalomin
    masshalomin : in Msun, minimum mass for the halo to show

    bhid : ID of the sink you want to center the images
    plotsinks : if -1 show sinks and their ID
                if list shows only the asked ones
                if [0] do not show
    sinkdynamics : draw lines to show BH dynamics between [t-sinkdynamics, t+sinkdynamics], in Myr
    BHcolor : color to show BHs and their ID 

    plotparticles: overplot the particles as black dots

    snap : list of snapshots you want to show, default ALL (-1)
    """

    # yt.funcs.mylog.setLevel(40)
    files = utils.find_outputs()
    path = os.path.join(folder, 'snapshots')
    _mkdir(path)

    ToPlot = [True] * len(files)

    if snap != [-1]:
        for i in range(len(files)):
            ToPlot[i] = ((i+1) in snap)

    if ((hnum is not None) and (bhid is not None)):
        print('Please specify only hnum or bhid but not both')
        return

    if hnum is not None:
        t = trees.Forest(Galaxy=Galaxy)
        prog = t.get_main_progenitor(hnum=hnum, timestep=timestep)
        for i in range(len(files)):
            ToPlot[i] = (ToPlot[i]) & (i+1 in np.array(prog.halo_ts))
        if Galaxy:
            path = os.path.join(path, 'Galaxy{:04}_output_{:05}'.format(
                hnum, timestep))
        else:
            path = os.path.join(path, 'Halo{:04}_output_{:05}'.format(
                hnum, timestep))
        _mkdir(path)

    if bhid is not None:
        path = os.path.join(path, 'BH%s' % bhid)
        _mkdir(path)
        s = sink.Sinks(ID=[bhid])
        tform = s.sink[bhid].t.min()
        tmerge = s.sink[bhid].t.max()
        for isnap, f in enumerate(files):
            ds = yt.load(f)
            ToPlot[isnap] = (ToPlot[isnap] &
                             ((ds.current_time >= ds.arr(tform, 'Gyr')) &
                              (ds.current_time <= ds.arr(tmerge, 'Gyr'))))

    if slice:
        path = os.path.join(path, 'Slice')
        _mkdir(path)
    else:
        path = os.path.join(path, 'Proj')
        _mkdir(path)

    if width is not None:
        if width == 'Rvir':
            path = os.path.join(path, 'rvir')
        else:
            path = os.path.join(path, '%s%s' % (width[0], width[1]))
        _mkdir(path)

    path = os.path.join(path, '%s%s' % (field[0], field[1]))
    _mkdir(path)
    path = os.path.join(path, 'Axis_%s' % axis)
    _mkdir(path)
    if LogScale:
        path = os.path.join(path, 'LogScale')
    else:
        path = os.path.join(path, 'LinScale')
    _mkdir(path)

    if sinkdynamics > 0:
        s = sink.Sinks()

    part=False
    if (('stars' in field[1]) or ('dm' in field[1])): part=True

    for fn in yt.parallel_objects(files):
        i = files.index(fn)
        if ToPlot[i]:
            c = center
            if hnum is not None:
                h = prog.loc[prog.halo_ts == i+1]
                hid = h.halo_num.item()
            else:
                hid = None
            ds = load_snap.load(fn, haloID=hid, Galaxy=Galaxy, bhID=bhid, radius=width, stars=part, dm=part)
            if bhid is not None:
                bh = ds.sink.loc[ds.sink.ID == bhid]
                c = [bh.x.item(), bh.y.item(), bh.z.item()]

            if hnum is not None:
                if Galaxy:
                    h = ds.gal.gal.loc[hid]
                else:
                    h = ds.halo.halos.loc[hid]
                c = [h.x, h.y, h.z]

            sp = ds.sphere(c, width)

            if slice:
                p = yt.SlicePlot(ds, data_source=sp, axis=axis, fields=field, center=sp.center, width=width)
            else:
                p = yt.ProjectionPlot(ds, data_source=sp, axis=axis, fields=field, weight_field=weight_field,
                                      center=sp.center, width=width, method=method)

            if (plotsinks != [0]):
                if plotsinks == [-1]:
                    BHsToShow = ds.sink.ID
                else:
                    BHsToShow = np.intersect1d(plotsinks , ds.sink.ID)
                for bhnum in BHsToShow:
                    ch = ds.sink.loc[ds.sink.ID == bhnum]
                    if (((c[0] - ch.x.item())**2 +
                        (c[1] - ch.y.item())**2 +
                        (c[2] - ch.z.item())**2) <
                        ((sp.radius.in_units('code_length') / 2)**2)):

                        p.annotate_marker([ch.x.item(), ch.y.item(), ch.z.item()],
                                          marker='.', plot_args={
                                              'color': BHcolor, 's': 100})

                        p.annotate_text(
                            [ch.x.item(), ch.y.item(), ch.z.item()],
                            text=str(ch.ID.item()),
                            text_args={'color': BHcolor},
                            inset_box_args={'alpha': 0.0}
                        )

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
                                    coord_system='data', plot_args={'color':BHcolor})

            if plothalos:
                if plothalos == 'halos':
                    hds = ds.halo.halos
                if plothalos == 'galaxies':
                    hds = ds.gal.gal 
                for hid in hds.index:
                    ch = hds.loc[hid]
                    w = ds.arr(width[0], width[1])
                    if ((ch.m > masshalomin) &
                        (((c[0] - ch.x.item())**2 +
                          (c[1] - ch.y.item())**2 +
                          (c[2] - ch.z.item())**2) <
                         ((w.in_units('code_length') / 2)**2))):

                        p.annotate_sphere([ch.x.item(), ch.y.item(), ch.z.item()],
                                          (ch.rvir.item(), 'Mpc'),
                                          circle_args={'color': BHcolor})

                        p.annotate_text([ch.x.item(), ch.y.item(),
                                         ch.z.item()], text='%s' % hid,
                                         text_args={'color' : BHcolor})

            if plotparticles:
                p.annotate_particles(width)

            if axis_units is None:
                p.annotate_scale(corner='upper_right')
                p.hide_axes()
            else:
                p.annotate_scale(corner='upper_right', draw_inset_box=True)
                p.set_axes_unit(axis_units)
            p.annotate_timestamp(corner='upper_left', time=True, redshift=True, draw_inset_box=True)

            p.set_cmap(field=field, cmap=cmap)
            if cbarunits is None:
                p.hide_colorbar()
            else:
                p.set_unit(field=field, new_unit=cbarunits)
            if cbarbounds is not None:
                p.set_zlim(field=field, zmin=cbarbounds[0], zmax=cbarbounds[1])
            p.set_log(field, log=False)
            if LogScale:
                p.set_log(field, log=True)

            p.set_background_color(field)

            p.set_width(width)

            p.save(path+'/'+str(ds)+'.'+extension)
        # yt.funcs.mylog.setLevel(20)
    return


def plot_profiles(folder='./', center=[0.5,0.5,0.5],
        rbound=[(0.01,'kpc'),(10, 'kpc')], ybound=None, units=None,
        n_bins=128, log=True,
        qtty=[('gas','density'),('deposit','stars_cic'),('deposit','dm_cic')],
        weight_field=('index','cell_volume'), bin_fields=('index', 'radius'),
        hnum=None, bhid=None, Galaxy=False, timestep=None,
        accumulation=False, snap=[-1], filter=None):
    """
    This routine plot the profile for all snapshots

    folder : location to save plots
    center : center of the sphere for the plot, useless if hnum/bhid

    rbound : min/max radius for the profile
    ybound : min/max value to show, in units 'units'
    units : unit to show the profile

    n_bins : number of bin for the radius
    log : if True then loglog plot
    filter (None) : add a particular filter (for instance a temperature floor for gas) CARE WITH UNITS
    example syntax for filter: "obj[('gas','temperature')] < 1e4]" (" and obj are mandatory)

    qtty : list qqty to be profiled, must have the same dimension
    weight_field : weight field for the profile

    hnum : centered on halo hid. If -1, and GalCen present, read it
    bhid : center on a particular BH

    accumulation : sum between 0 and r (cumulative profile)
    snap : list of snapshots to profile
    """

    yt.funcs.mylog.setLevel(40)
    files = utils.find_outputs()
    files.sort()

    path=folder + '/profiles'
    _mkdir(path)

    ToPlot = [True] * len(files)

    if snap != [-1]:
        for i in range(len(files)):
            ToPlot[i] = ((i+1) in snap)

    if ((hnum is not None) and (bhid is not None)):
        print('Please specify only hnum or bhid but not both')
        return

    if hnum is not None:
        t = trees.Forest(Galaxy=Galaxy)
        prog = t.get_main_progenitor(hnum=hnum, timestep=timestep)
        for i in range(len(files)):
            ToPlot[i] = (ToPlot[i]) & (i+1 in np.array(prog.halo_ts))
        if Galaxy:
            path = os.path.join(path, 'Galaxy{:04}_output_{:05}'.format(
                hnum, timestep))
        else:
            path = os.path.join(path, 'Halo{:04}_output_{:05}'.format(
                hnum, timestep))
        _mkdir(path)

    if bhid is not None:
        path = os.path.join(path, 'BH%s' % bhid)
        _mkdir(path)
        s = sink.Sinks(ID=[bhid])
        tform = s.sink[bhid].t.min()
        tmerge = s.sink[bhid].t.max()
        for isnap, f in enumerate(files):
            ds = yt.load(f)
            ToPlot[isnap] = (ToPlot[isnap] &
                             ((ds.current_time >= ds.arr(tform, 'Gyr')) &
                              (ds.current_time <= ds.arr(tmerge, 'Gyr'))))

    path = path + '/'
    for f in qtty:
        path = path + f[0]+f[1]+'+'
    os.system('mkdir ' + path)
    path = path + '/'
    path = path + bin_fields[0] + bin_fields[1]
    os.system('mkdir ' + path)

    part=False
    for field in qtty:
        if (('stars' in field[1]) or ('dm' in field[1])): part=True

    for fn in yt.parallel_objects(files):
      plt.clf()
      i = files.index(fn)
      if ToPlot[i]:
        c = center
        if hnum != None:
            h = prog.loc[prog.halo_ts == i+1]
            hid = h.halo_num.item()
        else:
            hid = None
        
        ds = load_snap.load(fn, stars=part, dm=part, haloID=hid, Galaxy=Galaxy, bhID=bhid, radius=rbound[1])

        p=analysis.profiles(ds, center=center,
            rbound=rbound, n_bins=n_bins,
            log=log,
            qtty=qtty,
            weight_field=weight_field, bin_fields=bin_fields,
            hnum=hid, bhid=bhid, accumulation=accumulation, filter=filter,
            Galaxy=Galaxy)

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

        plt.xlabel(bin_fields[1]+' ['+rbound[0][1]+']')
        plt.ylabel(qtty[0][1]+' ['+(units if units!=None else str(p[qtty[0]].units))+']')
        plt.title('t={:.3f} Gyr'.format(float(ds.current_time.in_units('Gyr'))))

        plt.savefig(path+'/profile{:03}'.format(i+1))
        plt.clf()
        yt.funcs.mylog.setLevel(20)
    return

