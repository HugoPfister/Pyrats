import yt
from yt.funcs import mylog
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import sink, analysis, load_snap, utils

def plot_snapshots(axis=['x','y','z'], center=None,
                   field=('gas', 'density'),
                   weight_field=('index','ones'), slice=False,
                   contour_field = None,
                   width=[(10, 'kpc')], folder='./',
                   cbarunits=None, cbarbounds=None, cmap='viridis', LogScale=True,
                   hnum=None, timestep=None, Galaxy=False, bhid=None,
                   plothalos=False, masshalomin=1e5,
                   plotsinks=[0], plotparticles=[False,'type'], sinkdynamics=0, text_color='black',
                   snap=[-1], extension='png', method='integrate', old_ramses=False):
    """
    Visualization function, by default it is applied to ALL snapshots

    axis : the projection axis #TODO off axis
    center : center (in code_length units) of the region to show, useless if hnum/bhid

    width : width of the window, default (10, 'kpc'), can be 'Rvir'
        in that case the size is set to the virial radius of each snapshots
    folder : folder to save the images

    field : yt field to show
    weight_field : yt field to use to weight, default ('index', 'ones')
    slice : True/False slice/projection, if True weight_field useless
    contour_field: field to overplot as contour plot (eg. ('gas','density'))

    cbarunits : units for the colorbar, defaut units of field
    cbarbounds : limits for the cbar, in units of cbarunits
    cmap : cmap to use
    LogScale (True) : plot in log

    hnum : ID of the halo, in the last output if timestep is None, you want to center the images
    timestep: timestep to consider the halo
    Galaxy: consider galaxies instead of halos in hnum
    bhid : ID of the sink you want to center the images
    
    plothalos : circles around halos with a mass larger than masshalomin
    masshalomin : in Msun, minimum mass for the halo to show
    plotparticles: overplot the particles as black dots
                    [True,'type'] to overplot and type for the type (cloud, stars...)
    plotsinks : if -1 show sinks and their ID
                if list shows only the asked ones
                if [0] do not show
    sinkdynamics : draw lines to show BH dynamics between [t-sinkdynamics, t+sinkdynamics], in Myr
    text_color : color to show BHs and their ID, halos and colorbar

    snap : list of snapshots you want to show, default ALL (-1)
    """

    mylog.setLevel(40)
    files = utils.find_outputs('./Outputs')
    if ((hnum != None) & (timestep == None)):
        timestep = snap[-1]
    ToPlot, haloid = utils.filter_outputs(snap=snap, hnum=hnum, timestep=timestep, Galaxy=Galaxy, bhid=bhid)
    if sinkdynamics > 0:
        s = sink.Sinks()

    for fn in yt.parallel_objects(files):
        i = files.index(fn)
        if ToPlot[i]:
            print('Images for {} in {}'.format(fn, folder))
            #looking at the larger width to load the largest one
            #instead of loading the snapshot many times
            ds = load_snap.load(fn, verbose=False)
            max_width=np.max([ds.arr(_[0],_[1]) for _ in width]) 
            ds = load_snap.load(fn, haloID=haloid[i], Galaxy=Galaxy, bhID=bhid, radius=max_width, old_ramses=old_ramses, verbose=False)
            if center != None:
                sp = ds.sphere(center, max_width)
            else:
                sp = load_snap.get_sphere(ds, max_width, bhid, haloid[i], Galaxy)
            
            for _w in width:
              for _axis in axis: 
                path = _make_path(folder, hnum, Galaxy, bhid, slice, _w,
                    field, _axis, LogScale, plotsinks, timestep)

                normal = _get_axis(_axis, ds, haloid[i])
                if slice:
                    #p = yt.OffAxisSlicePlot(ds, data_source=sp, normal=normal, fields=field, width=width)
                    p = yt.SlicePlot(ds, data_source=sp, axis=_axis, fields=field, width=_w)
                else:
                    #p = yt.OffAxisProjectionPlot(ds, data_source=sp, normal=normal, fields=field, weight_field=weight_field,
                    #                      width=_w, method=method)
                    p = yt.ProjectionPlot(ds, data_source=sp, axis=_axis, fields=field, weight_field=weight_field,
                                          center=sp.center, width=_w, method=method)

                if (plotsinks != [0]):
                    p = _add_sink(p, plotsinks, ds, sink, sp, text_color, sinkdynamics)
                
                if plothalos != False:
                    p = _add_halos(ds, plothalos, masshalomin, p, text_color)
            
                if plotparticles[0]:
                    p.annotate_particles(_w, ptype=plotparticles[1])

                p = _cleanup_and_save(cmap, p, field, cbarbounds, cbarunits, LogScale, _w,
                            path, extension, ds, text_color)

                if contour_field is not None:
                    p.annotate_contour(contour_field,ncont=[-24,-23],
                    plot_args = {'color':{'yellow','blue','black'}})
            
                p.save(path+'/'+str(ds)+'.'+extension, mpl_kwargs={'pad_inches':0, 'transparent':True})
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
    utils._mkdir(path)

    ToPlot = [True] * len(files)

    if snap != [-1]:
        for i in range(len(files)):
            ToPlot[i] = ((i+1) in snap)

    if ((hnum is not None) and (bhid is not None)):
        raise AttributeError('Please specify only hnum or bhid but not both')
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
        utils._mkdir(path)

    if bhid is not None:
        path = os.path.join(path, 'BH%s' % bhid)
        utils._mkdir(path)
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

def _make_path(folder, hnum, Galaxy, bhid, slice, width, field, axis, LogScale, plotsinks, timestep):
    #path = os.path.join(folder, 'snapshots')
    #utils._mkdir(path)
    path = folder


    #TODO : MAYBE DEAL WITH THIS WHEN WE LOOK AT THE HISTORY OF A GAL/HALO
    #if hnum is not None:
    #    if Galaxy:
    #        path = os.path.join(path, 'Galaxy{:04}_output_{:05}'.format(
    #            hnum, timestep))
    #    else:
    #        path = os.path.join(path, 'Halo{:04}_output_{:05}'.format(
    #            hnum, timestep))
    #    utils._mkdir(path)

    if bhid is not None:
        path = os.path.join(path, 'BH%s' % bhid)
        utils._mkdir(path)

    #if slice:
    #    path = os.path.join(path, 'Slice')
    #    utils._mkdir(path)
    #else:
    #    path = os.path.join(path, 'Proj')
    #    utils._mkdir(path)

    if width is None:
        raise AttributeError('Specify a width')
    else:
        #if width == 'Rvir':
        #    path = os.path.join(path, 'rvir')
        #else:
        path = os.path.join(path, '%s%s' % (width[0], width[1]))
        utils._mkdir(path)
    

    path = os.path.join(path, '%s%s' % (field[0], field[1]))
    utils._mkdir(path)
    path = os.path.join(path, 'Axis_%s' % axis)
    utils._mkdir(path)
    #if LogScale:
    #    path = os.path.join(path, 'LogScale')
    #else:
    #    path = os.path.join(path, 'LinScale')
    #utils._mkdir(path)

    #if plotsinks == [0]:
    #    path = os.path.join(path, 'NoBH')
    #elif plotsinks == [-1]:
    #    path = os.path.join(path, 'AllBH')
    #else:
    #    path = os.path.join(path, 'SomeBH')
    #utils._mkdir(path)
    return path


def _add_sink(p, plotsinks, ds, sink, sp, text_color, sinkdynamics):
    if plotsinks == [-1]:
        BHsToShow = ds.sink.ID
    else:
        BHsToShow = np.intersect1d(plotsinks , ds.sink.ID)

    for bhnum in BHsToShow:
       ch = ds.sink.loc[ds.sink.ID == bhnum]
       if (((float(sp.center[0].to('code_length')) - ch.x.item())**2 +
           (float(sp.center[1].to('code_length')) - ch.y.item())**2 +
           (float(sp.center[2].to('code_length')) - ch.z.item())**2) <
           ((sp.radius.in_units('code_length') / 2)**2)):

           p.annotate_marker([ch.x.item(), ch.y.item(), ch.z.item()],
                             marker='.', plot_args={
                                 'color': text_color, 's': 100})

           p.annotate_text(
               [ch.x.item(), ch.y.item(), ch.z.item()],
               text=str(ch.ID.item()),
               text_args={'color': text_color},
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
                       coord_system='data', plot_args={'color':text_color})

    return p

def _add_halos(ds, plothalos, masshalomin, p, text_color):
##this function is probably broken as is. 
    if plothalos == 'halos':
        hds = ds.halo.halos
    if plothalos == 'galaxies':
        hds = ds.gal.gal 
    for hid in hds.index:
        ch = hds.loc[hid]
        w = ds.arr(width[0], width[1])
        if ((ch.m > masshalomin) &
            (((float(sp.center[0].to('code_length')) - ch.x.item())**2 +
              (float(sp.center[1].to('code_length')) - ch.y.item())**2 +
              (float(sp.center[2].to('code_length')) - ch.z.item())**2) <
             ((w.in_units('code_length') / 2)**2))):

            p.annotate_sphere([ch.x.item(), ch.y.item(), ch.z.item()],
                              (ch.rvir.item(), 'Mpc'),
                              circle_args={'color': text_color})

            p.annotate_text([ch.x.item(), ch.y.item(),
                             ch.z.item()], text='%s' % hid,
                             text_args={'color' : text_color})
    
    return p


def _cleanup_and_save(cmap, p, field, cbarbounds, cbarunits, LogScale, width,
                path, extension, ds, text_color):

   my_cmap = plt.matplotlib.cm.get_cmap(cmap)
   my_cmap.set_bad(my_cmap(0))
   p.set_cmap(field=field, cmap=my_cmap)
   p.set_background_color(field=field)
   if cbarbounds is not None:
       if cbarunits is None:
           raise AttributeError('Specify a units for the boundaries of the colorbar')
       p.set_unit(field=field, new_unit=cbarunits)
       p.set_zlim(field=field, zmin=cbarbounds[0], zmax=cbarbounds[1])
   if LogScale:
       p.set_log(field, log=True)
   else:
       p.set_log(field, log=False)

   p.hide_colorbar()
   p.hide_axes()

   p.annotate_scale(draw_inset_box=True, corner='lower_right', text_args={'size':28})
   p.annotate_timestamp(draw_inset_box=True, time=True, redshift=True,
           corner='lower_left', text_args={'color':'white', 'size':28})
   p.set_width(width)

   mylog.info('Saving ',path+'/'+str(ds)+'.'+extension)
   #this line is here to effectively apply z_lim, units etc....
   p.save(path+'/'+str(ds)+'.'+extension, mpl_kwargs={'pad_inches':0, 'transparent':True})

   plot = p.plots[field]
   cbmap = plot.cb.mappable
   current_cmap = cbmap.get_cmap()
   current_cmap.set_bad(current_cmap(0))
   cb_axes = inset_axes(plot.axes, width='80%', height='3%', loc=9)
   cb_axes.tick_params(axis='x', which='major', length=4,
                     labelcolor=text_color, direction='in', top=True)
   cbar = plot.figure.colorbar(cbmap, cax=cb_axes, 
                     orientation='horizontal')
   label = plot.cax.get_ylabel()
   if (('Stars' in label) or ('Dm' in label) or ('Star' in label)):
       label = label.replace('Stars\ CIC', 'Stellar')
       label = label.replace('Stars', 'Stellar')
       label = label.replace('Star\ CIC', 'Stellar')
       label = label.replace('Star', 'Stellar')
       label = label.replace('Dm\ CIC', 'DM')
       label = label.replace('Dm', 'DM')
   cbar.set_label(label, color=text_color)
   cbar.ax.xaxis.label.set_font_properties(p._font_properties)
   p._font_properties.set_size(25)
   cbar.ax.tick_params(labelsize=25)

   return p

def _get_axis(axis, ds, haloid):
    if axis == 'x':
        return [1,0,0]
    elif axis == 'y':
        return [0,1,0]
    elif axis == 'z':
        return [0,0,1]
    elif ((axis == 'L') & (haloid in ds.gal.gal.index)):
        #L = ds.gal.gal.loc[haloid, ['Lx','Ly','Lz']].tolist()
        L = [1,0,0]
        return L / np.linalg.norm(L) 
    elif ((axis == 'Lperp') & (haloid in ds.gal.gal.index)):
        L = ds.gal.gal.loc[haloid, ['Lx','Ly','Lz']].tolist()
        L = np.cross(L, [1,0,0])
        return L / np.linalg.norm(L) 
    else:
        ValueError('Please return a correct axis')
        
