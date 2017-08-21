import yt
import glob
from tqdm import tqdm
from time import time
import os as os

import halos
import trees
import sink


def plot_all_snapshots(axis='z', field=('deposit', 'all_density'), folder='./', cbarmin=None, cbarmax=None, weight_field=('index', 'ones')):
    """
    Plot a map of all the snapshots for this simulations
    Parameters
    ----------
    axis ('z') : Projection axis
    field ('deposit','all_density') : Field that will be projected
    folder='./' : Folder to put the output (a subfolder 'snapshots' will be created with the outputs inside)
    cbarmin/cbarmax (None/None) : Set limits to the colorbar
    weight_field ('index','ones'): Field used to weight the map
    """
    files = glob.glob('output_*/info*')
    for dd in tqdm(files):
        ds = yt.load(dd)
        p = yt.ProjectionPlot(ds, axis=axis, fields=field,
                              weight_field=weight_field, axes_unit=('Mpccm'))
        p.set_cmap(field=field, cmap='viridis')
        p.annotate_timestamp(corner='upper_left', time=False, redshift=True)
        p.annotate_scale(corner='upper_right')
        p.set_zlim(field=field, zmin=cbarmin, zmax=cbarmax)
        os.system('mkdir ' + folder + '/snapshots')
        p.save(folder + '/snapshots/')
    return


def plot_halo_history(hnum, axis='z', field=('deposit', 'all_density'), folder='./', weight_field=('index', 'ones'), slice=False,
                      size=None, cmap='viridis', limits=[0, 0], plotsinks=False, units=None, plothalos=False, masshalomin=1e10):
    """
    TODO but not urgent: gather this and plot_bh_history in a function with a switch for BH/halos

    Plot a map, at eauch output of the halo with ID hnum at the last output of the simulation (computed with HaloMaker)
    Parameters
    ----------
    hnum: ID, for the last output, of the halo you want to plot all the progenitors
    Other paramters are the one used in halo.plot_halo
    """

    files = glob.glob('output_*/info*')
    files.sort()
    ds = yt.load(files[-1])
    t = trees.Forest(LoadGal=False)
    hid = int(t.trees[(t.trees.halo_ts == t.trees.halo_ts.max())
                      & (t.trees.halo_num == hnum)].halo_id)
    path = folder + '/Halo' + str(hnum)
    os.system('mkdir ' + path)
    if size != None:
        path = path + '/' + str(size[0]) + size[1]
        os.system('mkdir ' + path)
    path = path + '/' + field[0] + field[1]
    os.system('mkdir ' + path)
    prog_id = [_ for _ in t.get_main_progenitor(hid).halo_num]
    h = halos.HaloList(ds)
    if size == None:
        rvirfin = (h.halos.rvir[prog_id[-1]] *
                   (1 + ds.current_redshift), 'Mpccm')
    else:
        rvirfin = size

    for i in tqdm(range(len(files))):
        ds = yt.load(files[-i - 1])
        h = halos.HaloList(ds)
        h.plot_halo(prog_id[-i - 1], axis=axis, folder=path, field=field, r=rvirfin,
                    weight_field=weight_field, cmap=cmap, limits=limits, plotsinks=plotsinks, units=units)
    return


def plot_bh_history(bhid, axis='z', field=('deposit', 'all_density'), folder='./', weight_field=('index', 'ones'), slice=False,
                    size=(1, 'kpccm'), cmap='viridis', limits=[0, 0], units=None):
    """
    TODO but not urgent: gather this and plot_bh_history in a function with a switch for BH/halos

    Plot a map, at each output, of BH with ID bhid at the last output of the simulation
    Parameters
    ----------
    bhid: ID, for the last output, of the bh you want to plot
    Other paramters are the one used in halo.plot_halo (the routine halo.plot halo has almost been copy/paste)
    """

    files = glob.glob('output_*/info*')
    files.sort()
    ds = yt.load(files[-1])
    path = folder + '/BH' + str(bhid)
    os.system('mkdir ' + path)
    path = path + '/' + str(size[0]) + size[1]
    os.system('mkdir ' + path)
    path = path + '/' + field[0] + field[1]
    os.system('mkdir ' + path)

    if 'cm' in size[1]:
        r = (size[0] * (1 + ds.current_redshift), size[1])
    else:
        r = size

    for i in tqdm(range(len(files))):
        ds = yt.load(files[-i - 1])
        ds.sink = sink.get_sinks(ds)

        bh = ds.sink.loc[ds.sink.ID == bhid]
        c = [bh.x.item(), bh.y.item(), bh.z.item()]

        if 'stars' in field[1]:
            yt.add_particle_filter(
                "stars", function=fields.stars, filtered_type="all", requires=["particle_age"])
            ds.add_particle_filter("stars")
        if 'dm' in field[1]:
            yt.add_particle_filter(
                "dm", function=fields.dm, filtered_type="all")
            ds.add_particle_filter("dm")

        dd = ds.sphere(c, r)

        if slice:
            p = yt.SlicePlot(ds, data_source=dd, axis=axis,
                             fields=field, center=c)
        else:
            p = yt.ProjectionPlot(ds, data_source=dd, axis=axis,
                                  fields=field, center=c, weight_field=weight_field)

        p.set_width((float(dd.radius.in_units('kpccm')), str('kpccm')))
        if limits != [0, 0]:
            p.set_zlim(field, limits[0], limits[1])
            if limits[1] / limits[0] > 50:
                p.set_log(field, log=True)

        if units != None:
            p.set_unit(field=field, new_unit=units)

        for bhnum in ds.sink.ID:
            ch = ds.sink.loc[ds.sink.ID == bhnum]
            if (((bh.x.item() - ch.x.item())**2 + (bh.y.item() - ch.y.item())**2 + (bh.z.item() - ch.z.item())**2) <
                    ((dd.radius.in_units('code_length') / 2)**2)):
                p.annotate_marker([ch.x.item(), ch.y.item(), ch.z.item(
                )], marker='.', plot_args={'color': 'black', 's': 100})
                p.annotate_text([ch.x.item(), ch.y.item(), ch.z.item()], text=str(
                    ch.ID.item()), text_args={'color': 'black'})

        p.annotate_timestamp(corner='upper_left', time=True, redshift=True)
        p.set_cmap(field=field, cmap=cmap)
        p.annotate_scale(corner='upper_right')
        p.save(path + '/' + str(ds) + '_bh' + str(bhid))

    return
