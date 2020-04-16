import yt
from yt.utilities.logger import ytLogger as mylog
import numpy as np
import os as os
import pandas as pd
from tqdm import tqdm
import pyrats


def main():
    r_gal = 1 #distance looked for BH to galaxies
    r_typ = 'r90star' #
    matching_folder = 'matching'
    output_folder = 'Outputs'
    n_star_min = 200
    if not os.path.exists(matching_folder):
        os.mkdir(matching_folder)
        mylog.info('Making the matching folder {}'.format(matching_folder))
    matching_folder = os.path.join(matching_folder,'{}_{}'.format(r_gal,r_typ))
    if not os.path.exists(matching_folder):
        os.mkdir(matching_folder)
        os.mkdir(matching_folder+'/gal')
        os.mkdir(matching_folder+'/sinks')
        mylog.info('Making the matching folder {}'.format(matching_folder))
    fvir = [r_gal, r_typ]

    files = pyrats.utils.find_outputs(output_folder)
    files.sort()
    files = np.array(files)
    mylog.info('Found {} outputs from {} to {}'.format(files.size, files[0], files[-1]))
    for i_f,f in enumerate(tqdm(files)):
        ds  = pyrats.load(f, verbose=False)
        matching(ds, fvir, n_star_min,  matching_folder)
    
    return


def matching(ds, fvir, n_star_min, matching_folder):
     L = ds.length_unit.in_units('Mpc')
  
     mylog.info('Matching sinks to galaxies')
     # match sinks to galaxies
     check = False
     for galID in ds.gal.gal.loc[
            ds.gal.gal.nstar >= n_star_min].sort_values(['level','mstar'], ascending=[True,False]).index:
         g = ds.gal.gal.loc[galID]
         d = np.sqrt((g.xstar.item() - ds.sink.x)**2 + (g.ystar.item() - ds.sink.y)** 2 +
             (g.zstar.item() - ds.sink.z)**2)
         #bhid = ds.sink.loc[((d * L) < g.r.item()*fvir[2]) & (ds.sink.mgal < g.m.item())].index
         bhid = ds.sink.loc[((d * L) < g[fvir[1]].item()*fvir[0]) & (ds.sink.mgal < g.mstar.item())].index
         #bhid = sinks.loc[((d * L) < g.r.item()*fvir[2])].index
         if len(bhid) > 0:
             check = True
             ds.sink.loc[bhid, 'mgal'] = g.mstar.item()
             ds.sink.loc[bhid, 'galID'] = galID
             ds.sink.loc[bhid, 'mbulge'] = g.mbulge.item()
             ds.sink.loc[bhid, 'sigma_bulge'] = g.sigma_bulge.item()

     for galID in ds.sink.loc[ds.sink.galID > 0].galID.unique():
         arg = ds.sink.loc[ds.sink.galID == galID].M.idxmax()
         ds.gal.gal.loc[ds.sink.loc[arg].galID.item(), 'msink'] = ds.sink.loc[arg].M.item()
         ds.gal.gal.loc[ds.sink.loc[arg].galID.item(), 'bhid'] = ds.sink.loc[arg].ID.item()
         dist = ((np.array(ds.sink.loc[arg,['x','y','z']].tolist()) -\
            np.array(ds.gal.gal.loc[galID,['xstar','ystar','zstar']]))**2).sum()**0.5
         ds.gal.gal.loc[ds.sink.loc[arg].galID.item(), 'BH_dist'] = dist*L*1e3 #in kpc 
    
     if check:
        ds.sink[['hid','mhalo','galID','mgal','mbulge','sigma_bulge']].to_hdf(
             matching_folder+'/sinks/{}'.format(ds.ids), key='hdf5')   
        ds.gal.gal[['bhid','msink','BH_dist']].to_hdf(matching_folder+'/gal/{}'.format(ds.ids), key='hdf5')
  
     return
 
main()

