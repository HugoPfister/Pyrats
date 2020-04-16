import yt
import os as os
from glob import glob
import pandas as pd
from tqdm import tqdm
import pyrats

sims=[]
init=1


for s in sims:
    cx=[]; cy=[]; cz=[]; t=[]; x=[]; y=[]; z=[]
    os.chdir('/home/pfister/scratch/'+s)
    files=glob('output_*')
    #for i in tqdm(range(3)):
    for i in tqdm(range(len(files)-init+1)):
        ds=pyrats.load(i+init, dm=True, bh=True, stars=True)

        d=ds.all_data()
        
        t+=[float(ds.current_time.in_units('Gyr'))]
        
        ccx=float(d['dm','particle_position_x'].mean())
        ccy=float(d['dm','particle_position_y'].mean())
        ccz=float(d['dm','particle_position_z'].mean())
        for r in [3000,2500,2000,1500,1000,750,500,300]:
            sp=ds.sphere([ccx,ccy,ccz], (r,'pc'))
            ccx=float(sp['dm','particle_position_x'].mean())
            ccy=float(sp['dm','particle_position_y'].mean())
            ccz=float(sp['dm','particle_position_z'].mean())

        #arg=d['deposit','stars_cic'].argmax()
        #cx+=[float(d['index','x'][arg])]
        #cy+=[float(d['index','y'][arg])]
        #cz+=[float(d['index','z'][arg])]
        cx+=[ccx]
        cy+=[ccy]
        cz+=[ccz]
        x+=[float(d['sink','particle_position_x'].mean())]
        y+=[float(d['sink','particle_position_y'].mean())]
        z+=[float(d['sink','particle_position_z'].mean())]

    gal={'cx':cx, 'cy':cy, 'cz':cz, 't':t, 'x':x, 'y':y, 'z':z}
    gal=pd.DataFrame(data=gal)
    gal.to_csv('GalCenter.csv', index=False)

