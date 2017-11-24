import pyrats
import yt
from tqdm import tqdm
import os as os
import glob as glob
import pandas as pd

#write the path to the folder that contains the outputs
os.chdir('LOC')

#First output with a BH (if you relaxed your galaxy before planting BH)
imin = 1

files=glob.glob('output*/info*')
nsnaps=len(files)
tmp={'t':[], 'cx':[], 'cy':[], 'cz':[], 'x':[], 'y':[], 'z':[]}
for i in tqdm(range(nsnaps-1)):
    i=i+imin
    #ds=pyrats.load(i, stars=True, bh=True, dm=True)
    ds=pyrats.load(i, bh=True, dm=True)
    tmp['t']+=[float(ds.current_time.in_units('Gyr'))]
    sp=ds.sphere([0.5,0.5,0.5], (1000, 'pc'))
    #c=sp.argmax(('deposit','stars_cic'))
    c=sp.argmax(('deposit','dm_cic'))
    tmp['cx']+=[float(c[0])]
    tmp['cy']+=[float(c[1])]
    tmp['cz']+=[float(c[2])]
    tmp['x']+=[ds.sink.x.item()]
    tmp['y']+=[ds.sink.y.item()]
    tmp['z']+=[ds.sink.z.item()]
tmp=pd.DataFrame(data=tmp) 
tmp.to_csv('./GalCenter.csv', index=False)
