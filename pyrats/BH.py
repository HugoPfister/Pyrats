import glob
import scipy.io as ff
import pandas as pd
import yt
import numpy as np
from tqdm import tqdm
import os as os

props=['M','x','y','z','vx','vy','vz','jx_g','jy_g','jz_g','dMB','dME','dM','rho','cs','dv','Esave','jx_bh','jy_bh','jz_bh','spinmag','eps_sink', 'rho_dm', 'rho_stars','frac_dm', 'frac_stars', 'vx_part', 'vy_part', 'vz_part', 'fact_dm', 'fact_stars', 'n_dm', 'n_stars']

os.system('mkdir sinks')
os.system('mv sink_* ./sinks')
if os.path.exists('./sinks/BH00001.csv'):
    os.system('rm ./sinks/BH*')

files=glob.glob('output_*/info*')
ds=yt.load(files[0])

df={tmpprop:[] for tmpprop in props}
df=pd.DataFrame(data=df)
if ds.cosmological_simulation==1:
    df=pd.concat([pd.DataFrame(data={'a':[]}),df],axis=1)
else:
    df=pd.concat([pd.DataFrame(data={'t':[]}),df],axis=1)

files=glob.glob('./sinks/sink*')
files.sort()

for f in tqdm(files):
    p=ff.FortranFile(f)
    p.read_ints(); p.read_ints()
    a=list(p.read_reals('d'))
    scale_l=p.read_reals('d')
    scale_d=p.read_reals('d')
    scale_t=p.read_reals('d')
    bhid=p.read_ints()
    d={tmpprop:p.read_reals('d') for tmpprop in props[:-11]}
    d=pd.DataFrame(data=d, index=bhid)
    d = pd.concat([d, pd.DataFrame(data={tmpprop:p.read_reals('d') for tmpprop in props[-11:]}, index=bhid)], axis=1)
    t=list(p.read_reals('d'))

    d['M']*=scale_d*scale_l**3/2e33
    d['vx']*=scale_l/1e5/scale_t
    d['vy']*=scale_l/1e5/scale_t
    d['vz']*=scale_l/1e5/scale_t
    d['dMB']*=scale_d*scale_l**3/2e33 /scale_t * 3600*24*365
    d['dME']*=scale_d*scale_l**3/2e33 /scale_t * 3600*24*365
    d['dM']*=scale_d*scale_l**3/2e33
    d['rho']*=scale_d/1.67e-24
    d['cs']*=scale_l/1e5/scale_t
    d['dv']*=scale_l/1e5/scale_t
    d['Esave']*=scale_l/1e5/scale_t

    d['vx_part']*=scale_l/1e5/scale_t
    d['vy_part']*=scale_l/1e5/scale_t
    d['vz_part']*=scale_l/1e5/scale_t
    d['rho_stars']*=scale_d/1.67e-24
    d['rho_dm']*=scale_d/1.67e-24
    d['fact_dm']*=scale_d/1.67e-24
    d['fact_stars']*=scale_d/1.67e-24

    for tmpbhid in bhid:      
        if tmpbhid not in df.index:
            df.loc[tmpbhid]=[[]]+[[] for tmpprop in props]
        bh=df.loc[tmpbhid]
        dd=d.loc[tmpbhid]

        if ds.cosmological_simulation==1:
            bh['a']+=a
        else:
            bh['t']+=t

        for tmpprop in props:
            bh[tmpprop]+=[dd[tmpprop]]


for bhid in df.index:
    
    tmp={tmpprop:df.loc[bhid][tmpprop] for tmpprop in props}
    
    if ds.cosmological_simulation==1:
        tmp.update({'a':df.loc[bhid]['a']})
        tmp=pd.DataFrame(data=tmp)
        tmp=pd.concat([tmp, pd.DataFrame({'t':np.copy(ds.cosmology.t_from_z(1/np.copy(tmp.a)-1).in_units('Gyr'))})], axis=1)
    else:
        tmp.update({'t':df.loc[bhid]['t']})
        tmp=pd.DataFrame(data=tmp)
        tmp.t*=scale_t/(1e9*365*24*3600)

    dMdt=tmp.dM[1:]/np.diff(tmp.t)/1e9
    dMdt.index-=1
    dMdt.loc[dMdt.index.max()+1]=0 

    tmp['x']/=ds['boxlen']
    tmp['y']/=ds['boxlen']
    tmp['z']/=ds['boxlen']

    tmp['dM']=dMdt
    tmp.to_csv('./sinks/BH{:05}'.format(bhid)+'.csv', index=False)

tmp={tmpprop:[] for tmpprop in props}
tmp.update({'a':[],'t':[]})
tmp=pd.DataFrame(data=tmp)
tmp.to_csv('./sinks/BH00000.csv', index=False)
