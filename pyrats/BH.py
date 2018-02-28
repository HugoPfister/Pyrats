import glob
import scipy.io as ff
import pandas as pd
import yt
import numpy as np
import os as os

cic_levelmax = 10



props=['M','x','y','z','vx','vy','vz','jx_g','jy_g','jz_g','dMB','dME','dM','rho','cs','dv','Esave','jx_bh','jy_bh','jz_bh','spinmag','eps_sink', 'rho_stars', 'rho_dm',  'vx_stars', 'vy_stars', 'vz_stars', 'vx_dm', 'vy_dm', 'vz_dm', 'n_stars', 'n_dm', 'rho_lowspeed_stars', 'rho_lowspeed_dm', 'fact_fast_stars', 'fact_fast_dm']

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

dx=float(ds.length_unit.in_units('pc')/2**ds.max_level*(1+ds.current_redshift))
dx_dm=float(ds.length_unit.in_units('pc')/2**cic_levelmax*(1+ds.current_redshift))

for f in files:
    p=ff.FortranFile(f)
    p.read_ints(); p.read_ints()
    a=list(p.read_reals('d'))
    scale_l=p.read_reals('d')
    scale_d=p.read_reals('d')
    scale_t=p.read_reals('d')
    bhid=p.read_ints()
    d={tmpprop:p.read_reals('d') for tmpprop in props[:-14]}
    d=pd.DataFrame(data=d, index=bhid)
    d = pd.concat([d, pd.DataFrame(data={tmpprop:p.read_reals('d') for tmpprop in props[-14:]}, index=bhid)], axis=1)
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

    d['vx_stars']*=scale_l/1e5/scale_t
    d['vy_stars']*=scale_l/1e5/scale_t
    d['vz_stars']*=scale_l/1e5/scale_t
    d['vx_dm']*=scale_l/1e5/scale_t
    d['vy_dm']*=scale_l/1e5/scale_t
    d['vz_dm']*=scale_l/1e5/scale_t
    d['rho_stars']*=scale_d/1.67e-24
    d['rho_dm']*=scale_d/1.67e-24
    d['rho_lowspeed_stars']*=scale_d/1.67e-24
    d['rho_lowspeed_dm']*=scale_d/1.67e-24
    d['fact_fast_stars']*=scale_d/1.67e-24
    d['fact_fast_dm']*=scale_d/1.67e-24

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
    tmp['vsink_rel_stars'] = np.sqrt((tmp['vx_stars']-tmp['vx'])**2+(tmp['vy_stars']-tmp['vy'])**2+(tmp['vz_stars']-tmp['vz'])**2)
    tmp['vsink_rel_dm'] = np.sqrt((tmp['vx_dm']-tmp['vx'])**2+(tmp['vy_dm']-tmp['vy'])**2+(tmp['vz_dm']-tmp['vz'])**2)
    tmp['rinf_stars'] = (tmp.M / 1e7) / (tmp.vsink_rel_stars / 200)**2
    tmp['rinf_dm'] = (tmp.M / 1e7) / (tmp.vsink_rel_dm / 200)**2
    
    CoulombLog = np.maximum(np.zeros(len(tmp.t)), np.log(4*dx/tmp.rinf_stars))
    tmp['a_stars_slow']=4*np.pi*(6.67e-8)**2*tmp.M*2e33*tmp.rho_lowspeed_stars*1.67e-24*CoulombLog/(tmp.vsink_rel_stars*1e5)**2*3600*24*365*1e6/1e5
    CoulombLog = np.minimum(np.zeros(len(tmp.t)), tmp.rinf_stars-4*dx) / (tmp.rinf_stars - 4*dx) 
    tmp['a_stars_fast']=4*np.pi*(6.67e-8)**2*tmp.M*2e33*tmp.fact_fast_stars*1.67e-24*CoulombLog/(tmp.vsink_rel_stars*1e5)**2*3600*24*365*1e6/1e5

    CoulombLog = np.maximum(np.zeros(len(tmp.t)), np.log(4*dx/tmp.rinf_dm))
    tmp['a_dm_slow']=4*np.pi*(6.67e-8)**2*tmp.M*2e33*tmp.rho_lowspeed_dm*1.67e-24*CoulombLog/(tmp.vsink_rel_dm*1e5)**2*3600*24*365*1e6/1e5
    CoulombLog = np.minimum(np.zeros(len(tmp.t)), tmp.rinf_dm-4*dx) / (tmp.rinf_dm - 4*dx) 
    tmp['a_dm_fast']=4*np.pi*(6.67e-8)**2*tmp.M*2e33*tmp.fact_fast_dm*1.67e-24*CoulombLog/(tmp.vsink_rel_dm*1e5)**2*3600*24*365*1e6/1e5

    M=tmp.dv / tmp.cs
    tmp['rinf_gas'] = (tmp.M / 1e7) / (tmp.dv**2 + tmp.cs**2)/200**2
    CoulombLog = np.minimum(np.zeros(len(tmp.t)), tmp.rinf_gas-4*dx) / (tmp.rinf_gas - 4*dx) 
    fudge=M
    fudge.loc[M < 0.95] = 1/M**2*(0.5*np.log((1+M)/(1-M)) - M)
    fudge.loc[(M >= 0.95) & (M <= 1.007)] = 1
    fudge.loc[M > 1.007] = 1/M**2*(0.5*np.log(M**2-1) + 3.2)
    tmp['a_gas']=4*np.pi*(6.67e-8)**2*tmp.M*2e33*tmp.rho*1.67e-24/(tmp.cs*1e5)**2*fudge*(3600*24*365*1e6)/1e5*CoulombLog

    tmp.to_csv('./sinks/BH{:05}'.format(bhid)+'.csv', index=False)

tmp={tmpprop:[] for tmpprop in props}
tmp.update({'a':[],'t':[]})
tmp=pd.DataFrame(data=tmp)
tmp.to_csv('./sinks/BH00000.csv', index=False)
