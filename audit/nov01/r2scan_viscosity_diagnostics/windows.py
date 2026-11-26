"""Paired half-trajectory diagnostic, preserving each replica as one group."""
from analyze import *
out={}
for T in [1873,1941,2000,2100]:
    fp=OUT/f'input_{T}.npz'
    if fp.exists():a=np.load(fp);x=a['seq'];p=a['pref']
    else:
        root=BASE/f'models_check_cluster/results/viscosity/dpa4/neo/finetune/AlTi/{T}K/scan'
        fs=sorted((root/'03_stress_autocorr/c_Ti_1.00').glob('visc*/pressure_*.data'),key=lambda f:int(f.parent.name[4:]))
        x=np.array([channels(np.loadtxt(f)[:,1:]*1e5) for f in fs]);p=np.array([np.full(5,volume(f.parent/'mol.data')*1e-30/(1.380649e-23*T)) for f in fs])
    halves=[[compute_spectrum(z[:,sl],prefactors=f,timestep=1e-15) for z,f in zip(x,p)] for sl in [slice(0,10000),slice(10000,20000)]]
    merged=[merge(ss) for ss in halves];e={'fixed':{}}
    for fc in [1.25,1.5,2.]:e['fixed'][str(fc)]=[fixed(s,fc) for s in merged]
    e['paired_jackknife']={}
    for fc in [1.25,1.5]:
        vals=[]
        for i in range(len(x)):
            aa=[fixed(merge([s for j,s in enumerate(ss) if j!=i]),fc)['eta'] for ss in halves];vals.append(aa[1]-aa[0])
        full=e['fixed'][str(fc)];delta=full[1]['eta']-full[0]['eta'];se=jk(vals)
        e['paired_jackknife'][str(fc)]={'second_minus_first':delta,'se':se,'z':delta/se,'leaveout_differences':vals}
    if T==1941:
        vals=[];aa=[pack(auto(s)) for s in merged]
        for i in range(len(x)):
            yy=[auto(merge([s for j,s in enumerate(ss) if j!=i])).acint*1000 for ss in halves];vals.append(yy[1]-yy[0])
        delta=aa[1]['eta']-aa[0]['eta'];se=jk(vals)
        e['paired_auto_jackknife']={'second_minus_first':delta,'se':se,'z':delta/se,'leaveout_differences':vals}
    out[str(T)]=e;save('windows.json',out);print('WINDOWS',T,e['paired_jackknife'],e.get('paired_auto_jackknife'),flush=True)
