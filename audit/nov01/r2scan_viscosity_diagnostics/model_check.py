"""Independent model comparison and common-cutoff spectral diagnostics."""
from analyze import *
out={}
for T in [1873,1941,2000,2100]:
    fp=OUT/f'input_{T}.npz'
    if fp.exists():
        a=np.load(fp);x=a['seq'];p=a['pref']
    else:
        root=BASE/(f'Ti-viscosity-r2SCAN/scan_{T}K_wsl2/scan' if T in [1873,2000] else f'models_check_cluster/results/viscosity/dpa4/neo/finetune/AlTi/{T}K/scan')
        fs=sorted((root/'03_stress_autocorr/c_Ti_1.00').glob('visc*/pressure_*.data'),key=lambda f:int(f.parent.name[4:]))
        x=np.array([channels(np.loadtxt(f)[:,1:]*1e5) for f in fs]);p=np.array([np.full(5,volume(f.parent/'mol.data')*1e-30/(1.380649e-23*T)) for f in fs])
    s=compute_spectrum(x.reshape(-1,x.shape[-1]),prefactors=p.ravel(),timestep=1e-15)
    out[str(T)]={'models':{},'dense_cutoff':[]}
    for label,model in [('Lorentz',LorentzModel()),('ExpPoly02',ExpPolyModel([0,2])),('ExpPoly024',ExpPolyModel([0,2,4]))]:
        r=auto(s,model);out[str(T)]['models'][label]=pack(r)
        if label=='Lorentz':
            out[str(T)]['lorentz_history']=[{**pack(h),'weight':float(h['fcut_weight'])} for h in r.history]
        save('model_check.json',out);print('MODEL',T,label,pack(r),flush=True)
    for fc in np.linspace(.65,2.5,38):out[str(T)]['dense_cutoff'].append(fixed(s,float(fc)))
    save('model_check.json',out)
print('MODEL_DONE',flush=True)
