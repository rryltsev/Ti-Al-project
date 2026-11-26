"""Validate completed 90 ps points and publish data for Fig. 3 only when complete."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'audit/oct29/xc_liquid_density_extended';OUT=RUN/'analysis';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
sys.path.insert(0,str(ROOT/'audit/melting'))
import numpy as np
from scipy.stats import t as student
from ase.io.trajectory import Trajectory
from analyze_melting import order
sys.stdout.reconfigure(encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
cfg=load(RUN/'protocol.json');submission=load(RUN/'submission.json')
collection=load(RUN/'collection.json')
assert collection['status']=='verified'
manifest=load(RUN/'collection-manifest.json')
assert all(sha(RUN/'collected'/n)==d['sha256'] for n,d in manifest.items())

def stats(a,dt=.01):
    a=np.asarray(a);assert np.isfinite(a).all();checks={}
    for ps in [3,5,6,9,10,15]:
        n=round(ps/dt)
        if len(a)%n==0 and len(a)//n>=6:
            b=a.reshape(-1,n).mean(axis=1)
            checks[str(ps)]=dict(sem=float(b.std(ddof=1)/np.sqrt(len(b))),blocks=len(b),means=b.tolist())
    key=max(checks,key=lambda k:checks[k]['sem']);v=checks[key]
    mean=float(a.mean());half=float(student.ppf(.975,v['blocks']-1)*v['sem'])
    return dict(mean=mean,sem=v['sem'],ci95=[mean-half,mean+half],selected_block_ps=float(key),block_checks=checks,samples=len(a))

points=[]
for s in cfg['states']:
    folder=RUN/'collected/results'/s['id'];meta_path=folder/'summary.json'
    if not meta_path.exists():continue
    meta=load(meta_path);m=cfg['models'][str(s['checkpoint'])]
    assert meta['status']=='completed' and meta['state']==s
    assert meta['job_id']==submission['job_id'] and meta['model_sha256']==m['sha256']
    assert meta['protocol_sha256']==sha(RUN/'protocol.json')
    assert meta['worker_sha256']==sha(RUN/'workflow.py') and meta['common_sha256']==sha(RUN/'md_common.py')
    for stage,steps in [('equilibration',s['equil_steps']),('production',s['production_steps'])]:
        rr=[json.loads(line) for line in (folder/(stage+'-thermo.jsonl')).read_text().splitlines()]
        assert [r['step'] for r in rr]==list(range(0,steps+1,10))
    rows=rr;r=rows[1:];rho=np.array([d['rho_g_cm3'] for d in r]);N=s['N']
    phase=[]
    with Trajectory(str(folder/'production.traj')) as tr:
        assert len(tr)==901
        frames=[a for a in tr]
        assert all(len(a)==N and np.all(a.numbers==22) for a in frames)
        calc_rho=np.array([a.get_masses().sum()/a.get_volume()*1.66053906660 for a in frames])
        np.testing.assert_allclose(calc_rho,[d['rho_g_cm3'] for d in rows[::10]],rtol=0,atol=1e-10)
        for i in range(0,len(frames),20):
            a=frames[i];lengths=np.diag(a.cell.array)
            np.testing.assert_allclose(a.cell.array,np.diag(lengths),atol=1e-10,rtol=0)
            q,qbar,bonds=order(a.positions%lengths,lengths)
            phase.append(dict(time_ps=i*.1,qbar6_mean=float(qbar.mean()),solid_like_fraction=float(np.mean(qbar>.25)),coherent_fraction=float(np.mean(bonds>=7))))
        frac=np.array([a.get_scaled_positions(wrap=True) for a in frames]);cells=np.array([a.cell.array for a in frames])
    delta=np.diff(frac,axis=0);delta-=np.round(delta)
    cart=np.einsum('tni,tij->tnj',delta,.5*(cells[1:]+cells[:-1]));cart-=cart.mean(axis=1,keepdims=True)
    xyz=np.concatenate([np.zeros_like(cart[:1]),np.cumsum(cart,axis=0)])
    msd=float(np.mean(np.sum((xyz[100:]-xyz[:-100])**2,axis=-1)))
    pressures=stats([d['P_GPa'] for d in r]);temperature=float(np.mean([d['T_K'] for d in r])*N/(N-1))
    halves=float(rho[4500:].mean()-rho[:4500].mean());rs=stats(rho)
    late_fraction=float(np.mean([p['solid_like_fraction'] for p in phase if p['time_ps']>80]))
    warnings=[]
    if abs(temperature/s['T_K']-1)>.02:warnings.append('Temperature differs by more than 2% from target')
    if not pressures['ci95'][0]<=cfg['P_bar']*.0001<=pressures['ci95'][1]:warnings.append('Mean pressure excludes target 1 bar in block 95% interval; SEM excludes protocol bias')
    if abs(halves)>.002*rs['mean']:warnings.append('Density drift between halves exceeds 0.2%')
    if msd<5 or late_fraction>.2:warnings.append('Liquid phase needs review: low diffusion or substantial crystalline order')
    p=dict(id=s['id'],functional=s['functional'],key=s['key'],checkpoint=s['checkpoint'],model_sha256=m['sha256'],
           T_K=s['T_K'],phase_initial='liquid',N=N,production_ps=90.,rho=rs,source=str(folder),job_id=meta['job_id'],
           qc=dict(T_3N_minus_3_K=temperature,P_GPa=pressures,MSD_10ps_A2=msd,solid_like_fraction_last10=late_fraction,
                   phase_samples=phase,rho_second_half_minus_first=halves,rho_last30_minus_full=float(rho[-3000:].mean()-rho.mean()),
                   rho_100fs_minus_10fs=float(rho[9::10].mean()-rho.mean()),warnings=warnings))
    points.append(p)
    print(p['id'],f"rho={rs['mean']:.7f} SEM={rs['sem']:.7f}",'warnings:',warnings,flush=True)
result=dict(job_id=submission['job_id'],protocol_sha256=sha(RUN/'protocol.json'),analysis_sha256=sha(__file__),points=points,
            completed_points=len(points),planned_points=14,archived_points=load(RUN/'archived_liquid_points.json'),
            notes=['Primary density is mean instantaneous mass/volume from 90 ps, sampled every 10 fs; exclude t=0.',
                   'Maximum SEM over 3,5,6,9,10,15 ps blocks with at least six blocks. Not total uncertainty.',
                   'Original 30 ps data remain separate because their 100 fs cadence can alias barostat oscillations.',
                   'No smoothing or selection of windows according to curve shape. PBE and solid curves remain unchanged.',
                   'Structural qbar6 uses 14 nearest neighbors, threshold .25, every 2 ps; not an equilibrium phase boundary.'])
save(OUT/'statistics.json',result)
lines=['PBEsol / r2SCAN: расширенная статистика плотности жидкого Ti',f"Задача {submission['job_id']}: полностью готово {len(points)}/14 точек.",
       'Новые средние: 90 пс, запись 10 фс; старые 30 пс с записью 100 фс хранятся отдельно.',
       'SEM не включает систематическое смещение давления, эффекты размера и модели.']
for p in sorted(points,key=lambda p:(p['functional'],p['T_K'])):
    lines.append(f"{p['functional']} {p['T_K']} K: {p['rho']['mean']:.7f} +/- {p['rho']['sem']:.7f} г/см3 (SEM); P={1000*p['qc']['P_GPa']['mean']:.3f} МПа; MSD10={p['qc']['MSD_10ps_A2']:.2f} A2; {p['qc']['warnings']}")
(OUT/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if len(points)!=14:
    print('Registry updated; wait for all 14 points before replacing article curves.');sys.exit(0)
assert all(p['qc']['MSD_10ps_A2']>=5 and p['qc']['solid_like_fraction_last10']<=.2 for p in points),'Review phase warnings before labeling these points liquid'
save(ROOT/'manuscript/supporting_data/xc_liquid_density_extended.json',result)
print('All 14 points validated. Figure input published; run update_ti_density.py after PDF skill marker.')
