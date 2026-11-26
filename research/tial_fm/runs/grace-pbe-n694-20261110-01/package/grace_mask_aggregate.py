"""CPU postprocessing: verify and compare four new masks with four parent pilots."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np

ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def metrics(rows,reported,ntrain=200):
    assert len(rows)==ntrain+149 and len({r['id'] for r in rows})==ntrain+149
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation'];assert len(train)==ntrain and len(val)==149
    x=np.array([r['fractions'] for r in train]);y=np.array([(r['reference_energy']-r['energy'])/r['natoms'] for r in train]);b=np.linalg.lstsq(x,y,rcond=None)[0]
    assert np.linalg.matrix_rank(x)==2
    assert np.allclose(b,[reported['offset_eV_atom'][k] for k in ['Al','Ti']],rtol=1e-10,atol=1e-12)
    result={}
    for part,rr in [('train',train),('validation',val)]:
        e=np.array([(r['energy']-r['reference_energy'])/r['natoms'] for r in rr]);aligned=e+np.array([r['fractions'] for r in rr])@b
        d=dict(E_raw_RMSE_meV_atom=float(np.sqrt(np.mean(e**2))*1000),E_aligned_RMSE_meV_atom=float(np.sqrt(np.mean(aligned**2))*1000),F_RMSE_eV_A=float(np.sqrt(sum(r['force_sq'] for r in rr)/sum(3*r['natoms'] for r in rr))),stress_RMSE_GPa=float(np.sqrt(np.mean([r['stress_sq']/6 for r in rr]))))
        for k,v in d.items():assert np.isfinite(v) and np.isclose(v,reported['metrics'][part][k],rtol=1e-11,atol=1e-12)
        result[part]=d
    return result

def case_result(root,c,ref):
    w=root/'cases'/c['name'];t=load(w/'training-summary.json');assert t['status']=='PASS' and t['epochs']==c['epochs'] and t['steps']==c['steps'] and not t['test_access']
    for name,h in t['hashes'].items():assert sha(w/'final-checkpoint'/name)==h
    for name in ['evaluation-summary.json','numerical-QC.json','zero-step.json','harmonics/summary.json','checkpoint-path-check.json']:assert load(w/name)['status']=='PASS'
    assert load(w/'validation-selection.json')['best_checkpoint_reload_skipped']
    inv=load(w/'parameter-inventory.json');assert inv['trainable']==c.get('expected_trainable',21111690 if c['method']=='restricted' else 21764956)
    checks=[load(f) for f in (w/'checks').glob('*.json')];assert {v['steps'] for v in checks}=={1,c['steps'],*range(1000,c['steps']+1,1000)}
    assert all(v['status']=='PASS' and not v['frozen_changed'] and v['steps']==v['optimizer_iterations'] for v in checks)
    rows=load(w/'per-structure.json');m=metrics(rows,load(w/'metrics.json'),c['N']);xc=c['XC'];h=load(w/'harmonics/summary.json')['results'][xc]
    bands=np.load(w/'harmonics'/xc/'bands.npz');assert np.array_equal(bands['qpoints'],ref['qpoints'])
    rmse=float(np.sqrt(np.mean((bands['frequencies']-ref[xc+'_DFT_bands'])**2)));assert np.isclose(rmse,h['sorted_band_RMSE_to_DFT_THz'],rtol=1e-12)
    mesh=np.load(w/'harmonics'/xc/'mesh50.npz');assert mesh['weights'].sum()==125000
    imaginary=int(np.sum(mesh['weights'][:,None]*(mesh['frequencies']<-.001)));assert imaginary==h['model']['imaginary_modes_weighted']
    dos=np.load(w/'harmonics'/xc/'dos.npz');assert abs(np.trapezoid(dos['dos'],dos['frequency'])-3)<1e-5
    gamma=h['model']['gamma_THz'];check=next(v for v in checks if v['steps']==c['steps'])
    return dict(case=c,metrics=m,phonons=dict(band_RMSE_THz=rmse,gamma_doublet_THz=float(np.mean(gamma[3:5])),gamma_singlet_THz=gamma[5],gamma_singlet_abs_error_THz=float(abs(gamma[5]-ref[xc+'_DFT_bands'][0,5])),imaginary_modes_weighted=imaginary,mesh_min_THz=h['model']['mesh_min_THz']),trainable=inv['trainable'],training_seconds=t['training_seconds'],GPU_peak_GiB=t['GPU_memory']['peak']/2**30,checkpoint_hashes=t['hashes'],first5_batch_hashes=check['first5_batch_hashes'],ids={p:sorted(r['id'] for r in rows if r['split']==p) for p in ['train','validation']},parameter_checks=len(checks))

def main():
    out=ROOT/'comparison';out.mkdir(exist_ok=False);plan=load(ROOT/'plan.json');parent=Path(plan['parent_run']);ref=np.load(ROOT/'reference_figure.npz');cases={};failures={}
    for root in [parent,ROOT]:
        for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json','CHECKPOINT_FILES_SHA256.json','ANALYSIS_FILES_SHA256.json']:
            if (root/manifest).exists():
                for name,h in load(root/manifest).items():assert sha(root/name)==h,(str(root),name)
        for c in load(root/'plan.json')['cases']:
            try:cases[c['name']]=case_result(root,c,ref)
            except Exception as exc:failures[c['name']]=repr(exc)
    if failures:
        dump(out/'summary.json',dict(status='FAIL',failures=failures,completed=cases));raise RuntimeError(failures)
    assert len(cases)==8
    membership=next(iter(cases.values()))['ids'];assert all(c['ids']==membership for c in cases.values())
    comparisons={}
    for xc in ['PBE','r2SCAN']:
        cc=[c for c in cases.values() if c['case']['XC']==xc];assert all(c['first5_batch_hashes']==plan['reference_first5_batch_hashes'][xc] for c in cc)
        base=cases[f'{xc}-restricted-N200-seed10'];comparisons[xc]={}
        for method in ['full','restricted_mlp','restricted_radial']:
            c=cases[f'{xc}-{method}-N200-seed10'];v=c['metrics']['validation'];b=base['metrics']['validation']
            comparisons[xc][method]=dict(validation_reduction_vs_restricted_percent={k:100*(1-v[k]/b[k]) for k in v},phonon_reduction_vs_restricted_percent=100*(1-c['phonons']['band_RMSE_THz']/base['phonons']['band_RMSE_THz']))
    result=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,comparisons=comparisons,checks=dict(models=8,parameter_checks=sum(c['parameter_checks'] for c in cases.values()),identical_train200_validation149=True,identical_first5_batches_within_XC=True,checkpoint_hashes_verified=True,seed=10,seed_count=1,test_access=False))
    dump(out/'summary.json',result)
    labels={'full':'Full','restricted':'Restricted','restricted_mlp':'Restricted + MLP','restricted_radial':'Restricted + radial'}
    lines=['# GRACE N200: controlled mask comparison','',result['utc'],'','All eight models: fixed final epoch50 / step10000, fp64, seed10. Energy alignment is fitted on train200 only. No held-out test access.','',
        '| XC | Mask | E raw meV/atom | E aligned meV/atom | F eV/A | Stress GPa | Band RMSE THz | Gamma singlet error THz |','|---|---|---:|---:|---:|---:|---:|---:|']
    for xc in ['PBE','r2SCAN']:
        for method in labels:
            c=cases[f'{xc}-{method}-N200-seed10'];v=c['metrics']['validation'];h=c['phonons']
            lines.append(f'| {xc} | {labels[method]} | {v["E_raw_RMSE_meV_atom"]:.4f} | {v["E_aligned_RMSE_meV_atom"]:.4f} | {v["F_RMSE_eV_A"]:.6f} | {v["stress_RMSE_GPa"]:.6f} | {h["band_RMSE_THz"]:.6f} | {h["gamma_singlet_abs_error_THz"]:.6f} |')
    lines+=['','One seed per mask: these differences do not quantify seed variability. Matched DFT geometry; central +/-0.01 A displacements, same ASR/symfc, mesh50^3. No inference about DFT convergence or long-time MD stability.','',json.dumps(comparisons,indent=2)]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('COMPARISON_PASS',len(cases))

if __name__=='__main__':main()
