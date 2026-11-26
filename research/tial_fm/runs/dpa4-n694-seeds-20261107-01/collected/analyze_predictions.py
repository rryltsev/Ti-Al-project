"""Unified evaluator: equal-structure energies/stress, atom/component-pooled forces."""
from pathlib import Path
from collections import defaultdict
import argparse,json
import numpy as np
GPA=160.21766208
def stats(x):
    x=np.asarray(x,dtype=float).ravel()
    return dict(n=len(x),MAE=float(np.mean(np.abs(x))),RMSE=float(np.sqrt(np.mean(x*x))),bias=float(np.mean(x)),abs_q95=float(np.quantile(np.abs(x),.95)),abs_q99=float(np.quantile(np.abs(x),.99)),abs_max=float(np.max(np.abs(x))))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--train-size',type=int,default=200);ap.add_argument('--data-root',type=Path);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    metadata={r['geometry_id']:r for r in map(json.loads,(a.package/'data_manifest.jsonl').read_text().splitlines())}
    result={};allrows=[]
    for xcdir in sorted(a.predictions.iterdir()):
        xc=xcdir.name;rows=[]
        for partdir in sorted(xcdir.iterdir()):
            part=partdir.name
            for predfile in sorted(partdir.glob('*.npz')):
                pred=np.load(predfile,allow_pickle=False);src=(a.data_root if a.data_root else a.package/'data')/xc/part/predfile.stem
                ids=json.loads((src/'geometry_ids.json').read_text());assert list(pred['ids'])==ids
                raw={k:np.load(src/'set.000'/(k+'.npy')) for k in ['energy','force','virial','box']}
                types=np.loadtxt(src/'type.raw',dtype=int,ndmin=1);names=(src/'type_map.raw').read_text().split();n=len(types)
                for i,sid in enumerate(ids):
                    volume=float(np.linalg.det(raw['box'][i].reshape(3,3)));fd=pred['force'][i].reshape(n,3)-raw['force'][i].reshape(n,3)
                    vd=pred['virial'][i].reshape(3,3)-raw['virial'][i].reshape(3,3)
                    stress=-(vd+vd.T)/2/volume*GPA;stress=stress.ravel()[[0,4,8,5,2,1]]
                    e=float((pred['energy'][i]-raw['energy'][i].ravel()[0])/n)
                    r=dict(id=sid,XC=xc,split=part,natoms=n,group=metadata[sid]['group_id'],family=metadata[sid]['structural_family'],Al_fraction=names.count('Al') and float(np.mean(types==names.index('Al'))),energy_error_eV_atom=e,force_error=fd,stress_error_GPa=stress,types=types,names=names)
                    assert np.isfinite(e) and np.isfinite(fd).all() and np.isfinite(stress).all();rows.append(r)
        train=[r for r in rows if r['split']=='train'];assert len(train)==a.train_size
        X=np.array([[r['Al_fraction'],1-r['Al_fraction']] for r in train]);y=-np.array([r['energy_error_eV_atom'] for r in train]);offset,_,rank,_=np.linalg.lstsq(X,y,rcond=None);assert rank==2
        alignment=dict(order=['Al','Ti'],offset_eV=offset.tolist(),rank=int(rank),condition_number=float(np.linalg.cond(X)),fit=f'Equal-structure least squares in energy/atom using N{a.train_size} train only; prediction E += nAl*bAl + nTi*bTi.')
        groups=defaultdict(list)
        for r in rows:
            r['energy_aligned_error_eV_atom']=r['energy_error_eV_atom']+np.dot([r['Al_fraction'],1-r['Al_fraction']],offset)
            for key in [r['split'],r['split']+'/family/'+r['family'],r['split']+'/Al_bin/'+str(min(4,int(5*r['Al_fraction'])))]:groups[key].append(r)
        metrics={}
        for key,rr in sorted(groups.items()):
            metrics[key]=dict(structures=len(rr),atoms=sum(r['natoms'] for r in rr),composition_groups=len({r['group'] for r in rr}),E_raw_meV_atom=stats([1000*r['energy_error_eV_atom'] for r in rr]),E_aligned_meV_atom=stats([1000*r['energy_aligned_error_eV_atom'] for r in rr]),F_component_eV_A=stats(np.concatenate([r['force_error'].ravel() for r in rr])),F_norm_error_eV_A=stats(np.concatenate([np.linalg.norm(r['force_error'],axis=1) for r in rr])),stress_component_GPa=stats(np.concatenate([r['stress_error_GPa'] for r in rr])),F_structure_equal_RMSE_eV_A=float(np.sqrt(np.mean([np.mean(r['force_error']**2) for r in rr]))))
            for species in ['Al','Ti']:
                errors=[r['force_error'][r['types']==r['names'].index(species)].ravel() for r in rr if species in r['names']]
                errors=np.concatenate(errors)
                if errors.size:metrics[key]['F_'+species+'_component_eV_A']=stats(errors)
        boot={}
        for part in ['validation','test']:
            rr=[r for r in rows if r['split']==part]
            if not rr:continue
            bygroup=defaultdict(list)
            for r in rr:bygroup[r['group']].append(r)
            sums=[]
            for gg in bygroup.values():sums.append([sum((r['energy_aligned_error_eV_atom']*1000)**2 for r in gg),len(gg),sum(np.sum(r['force_error']**2) for r in gg),sum(r['force_error'].size for r in gg),sum(np.sum(r['stress_error_GPa']**2) for r in gg),6*len(gg)])
            sums=np.array(sums);rng=np.random.default_rng(61106);draw=sums[rng.integers(len(sums),size=(2000,len(sums)))].sum(axis=1);values=np.sqrt(draw[:,[0,2,4]]/draw[:,[1,3,5]])
            boot[part]=dict(groups=len(sums),replicates=2000,method='Resample whole conservative fixed-composition groups, pooling all records in each selected group. Liquid exact parents are unavailable; this is not exact-trajectory bootstrap.',RMSE_95pct={k:np.quantile(values[:,i],[.025,.975]).tolist() for i,k in enumerate(['E_aligned_meV_atom','F_component_eV_A','stress_component_GPa'])})
        result[xc]=dict(alignment=alignment,metrics=metrics,bootstrap=boot,failed_predictions=0)
        for r in rows:
            allrows.append({k:v for k,v in r.items() if k not in ['force_error','stress_error_GPa','types','names']}|dict(force_sq_sum=float(np.sum(r['force_error']**2)),force_abs_sum=float(np.sum(np.abs(r['force_error']))),stress_sq_sum=float(np.sum(r['stress_error_GPa']**2))))
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'per_structure.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allrows))
    print(json.dumps({xc:{part:{k:d['metrics'][part][k]['RMSE'] for k in ['E_raw_meV_atom','E_aligned_meV_atom','F_component_eV_A','stress_component_GPa']} for part in ['train','validation','test'] if part in d['metrics']} for xc,d in result.items()},indent=2))
if __name__=='__main__':main()
