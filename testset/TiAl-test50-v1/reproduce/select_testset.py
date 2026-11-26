"""Freeze a balanced geometry-only holdout using periodic SOAP diversity."""
from audit_sources import ROOT, CACHE, Index, same, composition
from collections import Counter
import csv, hashlib, json, pickle, subprocess, sys
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dscribe.descriptors import SOAP

SOAP_CONFIG=dict(species=['Al','Ti'],periodic=True,r_cut=5.0,n_max=4,l_max=3,sigma=0.5,average='outer')
QUOTAS={'liquid':25,'POSCAR_alpha-Ti-hcp_orthogonal':8,'POSCAR_beta-Ti-bcc':9,'POSCAR_gamma_AlTi_L1_0':8}

def descriptor(atoms, name):
    path=CACHE/(name+'.npy')
    h=hashlib.sha256(json.dumps(SOAP_CONFIG,sort_keys=True).encode())
    for a in atoms:
        for x in (a.numbers,a.positions,a.cell.array):h.update(np.ascontiguousarray(x).tobytes())
    key=h.hexdigest();keypath=path.with_suffix('.sha256')
    if path.exists() and keypath.exists() and keypath.read_text()==key:return np.load(path)
    soap=SOAP(**SOAP_CONFIG)
    out=[]
    for i,a in enumerate(atoms):
        d=soap.create(a);out.append(d/np.linalg.norm(d))
        if i%100==0:print('SOAP',name,i,len(atoms),flush=True)
    out=np.array(out);np.save(path,out);keypath.write_text(key);return out

def aligned_rmsd(a,b):
    """Near-duplicate diagnostic: species assignment, PBC, affine cell removal.

    Local translation refinement starts at zero; not a general crystal matcher.
    Sufficient for the fixed-origin source trajectories; exact audit separately
    tests all anchor translations for equal cells.
    """
    if composition(a)!=composition(b):return None
    strain=float(np.max(np.abs(np.linalg.solve(a.cell.array,b.cell.array)-np.eye(3))))
    x=a.get_scaled_positions(wrap=True); y=b.get_scaled_positions(wrap=True)
    shift=np.zeros(3)
    for _ in range(3):
        ds=[]
        for z in (13,22):
            xx,yy=x[a.numbers==z],y[b.numbers==z]
            if not len(xx):continue
            d=xx[:,None,:]+shift-yy[None,:,:];d-=np.rint(d)
            r,c=linear_sum_assignment(np.sum((d@a.cell.array)**2,axis=2))
            ds.extend(d[r,c])
        ds=np.array(ds);shift-=ds.mean(axis=0)
    residual=(ds-ds.mean(axis=0))@a.cell.array
    return float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),strain

def main():
    data=pickle.load((CACHE/'reference.pkl').open('rb'));refs=data['refs']
    solids=pickle.load((CACHE/'solid_candidates.pkl').open('rb'))
    candidates=[dict(r,family='liquid',trajectory=None,frame=int(r['member'].rsplit('.',1)[1]),training_gap_ps=None,time_ps=None,temperature_K=None)
                for r in data['liquids'] if not r['matches']]+solids
    idx=Index(refs)
    clean=[]; rejected=[]
    for r in candidates:
        a=r['atoms']; distances=neighbor_list('d',a,3.0)
        mind=float(distances.min()) if len(distances) else 3.0
        if not np.isfinite(a.positions).all() or np.linalg.det(a.cell.array)<=0 or mind<1.8 or idx.matches(a):
            rejected.append({'family':r['family'],'frame':r['frame'],'trajectory':r['trajectory'],'min_distance_A':mind});continue
        r['min_distance_A']=mind;clean.append(r)
    candidates=clean
    print('QC passed',len(candidates),'rejected',len(rejected),flush=True)
    d=descriptor([r['atoms'] for r in candidates],'candidate_soap')
    refd=descriptor(refs,'reference_soap')
    distances=cdist(d,refd)
    selected=[];near_rejected=[]
    # Within each family, equal-width composition strata spanning its source
    # range; then maximize distance from previously selected SOAP vectors.
    # Seed is the pool medoid, independent of training predictions/labels.
    for family,quota in QUOTAS.items():
        ids=np.array([i for i,r in enumerate(candidates) if r['family']==family])
        xs=np.array([np.mean(candidates[i]['atoms'].numbers==13) for i in ids])
        bins=np.minimum(((xs-xs.min())/max(xs.max()-xs.min(),1e-12)*min(quota,5)).astype(int),min(quota,5)-1)
        counts=Counter();family_selected=[];used_trajectories=set();blocked=set()
        while len(family_selected)<quota:
            possible=[int(i) for i in ids if i not in blocked and i not in selected and
                      (family=='liquid' or candidates[i]['trajectory'] not in used_trajectories)]
            if not possible:raise RuntimeError('Insufficient unique trajectories for '+family)
            bin_of={int(i):int(b) for i,b in zip(ids,bins)}
            min_count=min(counts[bin_of[i]] for i in possible)
            possible=[i for i in possible if counts[bin_of[i]]==min_count]
            if family_selected:
                scores=cdist(d[possible],d[family_selected]).min(axis=1);chosen=possible[int(np.argmax(scores))]
            else:
                centroid=d[ids].mean(axis=0);chosen=min(possible,key=lambda i:float(np.linalg.norm(d[i]-centroid)))
            a=candidates[chosen]['atoms'];near=[];best=None
            for j,b in enumerate(refs):
                if composition(a)!=composition(b):continue
                result=aligned_rmsd(a,b)
                if best is None or result[0]<best[0]:best=(*result,j)
                if result[0]<0.1 and result[1]<0.01:near.append(j)
            for j in selected:
                if same(a,candidates[j]['atoms']):near.append('selected:'+str(j))
                result=aligned_rmsd(a,candidates[j]['atoms'])
                if result and result[0]<0.1 and result[1]<0.01:near.append('near_selected:'+str(j))
            if near:
                blocked.add(chosen);near_rejected.append({'candidate':chosen,'reference_ids':near});continue
            r=candidates[chosen]
            r['nearest_reference_soap_id']=int(np.argmin(distances[chosen]));r['nearest_reference_soap_distance']=float(distances[chosen].min())
            r['nearest_same_composition_rmsd_A']=None if best is None else best[0]
            r['nearest_same_composition_cell_strain']=None if best is None else best[1]
            r['nearest_same_composition_reference_id']=None if best is None else best[2]
            selected.append(chosen);family_selected.append(chosen);counts[bin_of[chosen]]+=1;used_trajectories.add(r['trajectory'])
            print('selected',family,len(family_selected),'xAl',np.mean(a.numbers==13),'rmsd',None if best is None else best[0],flush=True)
    out=ROOT/'TiAl-test50-v1';out.mkdir(exist_ok=False);(out/'POSCARS').mkdir()
    manifest=[];frames=[]
    for serial,i in enumerate(selected,1):
        r=candidates[i];a=r['atoms'].copy();a.calc=None;a.info={}
        sid=f'TiAl-test-{serial:03d}'
        a.info={'structure_id':sid,'family':r['family'],'reference_status':'unlabelled'}
        path=out/'POSCARS'/('POSCAR.'+sid)
        write(path,a,format='vasp',direct=True,sort=True,vasp5=True)
        assert same(a,read(path,format='vasp'))
        record={k:v for k,v in r.items() if k not in ('atoms','matches')}
        record.update(structure_id=sid,n_atoms=len(a),n_Al=int(sum(a.numbers==13)),n_Ti=int(sum(a.numbers==22)),x_Al=float(np.mean(a.numbers==13)),volume_per_atom_A3=a.get_volume()/len(a),poscar=path.relative_to(out).as_posix(),poscar_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_index=i)
        manifest.append(record);frames.append(a)
    write(out/'test50.extxyz',frames,format='extxyz',write_results=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'candidate_manifest.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('atoms','matches')} for r in candidates],indent=2))
    fields=sorted(set().union(*(r.keys() for r in manifest)))
    with (out/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(manifest)
    protocol={'dataset_id':'TiAl-test50-v1','seed':'deterministic family medoid then composition-balanced SOAP farthest point selection','quotas':QUOTAS,'soap':SOAP_CONFIG,
              'geometry_matching_tolerance_A':5e-5,'near_duplicate_rmsd_threshold_A':0.1,'near_duplicate_cell_strain_threshold':0.01,
              'solid_min_training_gap_ps':0.5,'solid_one_frame_per_trajectory':True,'solid_discard_before_ps':1.0,
              'minimum_pair_distance_A':1.8,'qc_rejected':rejected,'near_rejected':near_rejected,
              'eligible_candidates':len(candidates),'n_reference_geometries':len(refs),'no_dft_labels':True,
              'limitations':['Holdout from supplied fine-tuning datasets only; foundation pretraining overlap unknown.',
              'Solid families describe parent lattices, not a verified finite-temperature phase classification. D019 trajectories are absent: all 11 source jobs failed before MD (NPT triangular-cell requirement).',
              '0.5 ps temporal separation is a heuristic, not a measured decorrelation time; source trajectories can be shared with training.',
              'Liquid archive lacks complete original trajectory/time metadata; trajectory-level independence cannot be certified. Liquid is the source-pool label, not a verified phase assignment.',
              'Diversity-selected benchmark, not an unbiased random sample of thermodynamic configurations.']}
    (out/'selection_protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'reference_origins.json').write_text(json.dumps(data['origins'],indent=2))
    np.savez_compressed(out/'selection_descriptors.npz',candidate_soap=d,reference_soap=refd,selected_indices=selected)
    (out/'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for color,fam in zip(['#167ca6','#a64b00','#13845c','#8a4a9f'],QUOTAS):
        pool=[r for r in candidates if r['family']==fam];sel=[r for r in manifest if r['family']==fam]
        axes[0].scatter([np.mean(r['atoms'].numbers==13) for r in pool],[r['atoms'].get_volume()/len(r['atoms']) for r in pool],s=8,alpha=.12,color=color)
        axes[0].scatter([r['x_Al'] for r in sel],[r['volume_per_atom_A3'] for r in sel],s=40,color=color,label=fam.replace('POSCAR_',''))
    axes[0].set(xlabel='Al atomic fraction',ylabel='Volume / atom (Å³)');axes[0].legend(fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2)
    bins=np.linspace(0,distances.min(axis=1).max()*1.01,36)
    axes[1].hist(distances.min(axis=1),bins=bins,alpha=.45,label='Eligible pool')
    axes[1].hist(distances[selected].min(axis=1),bins=bins,alpha=.8,label='Test50')
    axes[1].set(xlabel='Distance to nearest reference (normalized SOAP)',ylabel='Configurations');axes[1].set_yscale('log');axes[1].legend()
    fig.savefig(out/'coverage.png',dpi=180);fig.savefig(out/'coverage.pdf')
    print('DONE',out,Counter(r['family'] for r in manifest),flush=True)

if __name__=='__main__':main()
