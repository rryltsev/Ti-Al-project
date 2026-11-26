"""Read-only audit of available raw VASP fields for the N200/development labels."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,re,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/label-fields-20261106-02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array(node):return np.array([[float(x) for x in v.text.split()] for v in node.findall('v')])
def match(c,b,s,tc,tb,ts):
    assert np.max(np.abs(b-tb))<5e-5
    frac=c@np.linalg.inv(tb);target=tc@np.linalg.inv(tb)
    def trial(shift):
        perm=np.zeros(len(c),dtype=int);worst=0.
        for species in set(s):
            ii=np.where(s==species)[0];jj=np.where(ts==species)[0]
            if len(ii)!=len(jj):return None
            d=frac[ii,None,:]+shift-target[None,jj,:];d-=np.rint(d);dist=np.linalg.norm(d@tb,axis=2);best=dist.argmin(axis=1)
            if len(set(best))!=len(ii):return None
            worst=max(worst,float(np.max(dist[np.arange(len(ii)),best])));perm[ii]=jj[best]
        return (perm,worst) if worst<5e-5 else None
    found=trial(np.zeros(3))
    if found is not None:return found
    species=min(set(s),key=lambda x:np.count_nonzero(s==x));i=np.where(s==species)[0][0]
    for j in np.where(ts==species)[0]:
        found=trial(target[j]-frac[i])
        if found is not None:return found
    raise ValueError('Raw/label geometry mismatch')
def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--train-size',type=int,default=200,choices=[200,694]);ap.add_argument('--functional',choices=['both','r2SCAN'],default='both');ap.add_argument('--output',type=Path,default=OUT);a=ap.parse_args()
    OUT=a.output;OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(exist_ok=True);assert not (OUT/'summary.json').exists()
    subset=json.loads((ROOT/'datasets/nested-train-v1-20261106/subsets.json').read_text());training_ids=subset['subsets'][str(a.train_size)];chosen=set(training_ids+subset['validation_ids'])
    records=[json.loads(x) for x in (ROOT/'manifests/data_manifest.jsonl').read_text().splitlines()];results=[];missing=[]
    for r in records:
        if r['geometry_id'] not in chosen:continue
        for xc,folder in [('PBEsol','VASP_PBEsol'),('r2SCAN','r2SCAN')]:
            if a.functional!='both' and xc!=a.functional:continue
            label=r['labels'][xc];candidates=[Path(p).parent for p in label['primary_result_candidates_all_XC'] if p.endswith('POSCAR') and ('\\VASP-output\\'+folder+'\\RESULTS\\') in p]
            if not candidates:missing.append(dict(id=r['geometry_id'],XC=xc));continue
            assert len(candidates)==1
            src=candidates[0];xml=src/'vasprun.xml';tree=ET.parse(xml).getroot();calcs=tree.findall('calculation');assert len(calcs)==1
            calc=calcs[0];energy={v.attrib['name']:float(v.text) for v in calc.findall('energy/i')}
            frame=label['frame_index'];actual=float(np.load(label['energy_path'])[frame].ravel()[0]);delta={k:actual-v for k,v in energy.items()}
            matching=[k for k,v in delta.items() if abs(v)<1e-6];assert matching,delta
            st=tree.find("structure[@name='finalpos']");cell=array(st.find("crystal/varray[@name='basis']"));coord=array(st.find("varray[@name='positions']"))@cell
            symbols=np.array([rc.findall('c')[0].text.strip() for rc in tree.findall("atominfo/array[@name='atoms']/set/rc")])
            path=Path(label['coord_path']);target=np.load(path)[frame].reshape(-1,3);box=np.load(path.with_name('box.npy'))[frame].reshape(3,3)
            names=(path.parent.parent/'type_map.raw').read_text().split();types=np.loadtxt(path.parent.parent/'type.raw',dtype=int,ndmin=1);ts=np.array([names[t] for t in types])
            perm,geo=match(coord,cell,symbols,target,box,ts)
            forces=array(calc.find("varray[@name='forces']"));targetf=np.load(label['force_path'])[frame].reshape(-1,3)
            ferr=float(np.max(np.abs(forces-targetf[perm])));assert ferr<1e-6,(src,ferr)
            stress=array(calc.find("varray[@name='stress']"));volume=float(np.linalg.det(cell));virial=stress*volume/(160.21766208*10)
            verr=float(np.max(np.abs(virial-np.load(label['virial_path'])[frame].reshape(3,3))))
            params={v.attrib.get('name'):(v.text or '').strip() for v in tree.findall('parameters//i')}
            incar={v.attrib.get('name'):(v.text or '').strip() for v in tree.findall('incar/i')}
            for key in ['ENCUT','METAGGA']:params[key]=incar.get(key,params.get(key))
            if not params.get('ENCUT'):params['ENCUT']=params.get('ENMAX')
            text=(src/'OUTCAR').read_text(errors='replace');normal='General timing and accounting' in text;ediff='aborting loop because EDIFF is reached' in text
            pot=(src/'POTCAR').read_text(errors='replace');titles=re.findall(r'TITEL\s*=\s*(.+)',pot)
            results.append(dict(id=r['geometry_id'],XC=xc,role=f'train{a.train_size}' if r['geometry_id'] in training_ids else 'validation149',source=str(src),xml_sha256=sha(xml),OUTCAR_sha256=sha(src/'OUTCAR'),POTCAR_sha256=sha(src/'POTCAR'),POTCAR_titles=titles,energy_eV=actual,energy_field_matches=matching,energy_minus_raw_fields_eV=delta,force_max_difference_eV_A=ferr,virial_max_difference_eV=verr,geometry_max_difference_A=geo,normal_termination=normal,EDIFF_marker=ediff,SCF_iterations=len(calc.findall('scstep')),settings={k:params.get(k) for k in ['ENCUT','EDIFF','ISMEAR','SIGMA','NELM','LREAL','GGA','METAGGA','LASPH']},VASP_version=tree.find("generator/i[@name='version']").text.strip()))
    (OUT/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    summary=dict(scope=f'Existing N{a.train_size} train and149 development raw archives only. No DFT launched; immutable NPY labels unchanged.',PBE='Raw archive not located; PBE energy field remains unresolved.',missing=missing,results={})
    for xc in ['PBEsol','r2SCAN']:
        rr=[r for r in results if r['XC']==xc]
        if not rr:continue
        summary['results'][xc]=dict(records=len(rr),energy_matches=dict(Counter(k for r in rr for k in r['energy_field_matches'])),force_max_difference_eV_A=max(r['force_max_difference_eV_A'] for r in rr),virial_max_difference_eV=max(r['virial_max_difference_eV'] for r in rr),non_normal=[r['id'] for r in rr if not r['normal_termination']],missing_EDIFF_marker=[r['id'] for r in rr if not r['EDIFF_marker']],SCF_reached_NELM=[r['id'] for r in rr if r['SCF_iterations']>=int(r['settings']['NELM'])],settings_variants=dict(Counter(json.dumps(r['settings'],sort_keys=True) for r in rr)),POTCAR_titles=dict(Counter(tuple(r['POTCAR_titles']) for r in rr)) if False else sorted({t for r in rr for t in r['POTCAR_titles']}))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
