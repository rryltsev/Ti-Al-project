"""Lossless-precision extxyz inputs for GRACE from the frozen N200/validation split."""
from pathlib import Path
import hashlib,json
import numpy as np
from ase.io import read

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/grace-2l-preparation-20261107-01'
SOURCE=ROOT/'runs/dpa4-replicates-full-20261106-02/package'

def main(run=RUN,ntrain=200,xcs=('PBE','r2SCAN')):
    out=run/'data';out.mkdir(exist_ok=False)
    splits=json.loads((SOURCE/'splits.json').read_text());subsets=json.loads((SOURCE/'subsets.json').read_text())
    meta={r['geometry_id']:r for r in map(json.loads,(SOURCE/'data_manifest.jsonl').read_text().splitlines())}
    result={};fmt=lambda x:format(float(x),'.17g')
    for xc in xcs:
        result[xc]={};(out/xc).mkdir()
        for part in ['train','validation']:
            records=[];hashes={};max_anti=0.
            for system in sorted((SOURCE/'data'/f'N{ntrain}'/xc/part).iterdir()):
                names=(system/'type_map.raw').read_text().split();types=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
                syms=[names[t] for t in types];n=len(syms);ids=json.loads((system/'geometry_ids.json').read_text())
                data={k:np.load(system/'set.000'/(k+'.npy')) for k in ['coord','box','energy','force','virial']}
                for p in system.rglob('*'):
                    if p.is_file():hashes[p.relative_to(SOURCE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
                for i,sid in enumerate(ids):
                    pos=data['coord'][i].reshape(n,3);cell=data['box'][i].reshape(3,3);forces=data['force'][i].reshape(n,3)
                    virial=data['virial'][i].reshape(3,3);volume=np.linalg.det(cell);assert volume>0
                    stress=-(virial+virial.T)/(2*volume);max_anti=max(max_anti,float(np.max(abs(virial-virial.T))))
                    records.append(dict(id=sid,symbols=syms,pos=pos,cell=cell,energy=float(data['energy'][i].ravel()[0]),forces=forces,stress=stress))
            expected=set(splits['split_ids']['train'] if ntrain==694 else subsets['subsets'][str(ntrain)]) if part=='train' else set(splits['split_ids']['validation'])
            assert {r['id'] for r in records}==expected and len(records)==len(expected)
            assert not expected&set(splits['split_ids']['test'])
            path=out/xc/(part+'.extxyz')
            with path.open('w',encoding='utf8',newline='\n') as f:
                for r in records:
                    f.write(str(len(r['symbols']))+'\n')
                    f.write('Lattice="'+' '.join(map(fmt,r['cell'].ravel()))+'" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T T" energy='+fmt(r['energy'])+' free_energy='+fmt(r['energy'])+' stress="'+' '.join(map(fmt,r['stress'].ravel(order='F')))+'" geometry_id='+r['id']+' split='+part+' XC='+xc+'\n')
                    for s,p,force in zip(r['symbols'],r['pos'],r['forces']):f.write(s+' '+' '.join(map(fmt,np.r_[p,force]))+'\n')
            recovered=read(path,index=':',format='extxyz');assert len(recovered)==len(records)
            errors={k:0. for k in ['positions_A','cell_A','energy_eV','forces_eV_A','stress_eV_A3']}
            for at,r in zip(recovered,records):
                assert at.info['geometry_id']==r['id'] and at.get_chemical_symbols()==r['symbols']
                for key,a,b in [('positions_A',at.positions,r['pos']),('cell_A',at.cell.array,r['cell']),('energy_eV',at.get_potential_energy(),r['energy']),('forces_eV_A',at.get_forces(),r['forces']),('stress_eV_A3',at.get_stress(voigt=False),r['stress'])]:
                    err=float(np.max(abs(np.asarray(a)-b)));errors[key]=max(errors[key],err);assert err<1e-12,(path,key,err)
            result[xc][part]=dict(structures=len(records),atoms=sum(len(r['symbols']) for r in records),groups=len({meta[r['id']]['group_id'] for r in records}),ids=[r['id'] for r in records],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),roundtrip_max_abs=errors,source_files=hashes,max_virial_antisymmetry_eV=max_anti)
    groups=[{meta[i]['group_id'] for i in (splits['split_ids']['train'] if ntrain==694 else subsets['subsets'][str(ntrain)])},{meta[i]['group_id'] for i in splits['split_ids']['validation']}]
    assert not groups[0]&groups[1]
    report=dict(status='PASS',data=result,test_access=False,group_overlap=0,
        convention='Total E/free_energy in eV, forces eV/Angstrom; symmetric ASE tensile stress = -(virial+virial.T)/(2*volume), eV/Angstrom^3. Same symmetrization as common DPA4 evaluator. Native XC cells and atom order retained; 17-digit text roundtrip checked.',
        backend_validation_mapping='GRACE test_filename will point ONLY to validation.extxyz (149). It is the development validation role, never the held-out test149.',
        split_sha256=hashlib.sha256((SOURCE/'splits.json').read_bytes()).hexdigest())
    (run/'data-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',counts={xc:{p:d['structures'] for p,d in rows.items()} for xc,rows in result.items()},max_roundtrip={xc:{p:d['roundtrip_max_abs'] for p,d in rows.items()} for xc,rows in result.items()})))

if __name__=='__main__':main()
