"""Freeze training-only energy-reference calibration inputs, with holdout audit."""
from pathlib import Path
import sys
import hashlib
import json
import time
import numpy as np
from ase import Atoms
from ase.io import read

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from audit_sources import Index
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\DPA4-finetune')
OUT=ROOT/'alignment/train-offsets-v1'
XC={'PBE':'TiAl_PBE_upd','PBEsol':'TiAl_PBEsol_upd','r2SCAN':'TiAl_scan_upd'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    registry=json.loads((ROOT/'pretrained_registry.json').read_text())
    protocol={'status':'predeclared_before_training_predictions','created_unix':time.time(),
              'authorization':'User requested training-only energy alignment on 2026-10-22 after inspecting raw test results.',
              'model_sha256':registry['sha256'],'formula':'E_aligned = E_pretrained + n_Al*c_Al + n_Ti*c_Ti',
              'elements':['Al','Ti'],'fit':'Unregularized rank-2 least squares of per-atom energy residuals; equal weight per training frame.',
              'target':'(energy.npy - pretrained total energy)/n_atoms',
              'selection':'All frames in training_systems.json, separately for each functional; no validation/test labels, no outlier removal.',
              'constraints':'Fit once then freeze. Forces, stresses and pretrained weights unchanged. Retain raw metrics. Do not change stopping heuristic.',
              'test_geometry_manifest_sha256':sha(ROOT.parent/'TiAl-test50-v1/manifest.json')}
    save(OUT/'protocol.json',protocol)
    holdout_meta=json.loads((ROOT.parent/'TiAl-test50-v1/manifest.json').read_text())
    holdout=[read(ROOT.parent/'TiAl-test50-v1'/r['poscar'],format='vasp') for r in holdout_meta]
    holdout_index=Index(holdout)
    arrays={};geometries={};rows=[];files={};summaries={}
    def record_file(p):files[str(p)]=sha(p)
    for xc,name in XC.items():
        base=SOURCE/name/'npy_format';listing=base/'training_systems.json';record_file(listing)
        systems=json.loads(listing.read_text());relative=[s.split('/npy_format/',1)[1] for s in systems]
        assert len(relative)==len(set(relative)) and all(Path(s).parts[0]=='train' and '..' not in Path(s).parts for s in relative)
        assert set(relative)=={p.relative_to(base).as_posix() for p in (base/'train').iterdir() if p.is_dir()}
        if xc=='PBE':
            config=SOURCE/'TiAl_tune_pbe_DPA4-shard-Neo-OMat24-v20260805.json';record_file(config)
            assert json.loads(config.read_text())['training']['training_data']['systems']==systems
        xc_atoms=[];xc_rows=[]
        for rel in sorted(relative):
            system=base/rel
            for p in [system/'type.raw',system/'type_map.raw']:record_file(p)
            types=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
            names=(system/'type_map.raw').read_text().split();symbols=[names[t] for t in types]
            assert set(symbols)<=set(['Al','Ti']);n=len(symbols)
            for folder in sorted(system.glob('set.*')):
                for part in ['coord','box','energy']:record_file(folder/(part+'.npy'))
                coords=np.load(folder/'coord.npy',allow_pickle=False).reshape(-1,n,3)
                cells=np.load(folder/'box.npy',allow_pickle=False).reshape(-1,3,3)
                energies=np.load(folder/'energy.npy',allow_pickle=False).reshape(-1)
                assert len(coords)==len(cells)==len(energies)
                assert all(np.isfinite(a).all() for a in [coords,cells,energies])
                for frame,(pos,cell,energy) in enumerate(zip(coords,cells,energies)):
                    atoms=Atoms(symbols,positions=pos,cell=cell,pbc=True)
                    assert not holdout_index.matches(atoms),'Training/Test50 geometry overlap'
                    xc_atoms.append(atoms)
                    # Exact deduplication only: no coordinate rounding or geometry substitution.
                    ids=np.array([0 if s=='Al' else 1 for s in symbols],dtype='<i4')
                    order=np.lexsort((pos[:,2],pos[:,1],pos[:,0],ids))
                    ordered=np.asarray(pos[order],dtype='<f8');box=np.asarray(cell,dtype='<f8');ids=ids[order]
                    key=hashlib.sha256(ids.tobytes()+box.tobytes()+ordered.tobytes()).hexdigest()
                    if key not in geometries:
                        i=len(geometries);geometries[key]=i
                        arrays[f'coords_{i}']=ordered;arrays[f'cell_{i}']=box;arrays[f'types_{i}']=ids
                    row={'functional':xc,'split':'train','system':rel,'set':folder.name,'frame':frame,
                         'geometry_id':geometries[key],'geometry_sha256':key,'n_atoms':n,
                         'n_Al':symbols.count('Al'),'n_Ti':symbols.count('Ti'),'energy_eV':float(energy),
                         'energy_file':str(folder/'energy.npy')}
                    rows.append(row);xc_rows.append(row)
        assert len({r['geometry_sha256'] for r in xc_rows})==len(xc_rows),'Duplicate training frames'
        train_index=Index(xc_atoms);v_count=0
        vl=base/'validation_systems.json';record_file(vl)
        for s in json.loads(vl.read_text()):
            rel=s.split('/npy_format/',1)[1];assert Path(rel).parts[0]=='validate'
            system=base/rel;types=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
            names=(system/'type_map.raw').read_text().split();symbols=[names[t] for t in types]
            for folder in sorted(system.glob('set.*')):
                # Only geometry is read for split auditing; validation energies are never loaded.
                for p in [system/'type.raw',system/'type_map.raw',folder/'coord.npy',folder/'box.npy']:record_file(p)
                coords=np.load(folder/'coord.npy',allow_pickle=False).reshape(-1,len(symbols),3)
                cells=np.load(folder/'box.npy',allow_pickle=False).reshape(-1,3,3)
                for pos,cell in zip(coords,cells):
                    assert not train_index.matches(Atoms(symbols,positions=pos,cell=cell,pbc=True)), 'Train/validation overlap'
                    v_count+=1
        x=np.array([[r['n_Al']/r['n_atoms'],r['n_Ti']/r['n_atoms']] for r in xc_rows])
        assert np.linalg.matrix_rank(x)==2
        summaries[xc]={'training_frames':len(xc_rows),'validation_frames_excluded':v_count,'training_systems':len(relative),
                       'x_Al_range':[float(x[:,0].min()),float(x[:,0].max())],
                       'energy_eV_atom_range':[min(r['energy_eV']/r['n_atoms'] for r in xc_rows),max(r['energy_eV']/r['n_atoms'] for r in xc_rows)],
                       'train_validation_overlap':0,'train_Test50_overlap':0,'design_rank':2}
    np.savez_compressed(OUT/'training-geometries.npz',**arrays)
    save(OUT/'training-labels.json',rows)
    audit={'status':'passed','datasets':summaries,'unique_training_geometries':len(geometries),
           'training_rows':len(rows),'source_files_sha256':files,'protocol_sha256':sha(OUT/'protocol.json'),
           'geometries_sha256':sha(OUT/'training-geometries.npz'),'labels_sha256':sha(OUT/'training-labels.json'),
           'geometry_audit':'Same cell axes, periodic images, atom permutations and common translations, tolerance 5e-5 A.'}
    save(OUT/'input-audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='source_files_sha256'},indent=2))


if __name__=='__main__':main()
