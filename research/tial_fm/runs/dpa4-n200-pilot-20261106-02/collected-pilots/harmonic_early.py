"""Evaluate the independent nine-structure H0 while bulk B0 predictions continue."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parent;EARLY=ROOT/'harmonic-early'
def forces():
    import numpy as np
    from ase.io import read
    from deepmd.infer import DeepPot
    from baseline_worker import evaluate,sha,dump
    cfg=json.loads((ROOT/'protocol.json').read_text());assert sha(cfg['model']['path'])==cfg['model']['sha256']
    EARLY.mkdir(exist_ok=False)
    for name in ['harmonic_inputs','reference_figure.npz']:(EARLY/name).symlink_to(ROOT/name,target_is_directory=(ROOT/name).is_dir())
    dp=DeepPot(cfg['model']['path']);records={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        records[xc]={}
        for key in ['force000','force001','force002']:
            p=ROOT/'harmonic_inputs'/xc/key/'POSCAR';a=read(p,format='vasp');e,f,v=evaluate(dp,a.positions,a.cell.array,a.get_chemical_symbols())
            dst=EARLY/'baseline/harmonic_forces'/xc/key;dst.mkdir(parents=True);np.save(dst/'forces.npy',f)
            records[xc][key]=dict(POSCAR_sha256=sha(p),energy_eV=e,virial_eV=v.tolist())
    dump(EARLY/'predictions.json',dict(model=cfg['model'],job_id=os.environ['SLURM_JOB_ID'],structures=records))
def phonons():
    import harmonic_worker as h
    h.ROOT=EARLY;h.main()
    assert not (ROOT/'harmonics').exists()
    (EARLY/'harmonics').rename(ROOT/'harmonics')
    from harmonic_mode_diagnostics import analyze
    analyze(ROOT/'harmonics')
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    {'forces':forces,'phonons':phonons}[sys.argv[1]]()
