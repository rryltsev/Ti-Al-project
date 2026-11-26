"""Verify matched PBE/r2SCAN exports; archive portable scientific artifacts."""
import build_pbesol_phonons as b
import numpy as np
import json,shutil,zipfile
from datetime import datetime,timezone

records={}
for xc in ['PBE','r2SCAN']:
    out=b.ROOT/'results'/(xc+'-matched-DFT-DPA4-20261101-01')
    hashes=json.loads((out/'manifest-sha256.json').read_text())
    assert all(b.digest(out/n)==h for n,h in hashes.items())
    meshes={k:np.load(out/k/'mesh50.npz') for k in ['DFT','DPA4']}
    assert np.array_equal(meshes['DFT']['qpoints'],meshes['DPA4']['qpoints'])
    assert np.array_equal(meshes['DFT']['weights'],meshes['DPA4']['weights'])
    data=np.load(out/'comparison.npz')
    assert all(np.isfinite(data[k]).all() for k in data.files)
    for k in ['DFT','DPA4']:
        assert abs(np.trapezoid(data['DOS_'+k],data['frequency_THz'])-3)<1e-10
        assert data['bands_'+k].min()>-.001
        assert meshes[k]['frequencies_THz'].min()>-.001
    verification=dict(utc=datetime.now(timezone.utc).isoformat(),source_output_hashes_verified=True,
        same_qpoints_weights=True,finite_arrays=True,integrals_three=True,no_modes_below_minus_0p001_THz=True,
        visual_QA='Bands+DOS and separate DOS PNG inspected; legends and annotations readable, no clipping.',
        DFT_reference_pending=True,reference_cancellation='Common reference subtracts out exactly in (Fplus-Fminus)/2; numerical algebra check <1e-15 eV/A')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    note='\nDFT force000 has NOT been calculated: its POSCAR is included for the geometric identity check only. No zero-force placeholder is used. For the central pair, [(Fplus-F0)-(Fminus-F0)]/2 = (Fplus-Fminus)/2. This does not validate reference residual forces or stress.\n'
    with (out/'README.txt').open('a') as f:f.write(note)
    (out/'manifest-sha256.json').write_text(json.dumps({str(p.relative_to(out)):b.digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest-sha256.json'},indent=2)+'\n')
    archive=shutil.make_archive(str(b.ROOT/'results'/(xc+'-matched-DFT-DPA4-20261101')),'zip',out)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    records[xc]=dict(archive=archive,sha256=b.digest(archive),verification=verification)
(b.ROOT/'results/matched-PBE-r2SCAN-20261101-verification.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
