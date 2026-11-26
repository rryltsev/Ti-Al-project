"""Independent stdlib audit of explicitly identified converged alpha-Ti force calculations."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET
from cluster import ROOT, RECORD
from audit_checkpoint_densities import header


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def vectors(node):
    return [[float(x) for x in v.text.split()] for v in node.findall('v')]


def difference(a, b):
    assert len(a) == len(b)
    return max(abs(x-y) for ra, rb in zip(a,b) for x,y in zip(ra,rb))


def main(step='force-001', xc='PBEsol'):
    if step not in ['force-000','force-001','force-002']:raise ValueError(step)
    reference=step=='force-000'
    second=step=='force-002'
    job='20962927' if second else '20959338'
    revision='production-04' if second else 'production-03'
    expected_steps=27 if second else 25
    day='20261029' if second else '20261028'
    folder_step=step
    end_local='2026-10-28T23:17:44+05:00' if second else '2026-10-28T08:41:39+05:00'
    elapsed='13:40:49' if second else '08:31:00'
    if xc=='PBE' and reference:
        job,revision,folder_step,expected_steps,day='20973061','production-06','force-000',28,'20261102'
        end_local,elapsed='2026-11-02T13:50:11+05:00','10:02:58'
        candidates=sorted(RECORD.glob('PBE-force-000-production-06-collection-*.json'))
        assert candidates
        collection=json.loads(candidates[-1].read_text())
        assert job+'|COMPLETED|' in collection['scheduler'] and '|0:0' in collection['scheduler']
    elif xc in ['PBE','r2SCAN'] and second:
        job='20972819' if xc=='PBE' else '20972837'
        revision,folder_step,day='production-05','force-002','20261101'
        expected_steps=29 if xc=='PBE' else 16
        end_local='2026-11-01T07:48:38+05:00' if xc=='PBE' else '2026-11-01T14:19:38+05:00'
        elapsed='11:02:40' if xc=='PBE' else '12:24:53'
        collection=json.loads((RECORD/(xc+'-force-002-production-05-collection-20261101.json')).read_text())
        assert job+'|COMPLETED|' in collection['scheduler'] and '|0:0' in collection['scheduler']
    elif xc=='PBE':
        assert step=='force-001'
        job,revision,folder_step,expected_steps,day='20968908','production-02','force-001-restart-05',24,'20261031'
        end_local,elapsed='2026-10-31T19:31:36+05:00','09:04:39'
        collection=json.loads((RECORD/'PBE-force001-result-collection-20261031.json').read_text())
        assert job+'|COMPLETED|' in collection['scheduler'] and '|0:0' in collection['scheduler']
    elif xc=='r2SCAN':
        assert step=='force-001'
        job,revision,folder_step,expected_steps,day='20968910','production-02','force-001-restart-04',10,'20261031'
        end_local,elapsed='2026-10-31T20:44:37+05:00','08:18:35'
    else:
        assert xc=='PBEsol'
        if reference:
            job,revision,folder_step,expected_steps,day='20968907','production-04','force-000',28,'20261031'
            end_local,elapsed='2026-10-31T20:55:40+05:00','11:09:00'
    folder = RECORD/'collected'/xc/revision/folder_step
    snap = RECORD/'snapshot'/xc
    text = (folder/'OUTCAR').read_text(errors='replace')
    assert 'aborting loop because EDIFF is reached' in text
    assert 'General timing and accounting informations' in text
    # STOPCAR is not evidence against convergence if the final electronic step
    # independently satisfies both EDIFF criteria and normal termination.
    checkpoint_requested=(folder/'checkpoint-request.json').exists()
    state = json.loads((folder/'status.json').read_text())
    assert state['state'] == 'validated' and state['job_id'] == job
    hashes = json.loads((folder/'input-hashes.json').read_text())
    for name in ['INCAR','POSCAR','KPOINTS']:
        assert digest(folder/name) == hashes[name], name
    provenance = json.loads((folder/'potcar-provenance.json').read_text())
    assert hashes['POTCAR'] == provenance['potcar_sha256'] == '1a7a8efddf2a5b6c8861aa48543d0531abc8e9b6e54500458b415b06354c25ba'
    for name in ['POSCAR','KPOINTS']:
        assert digest(folder/name) == digest(snap/'recovery-01'/step/name)
    xml = ET.parse(folder/'vasprun.xml').getroot()
    assert len(xml.findall('calculation')) == 1
    calc = xml.find('calculation')
    steps = calc.findall('scstep')
    assert len(steps) == expected_steps
    actual = {i.attrib['name']:i.text.strip() for i in xml.findall('incar/i')}
    for key,value in {**({'METAGGA':'R2SCAN'} if xc=='r2SCAN' else {'GGA':'PE' if xc=='PBE' else 'PS'}),'ALGO':'Normal','PREC':'Accurate','ISPIN':'1','ISYM':'0','NSW':'0','LASPH':'T','LREAL':'F','NBANDS':'576'}.items():
        assert actual[key] == value, (key, actual.get(key))
    for key,value in [('KPAR',8),('NCORE',4)]:
        assert re.search(r'\b'+key+r'\s*=\s*'+str(value)+r'\b',text), key
    assert float(actual['ENCUT']) == 550 and float(actual['EDIFF']) == 1e-8
    assert float(actual['SIGMA']) == .1 and actual['ISMEAR'] == '0'
    forces = vectors(calc.find("varray[@name='forces']"))
    assert len(forces) == 96 and all(len(v)==3 for v in forces)
    assert all(math.isfinite(x) for v in forces for x in v)
    outblock = text.rsplit('TOTAL-FORCE (eV/Angst)',1)[1].splitlines()[2:98]
    outforces = [[float(x) for x in line.split()[-3:]] for line in outblock]
    force_error = difference(forces,outforces)
    assert force_error <= 5.01e-7
    with (folder/'forces.npy').open('rb') as f:
        assert f.read(6) == b'\x93NUMPY'
        version = f.read(2)
        assert version == b'\x01\x00'
        n = struct.unpack('<H',f.read(2))[0]
        h = ast.literal_eval(f.read(n).decode())
        assert h['shape'] == (96,3) and h['descr'] == '<f8' and not h['fortran_order']
        flat = struct.unpack('<288d',f.read())
    assert difference(forces,[flat[i:i+3] for i in range(0,288,3)]) < 1e-14
    with (folder/'POSCAR').open() as f: pos = header(f)
    with (snap/'recovery-01/force-000/POSCAR').open() as f: ref = header(f)
    assert pos['species'] == ref['species'] == ['Ti'] and pos['counts'] == [96]
    assert difference(pos['cell'],ref['cell']) < 1e-12
    displacement=[]
    for i,(a,b) in enumerate(zip(pos['positions'],ref['positions'])):
        frac=[x-y-round(x-y) for x,y in zip(a,b)]
        cart=[sum(frac[j]*pos['cell'][j][k] for j in range(3)) for k in range(3)]
        if math.dist(cart,[0,0,0]) > 1e-8: displacement.append((i+1,cart))
    if reference:
        assert not displacement
    else:
        assert len(displacement)==1 and displacement[0][0]==1
        assert abs(math.dist(displacement[0][1],[0,0,0])-.01) < 1e-12
        disptext=(snap/'recovery-01/phonopy_disp.yaml').read_text().split('displacements:')[1]
        expected=[float(x) for x in re.findall(r'\[([^]]+)\]',disptext)[1 if second else 0].split(',')]
        assert max(abs(x-y) for x,y in zip(displacement[0][1],expected)) < 1e-12
    structnode=calc.find('structure')
    assert difference(vectors(structnode.find("crystal/varray[@name='basis']")),pos['cell']) < 1e-7
    final_positions=vectors(structnode.find("varray[@name='positions']"))
    # VASP wraps the negative displacement back into [0,1) fractional coordinates.
    periodic_error=max(abs(a-b-round(a-b)) for row,ref in zip(final_positions,pos['positions']) for a,b in zip(row,ref))
    assert periodic_error < 1e-7
    scfline=re.findall(r'DAV:.*', (folder/'OSZICAR').read_text())[-1].split()
    de,deps=float(scfline[3]),float(scfline[4])
    assert abs(de)<1e-8 and abs(deps)<1e-8
    eq=json.loads((snap/'equilibrium-02/validation.json').read_text())
    assert eq['max_force_eV_A'] < 1e-4 and eq['max_stress_kbar'] < .2
    drift=[sum(v[k] for v in forces) for k in range(3)]
    report=dict(job_id=job,functional=xc,step=step,accepted=True,
                scientific_scope='One reference supercell; DOS and convergence diagnostics remain incomplete' if reference else 'One displaced supercell; DOS and convergence diagnostics remain incomplete',
                scheduler_state='COMPLETED',exit_code='0:0',end_local=end_local,elapsed=elapsed,
                scf_iterations=expected_steps,final_energy_change_eV=de,final_band_energy_change_eV=deps,
                ediff_eV=1e-8,EDIFF_reached=True,normal_termination=True,checkpoint_requested=checkpoint_requested,
                supercell=[4,4,3],atoms=96,kmesh=[6,6,6],encut_eV=550,
                displaced_atom_1based=None if reference else 1,displacement_A=[0,0,0] if reference else displacement[0][1],displacement_norm_A=0 if reference else .01,
                forces_XML_OUTCAR_max_component_difference_eV_A=force_error,
                forces_NPY_XML_identical=True,force_sum_eV_A=drift,
                max_force_eV_A=max(math.dist(v,[0,0,0]) for v in forces),
                equilibrium_max_force_eV_A=eq['max_force_eV_A'],equilibrium_max_stress_kbar=eq['max_stress_kbar'],
                input_hashes=hashes,output_hashes={n:digest(folder/n) for n in ['OUTCAR','vasprun.xml','forces.npy','CONTCAR']})
    if reference:
        report['reference_supercell_max_stress_kbar']=json.loads((folder/'validation.json').read_text())['max_stress_kbar']
        report['stress_note']='Primitive equilibrium gate passed; supercell stress is recorded separately for cutoff/kmesh diagnostics. Electronic convergence does not establish stress or force convergence.'
    results=ROOT/'results'
    results.mkdir(exist_ok=True)
    (results/(xc+'-'+step.replace('-','')+'-validation-'+day+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (results/(xc+'-'+step.replace('-','')+'-forces.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['atom_1based','Fx_eV_A','Fy_eV_A','Fz_eV_A'])
        w.writerows((i+1,*v) for i,v in enumerate(forces))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else 'force-001',sys.argv[2] if len(sys.argv)>2 else 'PBEsol')
