"""Read-only DFT export. Execute through SSH; stdout is a JSON bundle."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import time

ROOT = Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
XC = ['PBE', 'PBEsol', 'r2SCAN']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    config = json.loads((ROOT / 'protocol.json').read_text())
    source = Path(config['dataset_root'])
    manifest = json.loads((source / 'manifest.json').read_text())
    labels = {xc: {} for xc in XC}
    statuses = {xc: {} for xc in XC}
    problems = []
    for row in manifest:
        sid = row['structure_id']
        for xc in XC:
            d = ROOT / 'calculations' / xc / sid
            if not (d / 'status.json').exists():
                statuses[xc][sid] = {'state': 'not_started'}
                continue
            status = json.loads((d / 'status.json').read_text())
            statuses[xc][sid] = status
            if status['state'] != 'validated':
                continue
            try:
                label = json.loads((d / 'label.json').read_text())
                outputs = json.loads((d / 'output-manifest.json').read_text())
                assert sha(d / 'label.json') == outputs['label.json']
                assert label['structure_id'] == sid and label['functional'] == xc
                assert label['input_sha256']['POSCAR'] == row['poscar_sha256'] == sha(d / 'POSCAR')
                assert label['scf_converged'] and label['ionic_calculations'] == 1
                assert label['max_position_change_A'] < 1e-6 and label['max_cell_change_A'] < 1e-7
                assert label['n_atoms'] == row['n_atoms']
                label['label_sha256'] = outputs['label.json']
                labels[xc][sid] = label
            except Exception as exc:
                problems.append({'functional': xc, 'structure_id': sid, 'error': repr(exc)})
    common = sorted(set.intersection(*(set(labels[xc]) for xc in XC)))
    records = []
    by_id = {r['structure_id']: r for r in manifest}
    for sid in common:
        row = by_id[sid]
        p = source / row['poscar']
        assert sha(p) == row['poscar_sha256']
        lines = p.read_text().splitlines()
        assert float(lines[1]) == 1 and lines[7].lower().startswith('d')
        elements, counts = lines[5].split(), list(map(int, lines[6].split()))
        symbols = [e for e, n in zip(elements, counts) for _ in range(n)]
        frac = [[float(v) for v in line.split()[:3]] for line in lines[8:8 + len(symbols)]]
        cell = [[float(v) for v in line.split()] for line in lines[2:5]]
        for field in ['KPOINTS', 'POTCAR']:
            assert len({labels[xc][sid]['input_sha256'][field] for xc in XC}) == 1
        records.append({'metadata': row, 'symbols': symbols, 'fractional_positions': frac,
                        'cell_A': cell, 'labels': {xc: labels[xc][sid] for xc in XC}})
    def command(args):
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    return {'schema': 1, 'unix': time.time(), 'remote_root': str(ROOT),
            'protocol_sha256': sha(ROOT / 'protocol.json'), 'manifest_sha256': sha(source / 'manifest.json'),
            'manifest': manifest, 'common_ids': common, 'records': records, 'problems': problems,
            'validated_counts': {xc: len(labels[xc]) for xc in XC},
            'status_counts': {xc: dict(Counter(s['state'] for s in statuses[xc].values())) for xc in XC},
            'statuses': statuses,
            'scheduler': command(['squeue', '-h', '-r', '-j', '20957366,20957369,20957372', '-o', '%F|%K|%T|%R']),
            'accounting': command(['sacct', '-X', '-j', '20957366,20957369,20957372', '--format=JobID,State,ExitCode,Elapsed', '-P'])}


if __name__ == '__main__':
    print(json.dumps(collect()))
