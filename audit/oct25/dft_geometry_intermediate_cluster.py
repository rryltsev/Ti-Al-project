"""Submit and collect only the 10k/15k extension of the verified CPU protocol."""
from pathlib import Path
import base64, hashlib, json, shlex, sys, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'pbe_phonons_dft_geometry_intermediate'
OLD = HERE / 'pbe_phonons_dft_geometry'
REMOTE = '/home6/u3129/tial-phonons/pbe-dft-geometry-intermediate-20261025-01'
sys.path.insert(0, str(ROOT / 'testset/dft'))
import cluster_test50 as c

sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
OUT.mkdir(exist_ok=True)
def save(n, d):
    (OUT / n).write_text(json.dumps(d, indent=2), encoding='utf-8')

action = sys.argv[1]
if action == 'stage':
    assert not (OUT / 'staging-intent.json').exists(), 'Reconcile existing stage before retry'
    cfg = read(OLD / 'protocol.json')
    cfg['id'] = 'pbe-dft-geometry-intermediate-20261025-01'
    cfg['checkpoints'] = ['10000', '15000']
    models = read(HERE / 'pbe_checkpoint_sweep/remote-input.json')['models']
    cfg['models'] = {k: models[k] for k in cfg['checkpoints']}
    pos = (OLD / 'POSCAR-DFT').read_bytes()
    assert sha(pos) == cfg['geometry_sha256']
    files = {'protocol.json': json.dumps(cfg, indent=2).encode(), 'POSCAR-DFT': pos,
             'job.sbatch': (OLD / 'job.sbatch').read_bytes(),
             'dft_geometry_phonons_remote.py': (OLD / 'dft_geometry_phonons_remote.py').read_bytes()}
    hashes = {n: sha(b) for n, b in files.items()}
    save('staging-intent.json', {'remote': REMOTE, 'files': hashes, 'unix': time.time()})
    c.cluster.run('umask 077; mkdir ' + shlex.quote(REMOTE))
    for n, b in files.items():
        c.upload(b, REMOTE + '/' + n)
        (OUT / n).write_bytes(b)
    code = ('import pathlib,hashlib; p=pathlib.Path(' + repr(REMOTE) + '); h=' + repr(hashes)
            + '; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); '
            + 'models=' + repr(cfg['models'])
            + '; assert all(hashlib.sha256(pathlib.Path(v["path"]).read_bytes()).hexdigest()==v["sha256"] for v in models.values()); print("Inputs and both checkpoints verified")')
    print(c.cluster.run('python3 -c ' + shlex.quote(code)), flush=True)
    save('staging-complete.json', {'verified': True, 'files': hashes, 'unix': time.time()})
    args = ['sbatch', '--parsable', '--partition=apollo', '--account=imet-2026-2', '--qos=normal',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--time=00:30:00',
            '--array=0-1%2', '--no-requeue', '--job-name=pbe-dos-10k15k', '--chdir=' + REMOTE,
            '--output=' + REMOTE + '/run-%A_%a.out', REMOTE + '/job.sbatch']
    save('submission-intent.json', {'command': args, 'unix': time.time()})
    jid = c.cluster.run(shlex.join(args)).strip().split(';')[0]
    assert jid.isdigit(), jid
    save('submission.json', {'job_id': jid, 'remote': REMOTE, 'unix': time.time()})
    print('Submitted CPU array', jid)
else:
    jid = read(OUT / 'submission.json')['job_id']
    acct = c.cluster.run('sacct -j ' + jid + ' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT / 'accounting.txt').write_text(acct)
    print(acct, flush=True)
    code = ('import pathlib,json; p=pathlib.Path(' + repr(REMOTE)
            + '); print(json.dumps({f.name:f.read_text(errors="replace")[-5000:] for f in p.glob("run-*.out")}))')
    logs = json.loads(c.cluster.run('python3 -c ' + shlex.quote(code)))
    save('logs.json', logs)
    print(json.dumps({k: v.splitlines()[-1:] for k, v in logs.items()}, indent=2))
    if action == 'collect':
        records = {r.split('|')[0]: r.split('|') for r in acct.splitlines()}
        if not all(records.get(jid + '_' + str(i), ['', ''])[1:3] == ['COMPLETED', '0:0'] for i in range(2)):
            print('Collection deferred: jobs not both complete successfully.')
            sys.exit(0)
        manifest = {}
        for key in ['10000', '15000']:
            code = ('import pathlib,json,hashlib,base64; p=pathlib.Path(' + repr(REMOTE + '/' + key)
                    + '); print(json.dumps({str(f.relative_to(p)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),'
                    + '"base64":base64.b64encode(f.read_bytes()).decode()} for f in p.rglob("*") if f.is_file()}))')
            files = json.loads(c.cluster.run('python3 -c ' + shlex.quote(code)))
            for n, d in files.items():
                b = base64.b64decode(d['base64'])
                assert sha(b) == d['sha256']
                dest = OUT / key / n
                assert OUT.resolve() in dest.resolve().parents
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert dest.read_bytes() == b
                else:
                    dest.write_bytes(b)
                manifest[key + '/' + n] = d['sha256']
            print('Collected', key, len(files), flush=True)
        save('collection.json', {'job_id': jid, 'files': manifest, 'unix': time.time()})
