"""Quota-aware storage checks and a bounded CPU write probe for GRACE runs."""
from pathlib import Path
import argparse,json,os,re,subprocess,time

def quota_snapshot(minimum_bytes=0):
    p=subprocess.run(['quota','-w'],capture_output=True,text=True,timeout=20)
    match=re.search(r'^\s*stg06:/home6\s+(\d+)\*?\s+(\d+)\s+(\d+)',p.stdout,re.M)
    if not match:raise RuntimeError('Cannot verify /home6 quota: '+p.stdout+p.stderr)
    used,soft,hard=map(int,match.groups());limit=min(v for v in [soft,hard] if v>0)
    record=dict(used_bytes=used*1024,limit_bytes=limit*1024,available_bytes=max(0,limit-used)*1024,required_bytes=minimum_bytes)
    if record['available_bytes']<minimum_bytes:raise RuntimeError('Insufficient /home6 quota headroom: '+json.dumps(record))
    return record

def main():
    root=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    before=quota_snapshot(4*2**30 if a.probe else 0);record=dict(status='PASS',before=before)
    if a.probe:
        assert os.environ.get('SLURM_JOB_ID');target=root/('quota-write-probe-'+os.environ['SLURM_JOB_ID']+'.tmp');assert not target.exists()
        start=time.monotonic()
        try:
            with target.open('xb') as f:
                block=b'\0'*(8*2**20)
                for _ in range(64):f.write(block)
                f.flush();os.fsync(f.fileno())
            assert target.stat().st_size==512*2**20
            record.update(write_bytes=target.stat().st_size,write_seconds=time.monotonic()-start,during=quota_snapshot())
        finally:
            if target.exists():target.unlink()
        record['after']=quota_snapshot(3*2**30)
        (root/'storage-probe.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
