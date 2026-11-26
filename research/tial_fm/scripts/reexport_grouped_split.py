"""Recreate DeepMD files from frozen group assignments, without re-optimization."""
from pathlib import Path
import argparse,json,shutil
from build_grouped_split import ROOT,export,group_key

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    frozen=a.frozen.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    records=[json.loads(x) for x in (frozen/'data_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    assignments={group_key(r):r['split'] for r in records}
    for name in ['splits.json','groups.json','data_manifest.jsonl','split_assignments.csv']:
        shutil.copyfile(frozen/name,out/name)
    result=export(out,records,assignments)
    (out/'reexport_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
