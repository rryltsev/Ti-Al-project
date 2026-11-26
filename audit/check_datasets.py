from pathlib import Path
from collections import Counter
import numpy as np
import json, re, hashlib

B = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')

def key(symbols, coords, box):
    rows = sorted((s, *np.round(p, 5)) for s, p in zip(symbols, coords))
    return hashlib.sha256(repr((rows, np.round(box, 5).tolist())).encode()).hexdigest()

def xyz(path):
    out = []
    with path.open(encoding='utf-8') as f:
        while line := f.readline():
            if not line.strip(): continue
            n = int(line); header = f.readline()
            box = np.fromstring(re.search(r'Lattice="([^"]+)"', header)[1], sep=' ').reshape(3,3)
            rows = [f.readline().split() for _ in range(n)]
            syms = [r[0] for r in rows]
            pos = np.array([r[1:4] for r in rows], float)
            out.append((key(syms,pos,box), n, syms.count('Ti'), header))
    return out

selected = xyz(B/'MACE-dataset-gen/MACE-database/TiAl-liq-relax-ss_select-1000.xyz')
print('SELECTED', len(selected), 'sizes',Counter(r[1] for r in selected),'unique',len(set(r[0] for r in selected)))
print('HEADER',selected[0][3][:1000])
datasets={}
for p in sorted((B/'DPA4-finetune').glob('*/npy_format')):
    splits={}; report={}
    for split in ['train','validation','valid','test']:
        files=list((p/split).rglob('coord.npy'))
        if not files:continue
        keys=[]; bad=0; atomcounts=Counter()
        for f in files:
            system=f.parent.parent
            types=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
            mapping=(system/'type_map.raw').read_text().split()
            syms=[mapping[t] for t in types]; n=len(syms)
            coords=np.load(f).reshape(-1,n,3); boxes=np.load(f.with_name('box.npy')).reshape(-1,3,3)
            for kind in ['coord','box','energy','force','virial']:
                a=np.load(f.with_name(kind+'.npy'));bad+=int(not np.isfinite(a).all())
            keys += [key(syms,c,b) for c,b in zip(coords,boxes)]
            atomcounts[n]+=len(coords)
        splits[split]=set(keys)
        report[split]={'frames':len(keys),'systems':len(files),'unique':len(set(keys)),'sizes':dict(atomcounts),'nonfinite_arrays':bad}
    for f in p.glob('*.xyz'):
        x=xyz(f); print('FINALXYZ',p.parent.name,f.name,len(x),Counter(r[1] for r in x),'selected_overlap',len(set(r[0] for r in x)&set(r[0] for r in selected)))
    datasets[p.parent.name]=splits
    print('DATASET',p.parent.name,report,'split_overlap',[(a,b,len(splits[a]&splits[b])) for a in splits for b in splits if a<b])
for a in datasets:
    for b in datasets:
        if a>=b:continue
        print('PAIR',a,b,[(s,t,len(datasets[a][s]&datasets[b][t])) for s in datasets[a] for t in datasets[b]])

for p in (B/'MD-results/melting/beta_Ti_two-phase').iterdir():
    if not p.is_dir():continue
    print('MELTING',p.name,[x.name for x in p.iterdir()])
    dumps=list(p.rglob('*.xyz'));print('dump sample',str(dumps[0].relative_to(B)) if dumps else '')
    if dumps:
        with dumps[0].open() as f:print('dump header',''.join(f.readline() for _ in range(3)))
