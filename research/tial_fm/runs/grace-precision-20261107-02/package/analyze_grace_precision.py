"""Precision-induced prediction/phonon changes and paired inference cost."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from harmonic_worker import process
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','gpu'],required=True);a=ap.parse_args()
    source=ROOT/'results'/a.device;out=ROOT/'comparison'/a.device;out.mkdir(parents=True,exist_ok=False)
    summaries={};failures=[]
    for precision in ['fp32','fp64']:
        for block in [0,1]:
            key=f'{precision}-block{block}';p=source/key/'summary.json'
            if not p.exists():failures.append(key)
            else:summaries[key]=load(p);assert summaries[key]['status']=='PASS'
    if failures:
        (out/'summary.json').write_text(json.dumps(dict(status='INCOMPLETE_OR_FAILED',missing=failures),indent=2)+'\n');raise SystemExit(1)
    labels=load(ROOT/'inputs/panel-labels.json');pred={p:load(source/f'{p}-block0/panel-predictions.json') for p in ['fp32','fp64']}
    assert set(pred['fp32'])==set(pred['fp64'])==set(labels)
    de=[];df=[];ds=[];accuracy={}
    for sid,l in labels.items():
        x=pred['fp32'][sid];y=pred['fp64'][sid]
        de.append(1000*(x['energy']-y['energy'])/l['natoms']);df.extend((np.array(x['forces'])-np.array(y['forces'])).ravel());ds.extend((np.array(x['stress'])-np.array(y['stress']))*160.21766208)
    def metric(v):v=np.asarray(v);return dict(RMSE=float(np.sqrt(np.mean(v*v))),max_abs=float(np.max(abs(v))))
    delta=dict(E_meV_atom=metric(de),F_eV_A=metric(df),stress_GPa=metric(ds))
    for xc in ['PBE','r2SCAN']:
        accuracy[xc]={}
        for p in ['fp32','fp64']:
            es=[];fs=[];ss=[]
            for sid,l in labels.items():
                if l['XC']!=xc:continue
                r=pred[p][sid];es.append(1000*(r['energy']-l['energy'])/l['natoms']);fs.extend((np.array(r['forces'])-np.array(l['forces'])).ravel());ss.extend((np.array(r['stress'])-np.array(l['stress']))*160.21766208)
            accuracy[xc][p]=dict(E_raw_meV_atom=metric(es),F_eV_A=metric(fs),stress_GPa=metric(ss),structures=len(es))
    timing={}
    for name in [x['name'] for x in summaries['fp32-block0']['timings']]:
        row={}
        for p in ['fp32','fp64']:
            rr=[next(t for t in summaries[f'{p}-block{b}']['timings'] if t['name']==name) for b in [0,1]]
            values=[v for t in rr for v in t['wall_seconds']]
            row[p]=dict(median_ms=1000*float(np.median(values)),p10_ms=1000*float(np.quantile(values,.1)),p90_ms=1000*float(np.quantile(values,.9)),
                block_medians_ms=[1000*t['median_seconds'] for t in rr],cold_seconds=[t['cold_seconds'] for t in rr],natoms=rr[0]['natoms'],
                median_kernel_sync_ms=1000*float(np.median([v for t in rr for v in t['kernel_sync_seconds']])),
                GPU_peak_bytes=max(t['GPU_memory']['peak'] for t in rr) if a.device=='gpu' else None)
        row['fp32_speedup']=row['fp64']['median_ms']/row['fp32']['median_ms'];timing[name]=row
    reference=np.load(ROOT/'reference_figure.npz');harmonics={};bands={}
    for xc in ['PBE','r2SCAN']:
        rows={};fc={};bf={}
        for precision in ['fp32','fp64']:
            raw={k:np.load(source/f'{precision}-block0/harmonic_forces'/xc/(k+'.npy')) for k in ['force000','force001','force002']}
            hp,frequencies,info=process(ROOT/'harmonic_inputs'/xc,raw,out/'harmonics'/xc/precision,reference['qpoints'])
            fc[precision]=hp.force_constants.copy();bf[precision]=frequencies
            rows[precision]=dict(model=info,sorted_band_RMSE_to_DFT_THz=float(np.sqrt(np.mean((frequencies-reference[xc+'_DFT_bands'])**2))))
        rows['fp32_minus_fp64']=dict(band_RMSE_THz=float(np.sqrt(np.mean((bf['fp32']-bf['fp64'])**2))),band_max_abs_THz=float(np.max(abs(bf['fp32']-bf['fp64']))),
            gamma_max_abs_THz=float(np.max(abs(np.array(rows['fp32']['model']['gamma_THz'])-rows['fp64']['model']['gamma_THz']))),
            FC_relative_Frobenius=float(np.linalg.norm(fc['fp32']-fc['fp64'])/np.linalg.norm(fc['fp64'])))
        harmonics[xc]=rows;bands[xc]=bf
    memory={p:dict(parameter_bytes=summaries[f'{p}-block0']['parameter_bytes'],peak_process_RSS_MiB=[summaries[f'{p}-block{b}']['peak_process_RSS_KiB']/1024 for b in [0,1]]) for p in ['fp32','fp64']}
    fds={p:load(source/f'{p}-block0/finite-differences.json') for p in ['fp32','fp64']}
    result=dict(status='PASS',device=a.device,checkpoint_equivalence=load(ROOT/'weight-equivalence.json'),precision_delta=delta,subset_DFT_error=accuracy,timing=timing,memory=memory,finite_differences=fds,harmonics=harmonics,
        qualification='PASS means complete finite results, not final model selection. Eight fixed train structures per XC only, no validation/test ranking. CPU main graph JIT off; GPU main graph JIT on. ABBA fresh processes, all-calls cache-bypassed, warmup excluded; model+sync and full ASE timings retained. Not LAMMPS/Kokkos or long-MD validation.',
        pending='GPU measurements and full GRACE B0/H0/training qualification' if a.device=='cpu' else 'Full GRACE B0/H0/training qualification')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    x=np.arange(len(timing));width=.34
    for i,p in enumerate(['fp32','fp64']):axes[0].bar(x+(i-.5)*width,[r[p]['median_ms'] for r in timing.values()],width,label=p)
    axes[0].set_xticks(x,list(timing));axes[0].set_ylabel('Median E/F/stress latency (ms)');axes[0].set_title(a.device.upper()+' / ASE total');axes[0].legend()
    for ax,xc in zip(axes[1:],['PBE','r2SCAN']):
        ax.plot(bands[xc]['fp32']-bands[xc]['fp64'],linewidth=.7);ax.set_title(xc+': fp32 − fp64');ax.set_xlabel('Matched band-path point');ax.set_ylabel('Frequency difference (THz)')
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    text=f'# GRACE fp32/fp64: {a.device.upper()}\n\nControlled fp32 cast of the pinned fp64 checkpoint; model tensors are identical after dtype conversion. Complete diagnostics: PASS.\n\n'
    text+='| Structure | fp32, ms | fp64, ms | fp32 speedup |\n|---|---:|---:|---:|\n'
    for name,r in timing.items():text+=f'| {name} | {r["fp32"]["median_ms"]:.2f} | {r["fp64"]["median_ms"]:.2f} | {r["fp32_speedup"]:.2f}x |\n'
    text+='\nPrecision deltas: '+json.dumps(delta)+'\n\n'+result['qualification']+'\n\nPending: '+result['pending']+'\n'
    (out/'report.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(status='PASS',device=a.device,precision_delta=delta,speedups={k:r['fp32_speedup'] for k,r in timing.items()},phonon_deltas={xc:r['fp32_minus_fp64'] for xc,r in harmonics.items()})))

if __name__=='__main__':main()
