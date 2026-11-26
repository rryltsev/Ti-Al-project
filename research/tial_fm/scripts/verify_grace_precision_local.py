"""Independent recomputation of collected metrics and figure with timing spread."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_grace_precision import RUN,ROOT

def load(p):return json.loads(p.read_text())
def main():
    col=RUN/'collected';summary=load(col/'comparison/cpu/summary.json');manifest=load(RUN/'cpu-collection.json')
    for name,h in manifest['files'].items():assert hashlib.sha256((col/name).read_bytes()).hexdigest()==h,name
    pred={p:load(col/f'results/cpu/{p}-block0/panel-predictions.json') for p in ['fp32','fp64']};labels=load(RUN/'package/inputs/panel-labels.json')
    values={'E_meV_atom':[],'F_eV_A':[],'stress_GPa':[]}
    for sid,l in labels.items():
        x,y=pred['fp32'][sid],pred['fp64'][sid]
        values['E_meV_atom'].append((x['energy']-y['energy'])*1000/l['natoms'])
        values['F_eV_A'].extend((np.array(x['forces'])-y['forces']).ravel())
        values['stress_GPa'].extend((np.array(x['stress'])-y['stress'])*160.21766208)
    for key,v in values.items():
        v=np.array(v);assert abs(np.sqrt(np.mean(v*v))-summary['precision_delta'][key]['RMSE'])<1e-14
        assert abs(np.max(abs(v))-summary['precision_delta'][key]['max_abs'])<1e-14
    for xc in ['PBE','r2SCAN']:
        a=np.load(col/f'comparison/cpu/harmonics/{xc}/fp32/bands.npz')['frequencies'];b=np.load(col/f'comparison/cpu/harmonics/{xc}/fp64/bands.npz')['frequencies']
        assert abs(np.sqrt(np.mean((a-b)**2))-summary['harmonics'][xc]['fp32_minus_fp64']['band_RMSE_THz'])<1e-14
    dest=ROOT/'reports/grace-precision-20261107';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.4),constrained_layout=True);colors={'fp32':'#1874a2','fp64':'#c56b28'}
    x=np.arange(3);w=.34
    for j,p in enumerate(['fp32','fp64']):
        rows=[r[p] for r in summary['timing'].values()];med=np.array([r['median_ms']/1000 for r in rows]);low=np.array([r['p10_ms']/1000 for r in rows]);high=np.array([r['p90_ms']/1000 for r in rows]);pos=x+(j-.5)*w
        axs[0,0].bar(pos,med,w,label=p,color=colors[p],alpha=.85,yerr=np.stack([med-low,high-med]),capsize=3)
        for i,r in enumerate(rows):axs[0,0].scatter([pos[i]]*2,np.array(r['block_medians_ms'])/1000,s=16,color='black',zorder=4)
    axs[0,0].set_xticks(x,list(summary['timing']));axs[0,0].set_ylabel('E/F/stress call (s)');axs[0,0].set_title('4 CPU threads: median, P10–P90, process medians');axs[0,0].legend()
    for j,p in enumerate(['fp32','fp64']):
        r=summary['memory'][p];ys=np.array(r['peak_process_RSS_MiB'])/1024
        axs[0,1].bar(j,float(ys.mean()),.6,color=colors[p],alpha=.85);axs[0,1].scatter([j-.06,j+.06],ys,s=22,color='black',zorder=4)
    axs[0,1].set_xticks([0,1],['fp32','fp64']);axs[0,1].set_ylabel('Peak process RSS (GiB)');axs[0,1].set_title('TensorFlow process memory, two fresh processes')
    for p in ['fp32','fp64']:
        rows=summary['finite_differences'][p]['force_finite_difference'];axs[1,0].loglog([r['h_A'] for r in rows],[r['absolute_error_eV_A'] for r in rows],'o-',label=p,color=colors[p])
    axs[1,0].invert_xaxis();axs[1,0].set_xlabel('Displacement h (Å)');axs[1,0].set_ylabel('|finite-difference F − autodiff F| (eV/Å)');axs[1,0].set_title('Energy finite differences at one fixed component');axs[1,0].legend();axs[1,0].grid(alpha=.2)
    xs=np.arange(2);names=['PBE','r2SCAN']
    axs[1,1].bar(xs-.17,[summary['harmonics'][xc]['fp32_minus_fp64']['band_RMSE_THz'] for xc in names],.34,label='fp32 − fp64',color=colors['fp32'])
    axs[1,1].bar(xs+.17,[summary['harmonics'][xc]['fp64']['sorted_band_RMSE_to_DFT_THz'] for xc in names],.34,label='fp64 model − DFT',color='#777777')
    axs[1,1].set_yscale('log');axs[1,1].set_xticks(xs,names);axs[1,1].set_ylabel('Sorted-band RMSE (THz)');axs[1,1].set_title('Precision effect versus pretrained-model discrepancy');axs[1,1].legend()
    fig.savefig(dest/'cpu-precision.png',dpi=180);fig.savefig(dest/'cpu-precision.pdf');plt.close(fig)
    out=dict(status='PASS',collected_files_verified=len(manifest['files']),panel_delta_recomputed=True,phonon_delta_recomputed=True,figure=str(dest/'cpu-precision.png'),figure_SHA256=hashlib.sha256((dest/'cpu-precision.png').read_bytes()).hexdigest(),GPU_result_available=False)
    (RUN/'local-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
