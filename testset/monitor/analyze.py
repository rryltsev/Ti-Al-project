"""Paired cumulative Test50 analysis; raw energies and immutable model caches."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import energy_alignment

ROOT = Path(__file__).resolve().parent
XC = ['PBE','PBEsol','r2SCAN']
FAMILIES = {'liquid':'Liquid pool','POSCAR_alpha-Ti-hcp_orthogonal':'hcp',
            'POSCAR_beta-Ti-bcc':'bcc','POSCAR_gamma_AlTi_L1_0':r'$\gamma$-TiAl'}
COLORS = ['#276FBF','#D87C27','#36936B','#9B5EA3']
MODEL_COLORS = {'Fine-tuned':'#276FBF','Pretrained':'#D97917','Aligned pretrained':'#20804B'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def savefig(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def metric(errors,refs=None,bootstrap=2000):
    """Resample entire structures, retaining correlated force components."""
    arrays=[np.asarray(e,float).ravel() for e in errors if np.size(e)]
    if not arrays:
        return None
    assert all(np.isfinite(a).all() for a in arrays)
    n=np.array([len(a) for a in arrays])
    absolute=np.array([np.abs(a).sum() for a in arrays])
    squared=np.array([(a*a).sum() for a in arrays])
    all_errors=np.concatenate(arrays)
    m={'n_structures':len(arrays),'n_values':int(n.sum()),
       'mae':float(absolute.sum()/n.sum()),'rmse':float(np.sqrt(squared.sum()/n.sum())),
       'bias':float(all_errors.mean()),'max_abs':float(np.abs(all_errors).max()),
       'p95_abs':float(np.quantile(np.abs(all_errors),.95)),
       'equal_structure_mae':float(np.mean(absolute/n)),
       'equal_structure_rmse':float(np.sqrt(np.mean(squared/n)))}
    if len(arrays)>=2:
        draws=np.random.default_rng(20261022).integers(0,len(arrays),size=(bootstrap,len(arrays)))
        den=n[draws].sum(axis=1)
        m['mae_ci95']=np.quantile(absolute[draws].sum(axis=1)/den,[.025,.975]).tolist()
        m['rmse_ci95']=np.quantile(np.sqrt(squared[draws].sum(axis=1)/den),[.025,.975]).tolist()
    else:
        m['mae_ci95']=m['rmse_ci95']=None
    if refs is not None:
        ref=np.concatenate([np.asarray(r).ravel() for r in refs if np.size(r)])
        den=np.sum((ref-ref.mean())**2)
        m['r2']=float(1-squared.sum()/den) if den>1e-24 else None
    return m


def coverage(bundle,out):
    manifest=bundle['manifest'];ready=set(bundle['common_ids'])
    counts=Counter(r['metadata']['family'] for r in bundle['records'])
    totals=Counter(r['family'] for r in manifest)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    bins={}
    for i,(family,title) in enumerate(FAMILIES.items()):
        all_rows=[r for r in manifest if r['family']==family]
        done=[r for r in all_rows if r['structure_id'] in ready]
        axes[0].barh(i,totals[family],color='#e5e9ed')
        axes[0].barh(i,counts[family],color=COLORS[i])
        axes[0].text(totals[family]+.4,i,f'{counts[family]}/{totals[family]}',va='center')
        axes[1].scatter([r['x_Al'] for r in all_rows],np.full(len(all_rows),i),marker='o',facecolors='none',edgecolors='#adb4bd',s=75)
        axes[1].scatter([r['x_Al'] for r in done],np.full(len(done),i),color=COLORS[i],s=35)
        bounds=[min(r['x_Al'] for r in all_rows),max(r['x_Al'] for r in all_rows)]
        def bin_id(r):
            return min(4,int(5*(r['x_Al']-bounds[0])/max(bounds[1]-bounds[0],1e-12)))
        bins[family]={'completed':counts[family],'planned':totals[family],
                      'composition_bins_ready':sorted({bin_id(r) for r in done}),
                      'composition_bins_planned':sorted({bin_id(r) for r in all_rows}),
                      'composition_range':bounds}
    for ax in axes:
        ax.set_yticks(range(4),list(FAMILIES.values()));ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False)
    axes[0].set(xlabel='Structures with all three DFT labels',xlim=(0,30))
    axes[1].set(xlabel='Al atomic fraction',xlim=(-.04,1.04))
    axes[1].set_title('Filled: common subset; open: frozen Test50',fontsize=10)
    fig.suptitle(f'Common DFT subset: {len(ready)}/50 structures — preliminary')
    savefig(fig,out/'coverage')
    return bins


def compare(records,predictions,xc):
    values=[]
    for r in records:
        sid=r['metadata']['structure_id'];label,pred=r['labels'][xc],predictions[xc][sid]
        n=len(r['symbols']);ref_f,pred_f=np.array(label['forces_eV_A']),np.array(pred['forces_eV_A'])
        assert ref_f.shape==pred_f.shape==(n,3)
        values.append({'id':sid,'family':r['metadata']['family'],'x_Al':r['metadata']['x_Al'],
                       'symbols':np.array(r['symbols']),
                       'e_ref':label['energy_eV']/n,'e_pred':pred['energy_eV']/n,
                       'f_ref':ref_f,'f_pred':pred_f,
                       's_ref':np.array(label['stress_eV_A3'])[np.triu_indices(3)]*160.21766208,
                       's_pred':-np.array(pred['virial_eV'])[np.triu_indices(3)]/label['volume_A3']*160.21766208})
    def group(rows):
        if not rows:
            return None
        return {'n_structures':len(rows),
                'energy_meV_atom':metric([[1000*(v['e_pred']-v['e_ref'])] for v in rows],[[1000*v['e_ref']] for v in rows]),
                'force_eV_A':metric([v['f_pred']-v['f_ref'] for v in rows],[v['f_ref'] for v in rows]),
                'stress_GPa':metric([v['s_pred']-v['s_ref'] for v in rows]),
                'force_Al_eV_A':metric([(v['f_pred']-v['f_ref'])[v['symbols']=='Al'] for v in rows]),
                'force_Ti_eV_A':metric([(v['f_pred']-v['f_ref'])[v['symbols']=='Ti'] for v in rows])}
    groups={'all':group(values),'solid':group([v for v in values if v['family']!='liquid'])}
    groups.update({f:group([v for v in values if v['family']==f]) for f in FAMILIES})
    groups['composition_bins']={str(i):group([v for v in values if min(4,int(v['x_Al']*5))==i]) for i in range(5)}
    return values,groups


def parity(all_values,metrics,out,n,pretrained_values=None,pretrained_metrics=None,aligned_values=None,aligned_metrics=None,model_titles=None):
    fig,axs=plt.subplots(2,3,figsize=(12,7.6),constrained_layout=True)
    residual,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    series=[('Fine-tuned',MODEL_COLORS['Fine-tuned'],all_values,metrics)]
    if pretrained_values is not None:
        series.insert(0,('Pretrained',MODEL_COLORS['Pretrained'],pretrained_values,pretrained_metrics))
    if aligned_values is not None:
        series.insert(1,('Aligned pretrained',MODEL_COLORS['Aligned pretrained'],aligned_values,aligned_metrics))
    handles=[Line2D([],[],color=color,marker='o',markerfacecolor='none' if label=='Pretrained' else color,
                    linestyle='',label=label) for label,color,_,_ in reversed(series)]
    for col,xc in enumerate(XC):
        for label,color,values,_ in series:
            for family,marker in zip(FAMILIES,['o','s','^','D']):
                rows=[v for v in values[xc] if v['family']==family]
                if not rows:
                    continue
                axs[0,col].scatter([v['e_ref'] for v in rows],[v['e_pred'] for v in rows],
                                   s=52 if label=='Pretrained' else 25,
                                   facecolors='none' if label=='Pretrained' else color,
                                   edgecolors=color,linewidths=1,marker=marker,alpha=.9)
                if label!='Aligned pretrained':
                    axs[1,col].scatter(np.concatenate([v['f_ref'].ravel() for v in rows]),np.concatenate([v['f_pred'].ravel() for v in rows]),
                                  s=4,alpha=.23 if label=='Pretrained' else .18,color=color,rasterized=True)
                axes[col].scatter([v['x_Al'] for v in rows],[1000*(v['e_pred']-v['e_ref']) for v in rows],s=28,color=color,marker=marker)
        for row,key,unit in [(0,'energy_meV_atom','meV/atom'),(1,'force_eV_A','eV/Å')]:
            ax=axs[row,col];lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1])
            ax.plot([lo,hi],[lo,hi],'--',color='#666',lw=.8,zorder=0)
            ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal')
            row_series=[s for s in series if row==0 or s[0]!='Aligned pretrained']
            for i,(label,color,_,stats) in enumerate(reversed(row_series)):
                m=stats[xc]['all'][key]
                ax.text(.035,.965-i*.14,f"{label}: RMSE {m['rmse']:.3g}\nMAE {m['mae']:.3g} {unit}",
                        color=color,transform=ax.transAxes,va='top',fontsize=8,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1.5))
            ax.spines[['top','right']].set_visible(False)
        title=(model_titles or {}).get(xc,xc)
        axs[0,col].set(title=title,xlabel='DFT energy (eV/atom)',ylabel='DPA4 energy (eV/atom)')
        axs[1,col].set(xlabel='DFT force component (eV/Å)',ylabel='DPA4 force component (eV/Å)')
        axs[0,col].legend(handles=handles,loc='lower right',fontsize=8)
        axes[col].axhline(0,color='#666',lw=.8)
        axes[col].set(title=title,xlabel='Al atomic fraction',ylabel='Energy error (meV/atom)')
    axes[0].legend(handles=handles,fontsize=8)
    present={v['family'] for v in all_values[XC[0]]}
    if len(present)>1:
        family_handles=[Line2D([],[],color='#666',marker=marker,linestyle='',label=FAMILIES[family])
                        for family,marker in zip(FAMILIES,['o','s','^','D']) if family in present]
        fig.legend(handles=family_handles,loc='outside lower center',ncol=4,fontsize=8)
    energy_mode='raw + training-aligned energies' if aligned_values is not None else 'raw energies, no fitted offsets'
    fig.suptitle(f'DPA4-Neo vs DFT | same {n} structures | {energy_mode} | preliminary',fontsize=12)
    residual.suptitle('Energy errors: '+('raw and training-aligned' if aligned_values is not None else 'fine-tuned and raw')+' pretrained DPA4-Neo',fontsize=12)
    savefig(fig,out/'parity');savefig(residual,out/'energy_residuals')


def alignment_figure(values,metrics,aligned,aligned_metrics,out,n,model_titles=None):
    fig,axes=plt.subplots(2,3,figsize=(12,7.8),constrained_layout=True)
    for col,xc in enumerate(XC):
        for label,color,vs,ms,marker in [('Aligned pretrained','#20804B',aligned,aligned_metrics,'D'),('Fine-tuned','#276FBF',values,metrics,'o')]:
            rows=vs[xc];m=ms[xc]['all']['energy_meV_atom']
            axes[0,col].scatter([v['e_ref'] for v in rows],[v['e_pred'] for v in rows],s=38,marker=marker,color=color,label=label)
            axes[1,col].scatter([v['x_Al'] for v in rows],[1000*(v['e_pred']-v['e_ref']) for v in rows],s=38,marker=marker,color=color)
            i=0 if label=='Fine-tuned' else 1
            ci=m['rmse_ci95'];interval=f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ''
            axes[0,col].text(.035,.965-i*.17,f"{label}\nRMSE {m['rmse']:.2f}{interval}\nMAE {m['mae']:.2f} meV/atom",transform=axes[0,col].transAxes,va='top',fontsize=8,color=color,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
        ax=axes[0,col];lo=min(*ax.get_xlim(),*ax.get_ylim());hi=max(*ax.get_xlim(),*ax.get_ylim())
        ax.plot([lo,hi],[lo,hi],'--',color='#666',lw=.8,zorder=0)
        title=(model_titles or {}).get(xc,xc)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal',title=title,xlabel='DFT energy (eV/atom)',ylabel='DPA4 energy (eV/atom)')
        ax.legend(loc='lower right',fontsize=8)
        axes[1,col].axhline(0,color='#666',lw=.8)
        axes[1,col].set(xlabel='Al atomic fraction',ylabel='Energy error (meV/atom)',title=title)
    for ax in axes.ravel(): ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Training-only energy reference alignment | same {n} Test50 structures | preliminary\nFixed Al/Ti offsets; brackets: descriptive 95% structure-bootstrap RMSE intervals',fontsize=11)
    savefig(fig,out/'energy_alignment')


def comparable(h,report):
    return (h.get('models')==report['models'] and h.get('analysis_sha256')==report['analysis_sha256']
            and h.get('pretrained_model')==report.get('pretrained_model')
            and h.get('alignment')==report.get('alignment')
            and h.get('alignment_code_sha256')==report.get('alignment_code_sha256'))


def readiness(report,history):
    reasons=[]
    if len(report['common_ids'])<30:
        reasons.append('Fewer than 30 common structures (screening threshold, not a statistical guarantee).')
    for family,c in report['coverage'].items():
        required=12 if family=='liquid' else 4
        if c['completed']<required:
            reasons.append(f'{family}: {c["completed"]} structures; screening minimum {required}.')
        if len(c['composition_bins_ready'])<min(3,len(c['composition_bins_planned'])):
            reasons.append(f'{family}: composition coverage insufficient.')
    if not report['metrics']:
        reasons.append('Required models/predictions unavailable; accuracy and stability cannot be evaluated.')
    else:
        for xc in XC:
            for key in ['energy_meV_atom','force_eV_A']:
                m=report['metrics'][xc]['all'][key];ci=m['rmse_ci95'];floor=1. if key.startswith('energy') else .005
                if ci is None or ci[1]-ci[0]>max(.5*m['rmse'],floor):
                    reasons.append(f'{xc} {key}: RMSE bootstrap interval too wide for screening.')
        checkpoints=[]
        for h in reversed(history):
            if not comparable(h,report):
                continue
            if not checkpoints or len(checkpoints[-1]['common_ids'])-len(h['common_ids'])>=5:
                checkpoints.append(h)
            if len(checkpoints)==3:
                break
        if len(checkpoints)<3:
            reasons.append('Need three comparable checkpoints separated by at least five new structures.')
        else:
            for xc in XC:
                for key in ['energy_meV_atom','force_eV_A']:
                    for stat in ['mae','rmse']:
                        vals=[h['metrics'][xc]['all'][key][stat] for h in checkpoints]
                        floor=1. if key.startswith('energy') else .005
                        if max(vals)-min(vals)>max(.1*abs(vals[0]),floor):
                            reasons.append(f'{xc} {key} {stat}: changing across checkpoints.')
    return {'candidate_for_discussion':not reasons,'automatic_stop':False,'reasons':reasons,
            'qualification':'Heuristic discussion trigger based on fine-tuned models only. The pretrained baseline does not change the stopping rule. Report adaptive stopping, completion-order bias, coverage and uncertainty. Full fixed Test50 remains preferred.'}


def prediction_path(cache_key,sid,cache_directory=None):
    directory=cache_directory or cache_key
    assert Path(directory).name==directory and directory not in ('.','..')
    return ROOT/'predictions'/directory/(sid+'.json')


def load_predictions(bundle,cache_key,model_hash,cache_manifest,pending,cache_directory=None):
    result={}
    for r in bundle['records']:
        sid=r['metadata']['structure_id'];p=prediction_path(cache_key,sid,cache_directory)
        if not p.exists() or not model_hash:
            pending.append(cache_key+'/'+sid);continue
        pred=read(p)
        assert cache_manifest is not None and cache_manifest[p.relative_to(ROOT).as_posix()]==sha(p), 'Prediction cache changed after verified collection'
        assert pred['model_sha256']==model_hash and pred['structure_id']==sid
        assert pred['poscar_sha256']==r['metadata']['poscar_sha256'] and pred['symbols']==r['symbols']
        assert pred['energy_offset_applied_eV']==0
        assert np.isfinite(pred['energy_eV']) and np.isfinite(pred['forces_eV_A']).all() and np.isfinite(pred['virial_eV']).all()
        if cache_key=='pretrained':
            assert pred['model_role']=='pretrained'
        else:
            assert pred['functional']==cache_key
        result[sid]=pred
    return result


def convergence(report,history,out):
    series=[('Fine-tuned','metrics')]
    if report.get('pretrained_metrics'):
        series.append(('Pretrained','pretrained_metrics'))
    if report.get('pretrained_aligned_metrics'):
        series.append(('Aligned pretrained','pretrained_aligned_metrics'))
    fig,axs=plt.subplots(len(series),2,figsize=(10,3.4*len(series)),squeeze=False,constrained_layout=True)
    checkpoints=[h for h in history if comparable(h,report)]
    for row,(name,section) in enumerate(series):
        for xc,color in zip(XC,COLORS):
            for ax,key in zip(axs[row],['energy_meV_atom','force_eV_A']):
                ns=[len(h['common_ids']) for h in checkpoints];vals=[h[section][xc]['all'][key]['rmse'] for h in checkpoints]
                ax.plot(ns,vals,'o-',label=xc,color=color)
                intervals=[h[section][xc]['all'][key]['rmse_ci95'] for h in checkpoints]
                if all(ci is not None for ci in intervals):
                    ax.fill_between(ns,[ci[0] for ci in intervals],[ci[1] for ci in intervals],color=color,alpha=.12)
                ax.set(title=name,xlabel='Common structures',ylabel='RMSE ('+('meV/atom' if key.startswith('energy') else 'eV/Å')+')')
        axs[row,0].legend()
    fig.suptitle('Cumulative estimates; shaded: structure-bootstrap intervals')
    if report['models']['PBE'].get('checkpoint_step'):
        fig.suptitle('Cumulative estimates | PBE '+str(report['models']['PBE']['checkpoint_step']//1000)+'k | current model version only\nShaded: descriptive structure-bootstrap intervals')
    savefig(fig,out/'convergence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-alignment',action='store_true',help='Separate article variant with frozen training-only alignment; output in results/article-aligned.')
    args=parser.parse_args()
    bundle=read(ROOT/'data/latest.json');registry=read(ROOT/'model_registry.json')
    model_titles={xc:xc+(f" ({m['checkpoint_step']//1000}k)" if m.get('checkpoint_step') else '') for xc,m in registry['models'].items()}
    pretrained=read(ROOT/'pretrained_registry.json') if (ROOT/'pretrained_registry.json').exists() else None
    if pretrained:
        assert pretrained['status']=='frozen'
    calibration,alignment_pin=energy_alignment.load(pretrained['sha256']) if pretrained and args.with_alignment else (None,None)
    cache_manifest=read(ROOT/'prediction-manifest.json') if (ROOT/'prediction-manifest.json').exists() else None
    out=ROOT/'results'
    if args.with_alignment: out=out/'article-aligned'
    out.mkdir(parents=True,exist_ok=True)
    report={'unix':bundle['unix'],'common_ids':bundle['common_ids'],'validated_counts':bundle['validated_counts'],
            'coverage':coverage(bundle,out),'models':registry['models'],'metrics':None,
            'pretrained_model':pretrained,'pretrained_metrics':None,'pretrained_aligned_metrics':None,
            'alignment':alignment_pin,'alignment_coefficients':calibration['functionals'] if calibration else None,
            'analysis_mode':'article-aligned' if args.with_alignment else 'monitor-raw',
            'alignment_code_sha256':sha(energy_alignment.__file__) if calibration else None,
            'analysis_sha256':sha(__file__),'manifest_sha256':bundle['manifest_sha256'],
            'uncertainty':'2000 paired structure-bootstrap resamples; descriptive 95% percentile intervals, conditional on current subset. Unknown trajectory correlations and optional stopping are not covered. Aligned intervals condition on frozen training coefficients and exclude calibration uncertainty.',
            'energy_definition':'VASP TOTEN versus raw model energy and a separately reported training-only elemental alignment; meV/atom errors.',
            'force_definition':'All Cartesian components pooled; bootstrap samples whole structures. Equal-structure metrics and Al/Ti are also reported.',
            'pretrained_comparison':'The same unchanged pretrained predictions are compared with each functional on the same common geometries. Raw energy errors include energy-reference and functional differences. The added aligned variant uses fixed Al/Ti coefficients from training only, separately per functional. It was requested after raw test inspection, so this variant is not pre-registered blind analysis. Pretraining overlap is unknown.',
            'pending_predictions':[]}
    report['model_selection']=registry.get('model_selection')
    if not calibration:
        report['uncertainty']='2000 paired structure-bootstrap resamples; descriptive 95% percentile intervals, conditional on current subset. Unknown trajectory correlations and optional stopping are not covered.'
        report['energy_definition']='Raw VASP TOTEN versus raw model energy; no fitted offsets; meV/atom errors.'
        report['pretrained_comparison']='The same unchanged pretrained predictions are compared with each functional on the same common geometries. Raw energy errors include energy-reference and functional differences. Training-only alignment is preserved for a separate article variant and is not applied in monitoring. Pretraining overlap is unknown.'
    predictions={xc:load_predictions(bundle,xc,registry['models'][xc]['sha256'],cache_manifest,report['pending_predictions'],registry['models'][xc].get('prediction_cache')) for xc in XC}
    baseline=load_predictions(bundle,'pretrained',pretrained['sha256'],cache_manifest,report['pending_predictions']) if pretrained else None
    history_path=out/'history.json';history=read(history_path) if history_path.exists() else []
    if not report['pending_predictions'] and bundle['records']:
        all_values={};report['metrics']={};base_values=None;aligned_values=None
        for xc in XC:
            all_values[xc],report['metrics'][xc]=compare(bundle['records'],predictions,xc)
        if pretrained:
            base_values={};report['pretrained_metrics']={}
            same_predictions={xc:baseline for xc in XC}
            for xc in XC:
                base_values[xc],report['pretrained_metrics'][xc]=compare(bundle['records'],same_predictions,xc)
        if calibration:
            aligned_values={};report['pretrained_aligned_metrics']={}
            aligned={xc:{sid:energy_alignment.apply(pred,calibration['functionals'][xc]['coefficients_eV_per_atom']) for sid,pred in baseline.items()} for xc in XC}
            for xc in XC:
                aligned_values[xc],report['pretrained_aligned_metrics'][xc]=compare(bundle['records'],aligned,xc)
            alignment_figure(all_values,report['metrics'],aligned_values,report['pretrained_aligned_metrics'],out,len(bundle['records']),model_titles)
        parity(all_values,report['metrics'],out,len(bundle['records']),base_values,report['pretrained_metrics'],aligned_values,report['pretrained_aligned_metrics'],model_titles)
        identity={'ids':report['common_ids'],'models':report['models'],'pretrained_model':pretrained,
                  'labels':[[r['labels'][xc]['label_sha256'] for xc in XC] for r in bundle['records']],
                  'predictions':[[sha(prediction_path(xc,sid,registry['models'][xc].get('prediction_cache'))) for xc in XC] for sid in report['common_ids']],
                  'pretrained_predictions':[sha(ROOT/'predictions'/'pretrained'/(sid+'.json')) for sid in report['common_ids']] if pretrained else None,
                  'analysis':report['analysis_sha256'],'alignment':alignment_pin,'alignment_code':report['alignment_code_sha256']}
        report['input_identity']=identity
        report['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        if not any(h.get('fingerprint')==report['fingerprint'] for h in history):
            history.append(report.copy())
        write(history_path,history)
        convergence(report,history,out)
    report['readiness']=readiness(report,history);write(out/'latest.json',report)
    if report.get('fingerprint'):
        archive=out/f'n{len(bundle["records"]):02d}-{report["fingerprint"]}'
        if not archive.exists():
            archive.mkdir()
            for name in ['parity','energy_residuals','coverage','convergence']+(['energy_alignment'] if calibration else []):
                for ext in ['png','pdf']:
                    (archive/f'{name}.{ext}').write_bytes((out/f'{name}.{ext}').read_bytes())
            write(archive/'metrics.json',report)
    print(json.dumps({'common':len(report['common_ids']),'metrics_available':report['metrics'] is not None,
                      'pretrained_available':report['pretrained_metrics'] is not None,
                      'aligned_available':report['pretrained_aligned_metrics'] is not None,
                      'pending_predictions':len(report['pending_predictions']),'readiness':report['readiness']},indent=2))


if __name__=='__main__':
    main()
