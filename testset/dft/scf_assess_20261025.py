"""Descriptive, explicitly non-guaranteed SCF timing scenarios from snapshots."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

OUT=Path(__file__).resolve().parent/'scf-check-20261025'


def rows(value):
    result=[]
    for line in value['oszicar']['text'].splitlines():
        a=line.split()
        if a and a[0] in ('DAV:','RMM:','CG:'):
            result.append(dict(n=int(a[1]),energy_eV=float(a[2]),dE_eV=float(a[3]),
                               deps_eV=float(a[4]),rms=float(a[6]),
                               rms_charge=float(a[7]) if len(a)>7 else None))
    return result


def seconds(value):
    parts=value.split(':')
    return int(parts[0])*3600+int(parts[1])*60+int(parts[2])


if __name__=='__main__':
    sources=sorted(OUT.glob('snapshot-*.json'))
    b=json.loads(sources[-1].read_text(encoding='utf-8'))
    controls=b['completed_r2scan']
    assert len(controls)==20 and all(v['converged_marker'] for v in controls)
    overhead=[v['elapsed_seconds']-sum(v['loop_seconds']) for v in controls]
    assessment={'unix':b['unix'],'source':sources[-1].name,'targets':{},
      'control_count':20,'control_families':'liquid only; not independent crystal timing controls',
      'control_steps':sorted(len(rows(v)) for v in controls),
      'non_LOOP_overhead_seconds_range':[min(overhead),max(overhead)],
      'qualification':'Descriptive timing scenarios, not confidence bounds or guaranteed ETA. Linear log-residual fits may fail when convergence accelerates or stalls. No DFT changes.'}
    for sid,v in b['targets'].items():
        job=dict(re.findall(r'(\w+)=(\S*)',v['slurm']['stdout']))
        assert v['slurm']['returncode']==0 and job['JobState']=='RUNNING'
        assert job['TimeLimit']=='20:00:00' and job['Restarts']=='0'
        rs=rows(v);last=rs[-1];threshold=max(abs(last['dE_eV']),abs(last['deps_eV']))
        remains=[]
        for ctrl in controls:
            cr=rows(ctrl)
            stable=[j for j,x in enumerate(cr) if x['n']>=10 and all(max(abs(y['dE_eV']),abs(y['deps_eV']))<=threshold for y in cr[j:])]
            if stable:remains.append(len(cr)-stable[0]-1)
        loops=v['loop_seconds'][-5:]
        empirical=[(min(remains)*min(loops)+min(overhead))/3600,
                   (max(remains)*max(loops)+max(overhead))/3600]
        fits=[]
        for n in (5,8,10):
            a=rs[-n:];xx=[x['n'] for x in a];yy=[math.log10(abs(x['deps_eV'])) for x in a]
            xm=st.mean(xx);ym=st.mean(yy)
            slope=sum((x-xm)*(y-ym) for x,y in zip(xx,yy))/sum((x-xm)**2 for x in xx)
            steps=max(0,(-6-yy[-1])/slope) if slope<0 else None
            fits.append({'tail_steps':n,'log10_deps_slope_per_step':slope,'steps_to_EDIFF':steps,
                         'additional_hours_at_max_recent_step_plus_max_overhead':(math.ceil(steps)*max(loops)+max(overhead))/3600 if steps is not None else None})
        assessment['targets'][sid]={'job':job,'last_electronic_step':last,
          'runtime_hours':seconds(job['RunTime'])/3600,'remaining_hours':(seconds(job['TimeLimit'])-seconds(job['RunTime']))/3600,
          'loop_minutes_last5':[t/60 for t in loops], 'outcar_age_minutes':v['outcar_age_seconds']/60,
          'control_matched_remaining_steps':sorted(remains),'control_matched_additional_hours_range':empirical,
          'band_residual_extrapolations':fits,'SCF_warnings':v['warnings'],
          'stdout_warnings':[x for x in (v.get('StdOut') or {}).get('text','').splitlines() if 'WARNING' in x],
          'interpretation':{'TiAl-test-026':'Low timeout risk; descriptive practical estimate 3-5 additional hours.',
                            'TiAl-test-034':'Moderate timeout risk. Liquid-control analogy about 7-11 additional hours; slow-tail extrapolation about 14-15 hours can reach/exceed 20 h total. Reassess after a few completed iterations.',
                            'TiAl-test-043':'Low timeout risk; near EDIFF, about 0.5-1.5 additional hours including final work.'}[sid]}
    (OUT/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'n':v['last_electronic_step']['n'],'runtime_h':v['runtime_hours'],'remaining_h':v['remaining_hours'],
                       'empirical_additional_h':v['control_matched_additional_hours_range'],'band_scenarios_h':[x['additional_hours_at_max_recent_step_plus_max_overhead'] for x in v['band_residual_extrapolations']]} for k,v in assessment['targets'].items()},indent=2))
