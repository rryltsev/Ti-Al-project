"""Gap resolution when an interior ordered interval may disappear in resampling."""
import numpy as np
from isotherms import HERE,load_json,atomic_json


def calculate():
    records=[]
    for f in sorted(HERE.glob("final_T[0-9]*.json")):
        d=load_json(f)
        for c in ("0.37","0.47"):
            b=d["bootstrap"]["cuts"][c]["replicates"];admitted=[]
            for r in b:
                if max(r["edge_scores"])>=0 or r["maximum_on_edge"]:
                    continue
                if r["apex"][4]<=0:
                    admitted.append((0.,0.))
                elif set(r["roots"])=={"Ti","Al"}:
                    admitted.append(tuple(100*abs(r["roots"][side][3]) for side in ("Ti","Al")))
            val=np.asarray(admitted)
            records.append(dict(T_K=d["T_K"],W=d["cap"],cut_Q=float(c),count=len(b),admitted=len(admitted),
                fraction_with_supported_range=len(admitted)/len(b),
                zero_width_fraction=float(np.mean(val[:,0]==0)) if val.size else None,
                median_abs_gaps_pp=np.median(val,axis=0).tolist() if val.size else None,
                one_sided_95_percentile_abs_gaps_pp=np.quantile(val,.95,axis=0).tolist() if val.size else None))
    atomic_json(HERE/"gap_resolution.json",dict(columns=["Ti-rich","Al-rich"],records=records,
        convention="For a bounded connected order dome a bootstrap draw contributes zero width when its interior "
                   "maximum is nonpositive. If positive it contributes the absolute conditional composition jump at each "
                   "of two measured-range roots. Draws with either boundary nonnegative or a maximum on the search "
                   "edge are excluded and counted. The percentiles quantify the resolution of this finite-cell estimator, "
                   "not an unconditional infinite-system confidence bound."))


if __name__=="__main__":calculate()
