"""Summarize finite-size and cage differences for independently sampled Ti–Al ensembles."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def endpoints(fit):
    if fit is None:
        return None
    return dict(mu_eV=fit["mu_eV"],alpha_at_percent=100*fit["x_alpha"],
                alpha2_at_percent=100*fit["x_alpha2"],
                width_at_percent=100*(fit["x_alpha2"]-fit["x_alpha"]))


def vector(fit):
    result=endpoints(fit)
    return np.array(list(result.values()))


def draws(row):
    a=np.asarray(row.get("bootstrap",{}).get("joint_mu_xalpha_xalpha2",[]))
    if not len(a):
        return None
    return np.column_stack((a[:,0],100*a[:,1],100*a[:,2],100*(a[:,2]-a[:,1])))


def intervals(sample,columns):
    return {key:arr.tolist() for key,arr in zip(columns,np.percentile(sample,[2.5,97.5],axis=0).T)}


def difference(a,b):
    """b minus a; cross-product of independent bootstrap draws preserves asymmetric marginals."""
    if any(t["equal_single_variant"] is None for t in (a,b)):
        return None
    cols=list(endpoints(a["equal_single_variant"]))
    values=vector(b["equal_single_variant"])-vector(a["equal_single_variant"])
    out=dict(values=dict(zip(cols,values)),sign="second minus first")
    sa,sb=draws(a),draws(b)
    if sa is not None and sb is not None:
        deltas=(sb[None,:,:]-sa[:,None,:]).reshape(-1,len(cols))
        out["conditional_95"]=intervals(deltas,cols)
    return out


def reference_differences(row,reference):
    if not row["equal_single_variant"]:
        return None
    summary={}
    point=endpoints(row["equal_single_variant"])
    sa=draws(row)
    for col,phase in enumerate(("alpha","alpha2"),1):
        ref=reference["phase"][phase]["Al_at_percent"]
        l,r=reference["phase"][phase]["Al_at_percent_conditional_95"]
        item=dict(delta_at_percent=point[phase+"_at_percent"]-ref)
        if sa is not None:
            low,high=np.percentile(sa[:,col],[2.5,97.5])
            item["conditional_95_holding_C46_point_fixed"]=[low-ref,high-ref]
            item["span_from_marginal_95_intervals"]=[low-r,high-l]
        summary[phase]=item
    return summary


def group(row,reference):
    fit=row["equal_single_variant"]
    o=np.asarray(row["mbar_overlap"])
    eigen=np.sort(np.linalg.eigvals(o).real)
    bc=np.asarray(row["joint_histogram_BC"])
    spanning=minimum_spanning_tree(-bc).data
    diag=row["diagnostics"]
    boot=row.get("bootstrap",{})
    out=dict(N=row["N"],shape=row["shape"],cap=row["cap"],axial_free=row["axial_free"],
             status=row["status"],n_runs=row["n_runs"],production_frames=row["frames"],
             series=sorted({d["run"].rsplit("-q",1)[0] for d in diag}),
             trial_fields_eV=sorted({s[0] for s in row["bias_specs_mu_K_q0"]}),
             second_MBAR_eigenvalue=float(eigen[-2]) if len(eigen)>1 else None,
             joint_histogram_bin_width_x_Q=row.get("joint_histogram_bin_width_x_Q",[.025,.05]),
             weakest_joint_histogram_BC_on_max_spanning_tree=(float(-spanning.max())
                 if len(spanning)==len(o)-1 and len(spanning) else None),
             min_median_max_blocks_per_trajectory=np.percentile([d["blocks"] for d in diag],[0,50,100]).tolist(),
             bootstrap_seed=row.get("bootstrap_seed"),bootstrap_attempted=boot.get("count",0),bootstrap_resolved=boot.get("resolved",0),
             equal_height_separator=row.get("equal_height_separator"),
             primary=endpoints(fit),raw_equal_total_weight=endpoints(row.get("equal_total_weight")),
             minus_C46=reference_differences(row,reference))
    if fit:
        sample=draws(row)
        if sample is not None:
            out["conditional_95"]=intervals(sample,endpoints(fit))
        out.update(order_modes_at_primary=fit["basin_mode_q"],barriers_at_primary_kBT=fit["barrier_kBT"],
                   phase_volume_A3_atom=fit["phase_volumes_A3_atom"],
                   phase_c_a=fit["phase_observables"]["mean_c_a"],
                   phase_wall=fit["phase_observables"]["mean_wall"],
                   phase_ranked_sublattice_Al_fractions=np.array([fit["phase_observables"][
                       f"Al_fraction_sublattice_rank_{j}"] for j in range(1,5)]).T.tolist(),
                   phase_Kish_ess_frames=fit["phase_Kish_ess_frames"],
                   phase_Kish_approx_with_run_inefficiency=fit.get("phase_Kish_approx_with_run_inefficiency"),
                   phase_origin_fraction_by_mu=fit.get("phase_origin_fraction_by_mu"),
                   phase_fraction_near_volume_lower=fit["phase_observables"]["fraction_within_0p25_A3_lower_volume"],
                   phase_fraction_near_volume_upper=fit["phase_observables"]["fraction_within_0p25_A3_upper_volume"],
                   phase_fraction_near_axial_bounds=fit["phase_observables"]["fraction_within_0p01_axial_bounds"])
    specs=np.array(row["bias_specs_mu_K_q0"])
    cross={}
    for left,right in combinations(np.unique(specs[:,0]),2):
        li,ri=np.flatnonzero(specs[:,0]==left),np.flatnonzero(specs[:,0]==right)
        crossbc=bc[np.ix_(li,ri)]
        match=specs[li,2,None]==specs[ri,2]
        vals=crossbc[match]
        cross[f"{left:g} to {right:g}"]=dict(best_link=float(crossbc.max()),n_same_center=int(len(vals)),
            min_median_same_center=np.percentile(vals,[0,50]).tolist() if len(vals) else None)
    out["empirical_overlap_between_fields"]=cross
    if "field_cohort_split_eV" in row:
        out["field_cohort_split_eV"]=row["field_cohort_split_eV"]
    for label in ("by_initialization","by_field","by_field_cohort","production_halves","divider_sensitivity","bandwidth_sensitivity"):
        vals={key:endpoints(value) for key,value in row.get(label,{}).items()}
        out[label]=vals
        measured=[list(v.values()) for v in vals.values() if v]
        if len(measured)>=2:
            keys=next(v for v in vals.values() if v)
            out[label+"_span"]=dict(zip(keys,np.ptp(measured,axis=0)))
    out["by_field_support"]={key:{name:fit.get(name) for name in
        ("frames","n_bias_states","second_MBAR_eigenvalue","phase_Kish_ess_frames")} if fit else None
        for key,fit in row.get("by_field",{}).items()}
    return out


def summarize(rows,ref):
    result=dict(temperature_K=1000,pressure_GPa=0,model_sha256=ref["model_sha256"],
                chemical_field="configurational mu_Al minus mu_Ti, eV",
                phase_order=["alpha","alpha2"],
                composition="Al atomic percent; differences in percentage points",
                nested_cage_definition="Weights the subset W<0.35 of W<0.40 samples passing both volume guards. Independent simulations give the primary wall comparison; Kish counts here treat individual frames as independent.",
                conditional_95="2.5th and 97.5th bootstrap percentiles; contiguous blocks resampled within each independent trajectory",
                difference_intervals="Difference of independent empirical distributions of joint bootstrap samples",
                C46={"description":ref["reference"],"N":ref["N_sites"],"phase":ref["phase"],
                     "published_field_eV":ref["delta_mu_Al_minus_Ti_eV"],
                     "classical_config_field_start_eV":ref["cartesian_config_field_start_estimate_eV"],
                     "uncertainty":ref["uncertainty"]},
                groups={name:group(r,ref) for name,r in rows.items()},
                differences={})
    for kind in ("cage","axial_release","increasing_N"):
        pairs=[]
        for (ka,a),(kb,b) in combinations(rows.items(),2):
            if kind=="cage":
                retain=(a["N"],a["shape"],a["axial_free"])==(b["N"],b["shape"],b["axial_free"]) and a["cap"]!=b["cap"]
                comparison="cap"
            elif kind=="axial_release":
                retain=(a["N"],a["shape"],a["cap"])==(b["N"],b["shape"],b["cap"]) and a["axial_free"]!=b["axial_free"]
                comparison="axial_free"
            else:
                retain=(a["cap"],a["axial_free"])==(b["cap"],b["axial_free"]) and a["N"]!=b["N"]
                comparison="N"
                if retain:
                    sizes=sorted({r["N"] for r in rows.values() if (r["cap"],r["axial_free"])==(a["cap"],a["axial_free"])})
                    retain=abs(sizes.index(a["N"])-sizes.index(b["N"]))==1
            if retain:
                if a[comparison]>b[comparison]:
                    a,b,ka,kb=b,a,kb,ka
                pairs.append(dict(first=ka,second=kb,difference=difference(a,b)))
        result["differences"][kind]=pairs
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyses",type=Path,nargs="+")
    parser.add_argument("--nested",type=Path,nargs="*",default=[])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,default=Path(__file__).with_name("c46_reference.json"))
    args=parser.parse_args()
    rows={}
    checksums={}
    for path in args.analyses:
        blob=path.read_bytes()
        data=json.loads(blob)
        if rows.keys() & data.keys():
            raise ValueError("Duplicate ensemble summary: "+str(path))
        rows.update(data)
        checksums[path.name]=hashlib.sha256(blob).hexdigest()
    result=summarize(rows,json.loads(args.reference.read_text()))
    for path in args.nested:
        blob=path.read_bytes()
        checksums[path.name]=hashlib.sha256(blob).hexdigest()
        for key,check in json.loads(blob).items():
            if key not in result["groups"]:
                raise ValueError("Nested-cage check has no matching ensemble: "+key)
            if check.get("status")=="unresolved_source":
                result["groups"][key]["nested_cage_check"]=check
                continue
            selection=check["selection_by_run"]
            result["groups"][key]["nested_cage_check"]={
                name:value for name,value in check.items()
                if name in ("source_cap","target_wall","volume_guard_cap","source_mu_eV","source_order_cut",
                            "fractions_alpha_alpha2","phase_Kish_ess_retained_frames",
                            "delta_semigrand_alpha2_minus_alpha_eV_cell","linearized_delta_mu_eV")}
            sparse=min(check["phase_Kish_ess_retained_frames"])<10
            result["groups"][key]["nested_cage_check"].update(
                kept_frames=sum(r["kept"] for r in selection),total_frames=sum(r["total"] for r in selection),
                fewer_than_10_weighted_frames_in_a_phase=sparse,
                conditioned=endpoints(check.get("conditioned_fit")) if not sparse else None)
    result["source_sha256"]=checksums
    result["C46_reference_sha256"]=hashlib.sha256(args.reference.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
