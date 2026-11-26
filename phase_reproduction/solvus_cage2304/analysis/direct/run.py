"""Update direct estimates for completed temperature/cage groups. Supports interrupted runs."""
import argparse
import csv
from pathlib import Path
import time

from isotherms import HERE,STUDY,PRIMARY,DisconnectedWindows,atomic_json,discover,kinetic_record,load,load_json,manual


def combine_csv(namespace):
    rows=[]
    for path in sorted(HERE.glob(namespace+"_T*_W*.json")):
        data=load_json(path)
        if "fixed_cut" not in data:continue
        for key in (str(PRIMARY),".42",".47",".5"):
            cut=str(float(key)); model=data["fixed_cut"][cut]
            a=model["maximum"]
            bt=data.get("bootstrap",{}).get("cuts",{}).get(cut,{})
            ci=bt.get("apex",{}).get("interval_95")
            pci=(bt.get("apex_profile") or {}).get("interval_95")
            S=[f["score"] for f in model["sampled_field_scores"]]
            roots={r["branch"]:r for r in model["crossings"]}
            pair=a["profiles"]["0.04"]
            status=("both_contacts_inside" if len(roots)==2 else "one_contact_inside" if roots else
                    "single_direct_field" if len(S)==1 else "ordered_to_both_field_edges" if min(S[0],S[-1])>0 else
                    "positive_maximum_without_bracketed_contact" if a["score"]>0 else "no_equal_variant_contact_inside")
            row=dict(T_K=data["T_K"],W=data["cap"],N=data["N"],cut_Q=float(cut),n_windows=data["direct_runs"],
                     n_K0_starts=len(data["K0_independent"]),direct_central_status=status,
                     mu_min=data["direct_fields_eV"][0],mu_max=data["direct_fields_eV"][-1],
                     mu_at_max=a["mu_eV"],x_at_max_pct=100*a["x"],max_on_field_edge=model["maximum_on_edge"],
                     Smax=a["score"],S_low95=ci[4][0] if ci else None,S_high95=ci[4][1] if ci else None,
                     score_left=S[0],score_right=S[-1],bootstrap_fraction_positive_S=bt.get("fractional_positive_S"),
                     bootstrap_fraction_bracketing_both=bt.get("fractional_both_roots_inside"),
                     lower_barrier_kBT=pair["barrier"] if pair else None,Q_low_mode=pair["modes_Q"][0] if pair else None,
                     lower_barrier_low95=pci[3][0] if pci else None,lower_barrier_high95=pci[3][1] if pci else None,
                     Q_high_mode=pair["modes_Q"][1] if pair else None,delta_h_meV_atom=a["delta_h_meV"],
                     ESS_apex_low=a["conditional"][0]["time_ESS"],ESS_apex_high=a["conditional"][1]["time_ESS"])
            for b in ("Ti","Al"):
                r=roots.get(b)
                bl=(bt.get("branches",{}).get(b) or {}).get("interval_95")
                rp=r["profiles"]["0.04"] if r else None
                rstat=bt.get("root_profiles",{}).get(b) or {}
                row.update({b+"_mu":r["mu_eV"] if r else None,b+"_x_alpha_pct":100*r["x_low"] if r else None,
                            b+"_x_alpha2_pct":100*r["x_high"] if r else None,b+"_gap_pp":100*r["gap"] if r else None,
                            b+"_gap_low95_pp":100*bl[3][0] if bl else None,b+"_gap_high95_pp":100*bl[3][1] if bl else None,
                            b+"_mu_low95":bl[0][0] if bl else None,b+"_mu_high95":bl[0][1] if bl else None,
                            b+"_x_alpha_low95_pct":100*bl[1][0] if bl else None,b+"_x_alpha_high95_pct":100*bl[1][1] if bl else None,
                            b+"_x_alpha2_low95_pct":100*bl[2][0] if bl else None,b+"_x_alpha2_high95_pct":100*bl[2][1] if bl else None,
                            b+"_two_modes":bool(rp) if r else None,b+"_Q_mode_low":rp["modes_Q"][0] if rp else None,
                            b+"_Q_mode_high":rp["modes_Q"][1] if rp else None,b+"_B_from_lower_peak":rp["barrier"] if rp else None,
                            b+"_bootstrap_fraction_root_and_two_modes":rstat.get("count",0)/len(bt["replicates"]) if bt else None,
                            b+"_ESS_low":r["conditional"][0]["time_ESS"] if r else None,
                            b+"_ESS_high":r["conditional"][1]["time_ESS"] if r else None})
            rows.append(row)
    if rows:
        out=HERE/(namespace+"_summary.csv");tmp=out.with_suffix(".csv.tmp")
        with tmp.open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(out)


def update(base,mode,bootstrap,selection=None):
    groups,pending=discover(base)
    planned={}
    bias_paths=set()
    for path in base.glob("*/windows.csv"):
        with path.open(encoding="utf-8-sig") as f:
            all_rows=list(csv.DictReader(f));r=all_rows[0]
            planned.setdefault((float(r["T_K"]),float(r["cap"])),set()).add(path)
            if any(float(v["K_eV"])>0 for v in all_rows):
                bias_paths.add(path)
    completed=0
    for t,cap in sorted(planned,key=lambda key:(key[0]!=1243,key[1],key[0])):
        if selection and (t,cap)!=selection:continue
        manifests=groups.get((t,cap),[])
        complete=bool(manifests) and set(manifests)==planned[t,cap]
        primary_complete=bool(set(manifests)&bias_paths) and (planned[t,cap]&bias_paths)<=set(manifests)
        if mode=="final" and not primary_complete:continue
        if not manifests:continue
        name=f"{mode}_T{t:g}_W{cap:.2f}".replace(".","p")+".json"
        out=HERE/name
        paths=sorted(manifests)
        prev=load_json(out) if out.is_file() else None
        if prev and mode=="final" and prev.get("bootstrap",{}).get("count")==bootstrap:
            oldpaths={Path(s) for s in prev["manifests"]}
            newpaths=set(paths)-oldpaths
            if newpaths and oldpaths<=set(paths) and not newpaths&bias_paths:
                for tt,nn,_,ww,_,rr in load(sorted(newpaths),0):
                    if tt!=t or ww!=cap:raise ValueError("Mismatched K0 control")
                    prev["K0_independent"].extend(kinetic_record(nn,x) for x in rr if x["spec"][1]==0)
                prev["manifests"]=[str(m.resolve()) for m in paths]
                prev["all_expected_series_ready"]=complete
                atomic_json(out,prev)
        if not(prev and prev["manifests"]==[str(p.resolve()) for p in paths] and
               (not bootstrap or prev.get("bootstrap",{}).get("count")==bootstrap)):
            try:
                d=manual(paths,out,bootstrap,mode=="final",341247)
            except DisconnectedWindows as exc:
                if mode=="final":raise
                print(t,cap,str(exc),flush=True)
                continue
            d["all_expected_series_ready"]=complete
            atomic_json(out,d)
            a=d.get("fixed_cut",{}).get(str(PRIMARY))
            if a:
                print(name,"direct fields",d["direct_fields_eV"],"Smax",round(a["maximum"]["score"],3),
                      "edge log odds",[round(f["score"],3) for f in (a["sampled_field_scores"][0],a["sampled_field_scores"][-1])],
                      "inside roots",[(r["branch"],round(r["mu_eV"],5)) for r in a["crossings"]],flush=True)
        if complete:completed+=1
    combine_csv(mode)
    finals=list(HERE.glob("final_T[0-9]*.json"))
    zeros=HERE/"direct_temperature_zeros.json"
    if mode=="final" and finals and (not zeros.is_file() or max(f.stat().st_mtime for f in finals)>zeros.stat().st_mtime):
        from temperature_zeros import generate
        from matched_cages import compare
        from bracket_checks import assess
        from free_chain_check import compare as compare_free_starts
        generate()
        compare()
        assess()
        compare_free_starts()
    return completed,len(planned),pending


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["scout","final"],default="final")
    p.add_argument("--bootstrap",type=int,default=64)
    p.add_argument("--watch",action="store_true")
    p.add_argument("--require-all",action="store_true")
    p.add_argument("--group",nargs=2,type=float,metavar=("T","W"))
    args=p.parse_args()
    while True:
        count,planned,pending=update(STUDY/"atomistic"/"series",args.mode,args.bootstrap if args.mode=="final" else 0,
                                     tuple(args.group) if args.group else None)
        done=count==1 if args.group else count>=8 and count==planned
        if done or not args.watch:break
        time.sleep(30)
    print("complete groups",count,"of",planned,"pending series",len(pending),flush=True)
    raise SystemExit(2 if args.require_all and not done else 0)
