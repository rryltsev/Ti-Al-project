"""Two-mask finite differences and digitization stress tests of the 1000 K MFA tangent."""
import json
from pathlib import Path
import sys

import numpy as np


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from fit_sensitivity import alpha2, audit_tangent, contact, derivative, read_csv, save, sublattice_weights


def main():
    src=read_csv(HERE/"coefficients_1000K_vref.csv")
    def values(key):
        return .001*np.array([float(row[key]) if row[key] else 0. for row in src])
    j39=values("new_static39_meV")
    j37=values("new_static37_meV")
    vib=values("new_vib_meV")
    w=sublattice_weights()
    records,attributions,directions=[],[],[]
    def solve(family,case,j,seed=(.08,.8,.002)):
        j=j.copy()
        j[:2]=0.
        solution=contact(w,j,seed)
        audit_tangent(w,j,solution)
        p,jac,_,_=solution
        pair=np.array([100*p[0],alpha2(p)])
        records.append(dict(family=family,case=case,alpha_at_pct=pair[0],alpha2_at_pct=pair[1],
                       q_alpha2=p[1],r_alpha2=p[2]))
        return pair,p,derivative(w,p,jac)

    # Inclusion-exclusion of precisely the two rows with imaginary internal modes.
    for family,offset in [("new_static",np.zeros(20)),("new_HA_vib37_fixed",vib)]:
        base,seed,_=solve(family,"39",j39+offset)
        minus26=solve(family,"38_omit26",j39+values("static_shift_omit_train_26_only_meV")+offset,seed)[0]
        minus30=solve(family,"38_omit30",j39+values("static_shift_omit_train_30_only_meV")+offset,seed)[0]
        both=solve(family,"37_omit26_and30",j37+offset,seed)[0]
        interaction=both-minus26-minus30+base
        shap26=minus26-base+.5*interaction
        shap30=minus30-base+.5*interaction
        assert np.allclose(shap26+shap30,both-base,atol=1e-10,rtol=0)
        for i,branch in enumerate(["alpha","alpha2"]):
            attributions.append(dict(family=family,branch=branch,
                 delta_omit26_alone=(minus26-base)[i],delta_omit30_alone=(minus30-base)[i],
                 second_order_interaction=interaction[i],delta_omit_both=(both-base)[i],
                 shapley_26=shap26[i],shapley_30=shap30[i]))

    paper=values("paper_static_meV")
    red=values("paper_vib_digitized_meV")
    blue=values("paper_Fig2_static_digitized_meV")
    half=values("paper_S1_half_last_digit_meV")
    pixel=values("paper_Fig2_one_pixel_meV")
    # The two affine coordinates of the free energy drop out of fixed-volume contacts.
    half[:2]=0.
    for family,baseline,boxes in [
        ("paper_static",paper,[("S1_half_last_digit",half)]),
        ("paper_HA",paper+red,[("S1_half_last_digit",half),("red_one_pixel",pixel),("both",half+pixel)])]:
        nominal,seed,grad=solve(family,"nominal",baseline)
        grad[:2]=0.
        for box,amp in boxes:
            delta=np.sign(grad)*amp
            for sign,word in [(-1,"low"),(1,"high")]:
                label=box+"_"+word
                estimate=sign*float(grad@delta)
                point,_,_=solve(family,label,baseline+sign*delta,seed)
                directions.append(dict(family=family,experiment=label,
                            delta_alpha_at_pct=point[0]-nominal[0],delta_alpha2_at_pct=point[1]-nominal[1],
                            linearized_delta_alpha2_at_pct=estimate))
        alt=blue+(red if family=="paper_HA" else 0)
        point,_,_=solve(family,"blue_static_in_place_of_S1",alt,seed)
        directions.append(dict(family=family,experiment="blue_static_in_place_of_S1",
                    delta_alpha_at_pct=point[0]-nominal[0],delta_alpha2_at_pct=point[1]-nominal[1],
                    linearized_delta_alpha2_at_pct=float(grad@(alt-baseline))))
    save(records,HERE/"mfa_contact_stress_1000K.csv")
    save(attributions,HERE/"mask_contact_decomposition.csv")
    save(directions,HERE/"print_precision_corners.csv")
    print(json.dumps(attributions,indent=2))


if __name__=="__main__":
    main()
