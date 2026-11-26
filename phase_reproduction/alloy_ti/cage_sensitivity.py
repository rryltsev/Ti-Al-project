"""Reweight saved displacements into nested scaled Voronoi cells at matched classical HA and MTP."""
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
import re
import sys
import tarfile
import numpy as np
from scipy.spatial import Voronoi
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent/"cluster/runs"
CAPS = (.35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, 1.)
KT = 8.617333262145e-5*1000.
RAW_KEYS = ("delta_eV_cell","log_harmonic_to_proposal","max_wall_fraction")


def npz_data(file):
    with np.load(file,allow_pickle=False) as src:
        return [src[key] for key in RAW_KEYS]


def archive_data(path, prefix):
    with tarfile.open(path) as archive:
        return {Path(item.name).stem.removeprefix(prefix):npz_data(BytesIO(archive.extractfile(item).read()))
                for item in archive.getmembers() if item.isfile() and item.name.endswith(".npz")}


def se(deleted):
    return float(np.sqrt((len(deleted)-1)*np.var(deleted)))


def free(logw):
    blocks, size = logw.shape
    logz = logsumexp(logw, axis=1)
    total = logsumexp(logz)
    deleted = np.array([logsumexp(np.delete(logz,i)) for i in range(blocks)])
    if not np.isfinite(deleted).all():
        raise ValueError("Empty domain in a delete-one sample")
    return np.r_[-KT*(total-np.log(blocks*size)), -KT*(deleted-np.log((blocks-1)*size))]


def measure(raw, cap):
    delta, rho, wall = raw
    h = np.where(wall < cap, rho, -np.inf)
    w = h-delta/KT
    lognorm = logsumexp(w)
    within = np.isfinite(w)
    h_corr = free(h)
    a_corr = free(w)
    correction = a_corr-h_corr
    diag = dict(target_ess=float(np.exp(2*lognorm-logsumexp(2*w))),
                largest_weight=float(np.exp(w.max()-lognorm)), accepted_draws=int(within.sum()),
                reference_probability=float(np.exp(-h_corr[0]/KT)), 
                target_free_rel_global_ha_eV_cell=float(a_corr[0]),
                matched_anharmonic_shift_eV_cell=float(correction[0]),
                matched_shift_SE_eV_cell=se(correction[1:]),
                minimum_sampled_delta_eV_cell=float(delta[within].min()))
    return h_corr, a_corr, diag


def report(values,scale=1.):
    return dict(value=float(values[0]*scale),se=se(values[1:])*abs(scale))


def compare_proposals(gaussian, broad):
    h0, e0, d0 = gaussian
    h1, e1, d1 = broad
    a, b = report(e0-h0,125), report(e1-h1,125)
    return dict(gaussian_matched_shift_meV_atom=a,student_matched_shift_meV_atom=b,
                student_minus_gaussian_meV_atom=b["value"]-a["value"],
                difference_se_meV_atom=float(np.hypot(a["se"], b["se"])),
                gaussian_ess=d0["target_ess"],student_ess=d1["target_ess"],
                gaussian_largest_weight=d0["largest_weight"],student_largest_weight=d1["largest_weight"])


def combine(logd, energies):
    return -KT*logsumexp(np.asarray(logd)[:,None]-np.asarray(energies)/KT,axis=0)


def cell_geometry():
    sys.path.insert(0,str(HERE.parent/"atomistic"))
    from hcp_orbits import hcp_geometry
    cell,sites=hcp_geometry((2,2,2))
    shifts=np.array(list(product(range(-1,2),repeat=3)))@cell
    xyz=(shifts[:,None,:]+sites).reshape(-1,3)
    voronoi=Voronoi(xyz)
    distances=[]
    for site in sites[:2]:
        ref=np.argmin(np.linalg.norm(xyz-site,axis=1))
        touching=voronoi.ridge_points[np.any(voronoi.ridge_points==ref,axis=1)]
        radii=np.linalg.norm(xyz[touching[touching!=ref]]-xyz[ref],axis=1)
        if len(radii)!=12 or radii.max()>=3.5:
            raise ValueError("Periodic Voronoi facets differ from pilot.site_planes")
        distances.extend(radii)
    minimum=float(min(distances))
    model=HERE.parent/"inputs/fitted.mtp"
    rb_min=float(re.search(r"^\s*min_dist\s*=\s*(\S+)",model.read_text(),re.M)[1])
    return dict(parent_nearest_neighbor_A=minimum,mtp_radial_basis_min_dist_A=rb_min,
                primary_lower_bound_A=.6*minimum,
                distance_bound="For parent-site separation d, both scaled Voronoi inequalities along the line of centers imply r_ij >= (1-cap)*d, hence r_min >= (1-cap)*d_parent_min.",
                interpretation="At cap 0.40 the geometric lower bound exceeds the fitted radial basis min_dist. This input parameter is not an experimental or certified validity boundary.")


def n8_observables(data, base, degeneracy):
    names = list(data)
    hc, an = ({k:base[k]+data[k][i] for k in names} for i in (0,1))
    htot, atot = (combine([np.log(degeneracy[k]) for k in names],[energy[k] for k in names])
                  for energy in (hc,an))
    gaps = {}
    for name in names:
        harmonic = hc[name]-hc["d019"]
        direct = an[name]-an["d019"]
        gaps[name]=dict(harmonic_meV_atom=report(harmonic,125),target_meV_atom=report(direct,125),
                        target_minus_matched_harmonic_meV_atom=report(direct-harmonic,125))
    prob = np.exp(np.log(degeneracy["d019"])-(an["d019"]-atot)/KT)
    return dict(gaps_relative_to_d019=gaps,canonical_anharmonic_shift_meV_atom=report(atot-htot,125),
                p_d019=report(prob),per_orbit={k:v[2] for k,v in data.items()})


def n16_observables(data, rows):
    groups = {}
    hf, af = [], []
    for k in (2,3,4):
        group = [r for r in rows if r["N_Al"]==k]
        logd = np.log([r["degeneracy"] for r in group])
        harmonic, target = (np.array([r["harmonic_config_total_eV_cell"]+data[r["id"]][i] for r in group])
                            for i in (0,1))
        hg, ag = combine(logd,harmonic), combine(logd,target)
        hf.append(hg)
        af.append(ag)
        prob=np.exp(logd-(target[:,0]-ag[0])/KT)
        ess=np.array([data[r["id"]][2]["target_ess"] for r in group])
        maxw=np.array([data[r["id"]][2]["largest_weight"] for r in group])
        groups[k]=dict(harmonic_config_eV_cell=report(hg),target_config_eV_cell=report(ag),
                       target_minus_matched_harmonic_meV_atom=report(ag-hg,1000/16),
                       joint_ess=float(1/np.sum(prob**2/ess)),worst_ess=float(ess.min()),
                       largest_joint_weight=float((prob*maxw).max()),
                       orbit_issues={r["id"]:dict(ess=float(e),largest_weight=float(w))
                                    for r,e,w in zip(group,ess,maxw) if e<50 or w>.1})
    second = np.array([1.,-2.,1.])
    h2,a2 = second@np.array(hf),second@np.array(af)
    centre_scale=-1000/(2*16)
    return dict(compositions=groups,
                middle_excess_harmonic_meV_atom=report(h2,centre_scale),
                middle_excess_target_meV_atom=report(a2,centre_scale),
                middle_excess_anharmonic_shift_meV_atom=report(a2-h2,centre_scale),
                central_second_difference_harmonic_eV=report(h2),
                central_second_difference_target_eV=report(a2),
                central_second_difference_anharmonic_shift_eV=report(a2-h2))


def independent_n16(archive_path, rows, gaussian_data):
    # Only W < 0.40 was evaluated by the independent job. Its stored delta
    # outside the cage is a placeholder, never use it to evaluate a larger cage.
    student={}
    with tarfile.open(archive_path) as archive:
        for item in archive:
            if item.name.startswith("cage_k"):
                with np.load(BytesIO(archive.extractfile(item).read())) as packed:
                    if float(packed["cap"]) != .4:
                        raise ValueError("Unexpected cap in independent N16 data")
                    student[Path(item.name).stem.removeprefix("cage_")]=packed["data"].transpose(1,0,2)
    if set(student)!=set(gaussian_data):
        raise ValueError("Independent N16 dataset missing representatives")
    result={}
    for cap in (.35,.40):
        first={k:measure(draws,cap) for k,draws in gaussian_data.items()}
        second={k:measure(draws,cap) for k,draws in student.items()}
        decomposition={}
        for k,coef in ((2,1),(3,-2),(4,1)):
            group=[r for r in rows if r["N_Al"]==k]
            logd=np.log([r["degeneracy"] for r in group])
            def logpart(measured):
                return logd-np.array([r["harmonic_config_total_eV_cell"]+measured[r["id"]][1][0]
                                      for r in group])/KT
            g,t=logpart(first),logpart(second)
            logr=logsumexp(t)-logsumexp(g)
            factor=1. if logr==0 else logr/np.expm1(logr)
            # Sum of orbit contributions exactly recovers the difference of
            # the two log partition sums, including the chemical multiplicities.
            parts=-1000*coef*KT*factor*(np.exp(t-logsumexp(g))-np.exp(g-logsumexp(g)))
            for row,part in zip(group,parts):
                name=row["id"]
                delta,rho,wall=student[name]
                lw=np.where(wall<cap,rho-delta/KT,-np.inf)
                weight=np.exp(lw-logsumexp(lw))
                imax=np.unravel_index(weight.argmax(),weight.shape)
                decomposition[name]=dict(student_minus_gaussian_contribution_meV_cell=float(part),
                    student_ess=second[name][2]["target_ess"],gaussian_ess=first[name][2]["target_ess"],
                    student_largest_weight=second[name][2]["largest_weight"],
                    student_block_weight=weight.sum(axis=1).tolist(),
                    largest_student_weight_wall=float(wall[imax]),
                    largest_student_weight_excess_eV_cell=float(delta[imax]))
        result[f"{cap:.2f}"]=dict(student=n16_observables(second,rows),
                                  orbit_contributions=decomposition)
    return result


def main():
    def output(name):
        return RUNS/f"alloy-ti-{name}-01"/"outputs"
    n8_dir=output("n8")
    ref_dir=output("n16ref")
    inputs=[n8_dir/"draws.tar.gz",output("tail")/"tail_samples.npz",
            output("n16ti")/"n16_draws.tar.gz",ref_dir/"n16_refined_draws.tar.gz",
            n8_dir/"pilot.json",ref_dir/"n16_refined.json",HERE/"c46_N8_1000K.json"]
    original=archive_data(inputs[0],"samples_")
    broad=npz_data(inputs[1])
    n16=archive_data(inputs[2],"raw_")
    refined=archive_data(inputs[3],"raw_")
    n8_rows=json.loads(inputs[4].read_text())["results"]
    n16_run=json.loads(inputs[5].read_text())
    if set(refined)!=set(n16_run["refinement"]["replacements"]):
        raise ValueError("Missing refinement draws")
    n16.update(refined)
    if len(n16)!=43 or set(n16)!={r["id"] for r in n16_run["results"]}:
        raise ValueError("Incomplete N16 orbit data")
    deg={k:r["degeneracy"] for k,r in json.loads(inputs[6].read_text())["orbits"].items()}
    base={r["name"]:r["static_eV_cell"]+r["harmonic_classical_eV_cell"] for r in n8_rows}
    result={}
    for cap in CAPS:
        small={k:measure(draws,cap) for k,draws in original.items()}
        student=measure(broad,cap)
        alt={**small,"interlayer":student}
        large={k:measure(draws,cap) for k,draws in n16.items()}
        result[f"{cap:.2f}"]=dict(interlayer_independent_check=compare_proposals(small["interlayer"],student),
                                 N8_gaussian=n8_observables(small,base,deg),
                                 N8_with_student_interlayer=n8_observables(alt,base,deg),
                                 N16_gaussian=n16_observables(large,n16_run["results"]))
    independently_caged=RUNS/"alloy-ti-caged-01/outputs/caged_draws.tar.gz"
    independent_result=independent_n16(independently_caged,n16_run["results"],n16)
    inputs.append(independently_caged)
    output=dict(primary_cap=.4,temperature_K=1000., model_sha256=n16_run["model_sha256"],
                geometry=cell_geometry(),
                domain="max_i,j 2*(r_i-R_i).(R_j-R_i)/|R_j-R_i|^2 < cap for each parent-neighbor plane",
                estimator="Endpoint importance normalization relative to the unrestricted harmonic reference. In every anharmonic shift the matched harmonic normalization uses the identical cap and proposal samples.",
                scope="Fixed-cell, site-confined observables. A wide-domain Student check covers the N8 interlayer orbit. Before targeted resampling, the independent N16 Student check disagreed with the initial Gaussian curvature. See targeted_comparison.json for the final high-statistics estimates at caps 0.35 and 0.40. Extreme tails with ESS close to one do not resolve free energies.",
                units="N16 middle excess = [F3-(F2+F4)/2]*1000/16 meV/atom; cell F is canonical. SE is delete-one independent-group jackknife.",
                inputs_sha256={str(p.relative_to(HERE.parent)):sha256(p.read_bytes()).hexdigest() for p in inputs},
                caps=result,independent_N16=independent_result)
    (HERE/"cage_sensitivity.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
