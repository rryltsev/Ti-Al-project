"""Record the N=15 review from immutable summaries and inspected plots."""
import hashlib
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
batches = {n: read(ROOT / f"collected/analysis/batch_{n:04d}/summary.json") for n in (5, 10, 15)}
points = {n: {p["id"]: p for p in s["points"]} for n, s in batches.items()}
baseline = {round(1 - b["c_Ti"], 2): b for b in read(ROOT / "baseline_5k.json")}
latest = read(ROOT / "collected/analysis/latest.json")
assert {k: v for k, v in latest.items() if k != "plot"} == {k: v for k, v in batches[15].items() if k != "plot"}
rows = []
for p in batches[15]["points"]:
    sid = p["id"]
    assert [s["replica"] for s in p["sources"]] == list(range(1, 16))
    assert p["sources"][:10] == points[10][sid]["sources"]
    assert p["sources"][:5] == points[5][sid]["sources"]
    base = baseline[p["x_Al"]]
    assert base["n_replicas"] == 50 and base["n_steps"] == 20001
    wrows = {}
    for window, result in p["windows"].items():
        history = {n: points[n][sid]["windows"][window] for n in batches}
        eta = result["eta_mPa_s"]
        wrows[window] = {
            "history": history,
            "N15_change_percent": {
                "vs_N5": 100 * (eta / history[5]["eta_mPa_s"] - 1),
                "vs_N10": 100 * (eta / history[10]["eta_mPa_s"] - 1),
                "vs_5k_20ps": 100 * (eta / base["eta_mPa_s"] - 1),
            },
            "N15_relative_error_percent": 100 * result["relative_std"],
            "N15_change_vs_40ps_percent": 100 * (eta / p["windows"]["40ps"]["eta_mPa_s"] - 1),
        }
    new, old = p["new_batch_40ps"], p["previous_replicas_40ps"]
    z = abs(new["eta_mPa_s"] - old["eta_mPa_s"]) / math.hypot(new["eta_std_mPa_s"], old["eta_std_mPa_s"])
    assert math.isclose(z, p["new_batch_z"], rel_tol=1e-12)
    checks = next(c for c in batches[15]["checks"] if c["id"] == sid)
    values = [points[n][sid]["windows"]["40ps"]["eta_mPa_s"] for n in (5, 10, 15)]
    assert math.isclose((max(values)-min(values))/values[-1], checks["last_three_relative_range"], rel_tol=1e-12)
    rows.append({"id": sid, "baseline_5k_20ps": {k: base[k] for k in ("eta_mPa_s", "eta_std_mPa_s", "n_replicas", "n_steps")},
                 "windows": wrows, "new_replicas_11_15_40ps": new, "previous_replicas_1_10_40ps": old,
                 "new_batch_z": z, "new_batch_change_percent": 100*(new["eta_mPa_s"]/old["eta_mPa_s"]-1),
                 "checks": checks, "failed_criteria": [k for k, v in checks["flags"].items() if not v]})

plots = [ROOT / "collected/analysis/batch_0015/convergence.png"]
plots += [ROOT / f"collected/analysis/batch_0015/{sid}/40ps/{name}.png"
          for sid in ("xAl_0.20", "xAl_0.45") for name in ("spectrum", "extras")]
plots += [ROOT / f"collected/analysis/batch_{n:04d}/xAl_0.80/{window}/{name}.png"
          for n in (5, 10, 15) for window in ("40ps", "first20ps", "second20ps") for name in ("spectrum", "extras")]
assert len(plots) == 23
sources = [ROOT / f"collected/analysis/batch_{n:04d}/summary.json" for n in (5, 10, 15)]
sources += [ROOT / f for f in ("baseline_5k.json", "protocol.json", "collection.json", "scheduler-latest.txt", "status.json")]
findings = {
    "convergence": "All three cumulative 40ps estimates change by less than 1.53% from N10; errors shrink. No composition passes all stopping criteria.",
    "xAl_0.20_40ps": "Cutoff weight near 1.5 THz; broad viscosity plateau around 4.0 mPa s. Cost and cutoff pass. Growth slows but the N5,N10,N15 range remains 12.28%.",
    "xAl_0.45_40ps": "Cutoff weight near 1.3 THz; broad low-frequency viscosity plateau. Cost and cutoff pass. neff=52.13 fails 60; first20ps is 4.30% below full40ps.",
    "xAl_0.80_40ps": "N5/N10/N15 cutoff shifts roughly 2.4/1.9/1.7 THz, with a persistent low-frequency viscosity plateau near 1.8. N15 cutoff z=1.257 now passes.",
    "xAl_0.80_first20ps": "The secondary cutoff-weight peak visible at N10 around 4 THz disappears at N15; dominant weight is now around 1.8 THz. N15 cutoff z=0.703 passes.",
    "xAl_0.80_second20ps": "The pathological N5 selection near 9 THz and unstable fit uncertainty are absent at N10 and N15. N15 weight remains near 2-2.5 THz with a low-frequency shoulder. Cutoff z decreases 11.096 -> 2.449 -> 2.036 but still fails 2.",
    "xAl_0.80_new_only": "Numerical diagnostics only (the pipeline does not save new-only plots): replicas11-15 eta=1.5153+/-0.0873, cutoff z=8.118, new-vs-prior z=2.699. The low estimate is unreliable for a physical-drift conclusion. Prior new-only replicas6-10 cutoff z=3.479."
}
assessment = {
    "batch": 15, "review_unix": time.time(), "job_id": "20962153", "scheduler_state": "RUNNING",
    "collection": read(ROOT / "collection.json"), "progress_at_collection": {"composition": "xAl_0.20", "replica": 16, "phase": "production"},
    "preflight": "PASS", "source_checks": {"15_unique_replicas_each": True, "prior_5_and_10_sources_unchanged": True, "latest_matches_summary_except_plot": True},
    "source_manifest": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in sources],
    "rows": rows, "visual_review": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in plots],
    "visual_findings": findings,
    "cutoff_conclusion": "Weakened in cumulative data but persists in second20ps; strongly present in new-only11-15. Not eliminated; do not interpret the suspect estimate as physical drift.",
    "interpretation": ["Cumulative estimates and 20/40ps windows are correlated.", "5k is a separate 50-replica,20ps reference and was not pooled.", "Agreement with 5k is not a stopping criterion; preparation and duration also differ.", "Stopping is a heuristic; sampling errors exclude model and other systematic errors.", "Operational cutoff/cost/neff flags apply to full40ps; second20ps cutoff failure is additionally reported."],
    "action": "Continue the same job and unchanged physical protocol; monitor N20. No submission/requeue/cluster mutation performed.",
    "stop_recommended": False, "pass_streak": 0,
}
(HERE / "assessment.json").write_text(json.dumps(assessment, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
lines = ["PBE 15k, 1873 K: обзор N=15", "", "Вязкость и ошибка STACIE 1 sigma в мПа с; N — число реплик каждого состава."]
labels = {"40ps": "40 ps", "first20ps": "0–20 ps", "second20ps": "20–40 ps"}
for row in rows:
    b = row["baseline_5k_20ps"]
    lines += ["", row["id"], f"5k: {b['eta_mPa_s']:.6f} +/- {b['eta_std_mPa_s']:.6f}; 50 реплик, 20 ps"]
    for window, r in row["windows"].items():
        lines.append(labels[window]+": "+"; ".join(f"N={n}: {v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f}" for n,v in r["history"].items()))
        d = r["N15_change_percent"]
        lines.append(f"  N15 к N5/N10/5k: {d['vs_N5']:+.3f}% / {d['vs_N10']:+.3f}% / {d['vs_5k_20ps']:+.3f}%; ошибка {r['N15_relative_error_percent']:.3f}%")
    lines.append("Невыполненные критерии: "+", ".join(row["failed_criteria"]))
    lines.append(f"Размах N5,N10,N15: {100*row['checks']['last_three_relative_range']:.3f}%; максимальное расхождение 20/40 ps: {100*row['checks']['max_20_40_relative_difference']:.3f}%")
    new, old = row["new_replicas_11_15_40ps"], row["previous_replicas_1_10_40ps"]
    lines.append(f"Новые 11–15: {new['eta_mPa_s']:.6f} +/- {new['eta_std_mPa_s']:.6f}; прежние 1–10: {old['eta_mPa_s']:.6f} +/- {old['eta_std_mPa_s']:.6f}; изменение {row['new_batch_change_percent']:+.3f}%; z={row['new_batch_z']:.6f}")
lines += ["", "xAl=0.80: criterion_z, порядок окон 40 / 0–20 / 20–40 ps"]
for n in (5, 10, 15):
    lines.append(f"N={n}: "+" / ".join(f"{points[n]['xAl_0.80']['windows'][w]['criterion_z']:.6f}" for w in labels))
lines += ["", "Cutoff-проблема в накопленных данных ослабла, но сохранилась во вторых 20 ps (z=2.036). В новых 11–15 репликах z=8.118: их низкая оценка не доказывает физический дрейф.",
          "Для xAl=0.20 рост N10→15 +1.524% после +12.291% на N5→10. Окна отклоняются от 40 ps на −0.739% и +1.678%. Новые 11–15 согласуются с прежними: z=0.482<2.5. Полной сходимости пока нет.",
          "Проверены 23 графика: convergence N15; spectrum/extras для 40 ps составов 0.20 и 0.45 при N15; spectrum/extras всех трёх окон состава 0.80 при N5,N10,N15.",
          "Накопленные оценки и окна 20/40 ps коррелированы. Старые 5k (50 реплик, 20 ps) не объединялись с новыми; совпадение с ними не является критерием остановки.",
          "Условия нужны на двух последовательных проверках; pass_streak=0. Правило эвристическое и не гарантирует отсутствия систематических ошибок.",
          "20962153 RUNNING, одна A100; начат следующий пакет (xAl=0.20, реплика16). Протокол не изменён."]
(HERE / "report_ru.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
print(json.dumps({"review": str(HERE / 'assessment.json'), "images_reviewed": len(plots), "source_checks": assessment['source_checks']}, ensure_ascii=True))
