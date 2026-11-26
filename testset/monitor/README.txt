TiAl Test50: cumulative paired validation, updated2026-10-29

EARLY STOP / MONITOR CLOSED (explicit user instruction, 2026-11-04)
The final validated common subset contains 49/50 structures (001..024 and
026..050). TiAl-test-025 r2SCAN array task 20957372_24 was cancelled while
PENDING, with 00:00:00 allocated elapsed time. The two-hour monitoring
automation tial-test50-parity-plots was deleted. No further scheduled collection
or analysis is planned. Existing results/n49-f94a64dc4d95bca4 and model
histories remain unchanged. This is an adaptive early stop after inspecting
Test50 results and the readiness heuristic; report the actual 49/50 coverage,
missing liquid 025, model checkpoint selection after earlier Test50 inspection,
and the crystal-first scheduling change in any article interpretation. The
user's scientific judgement was that the last structure would not materially
change the conclusions; it was not calculated, so this cannot be verified
empirically. See runs/early-stop-20261104/closure.json for the Slurm evidence.

ACTIVE PBE CHECKPOINT (explicit user instruction,2026-10-29)
Use checkpoint20000 (20k) from TiAl_tune_pbe_Neo_DPA4_shard_20260920_140047.
Source SHA256:3b0b4b69f8fca7be5bae4a5f24498aac904efad08f126f4a57de035e0379c67e.
The original model.ckpt-20000.pt was evaluated directly with CPU DeepPot,
without export, training or energy correction; stored train_infos.step19999.
Registry models.PBE.prediction_cache selects predictions/PBE-20000 (all50).
Reuse verified inference job20964816 and runs/pbe20k-comparison-20261029-01.
Model identity and derivative checks are in models/PBE-20000.
The activation and before-files are in runs/pbe20k-switch-20261029-01.

PBE15k history is preserved SEPARATELY in results/model-history/PBE-15000:
history.json holds counts23,24,25,28,29; latest.json and four PNG/PDF pairs
preserve the last15k analysis. Original nNN-fingerprint archives are unchanged.
Old predictions/PBE-15000 and models/PBE-15000 are unchanged, as are the
historical5k predictions/PBE and models/PBE. All250 preceding prediction hashes
were checked;300 cached files now include200 active and100 archival PBE files.
PBEsol, r2SCAN, pretrained, their models and predictions are unchanged.
Graphs identify PBE(20k). The20k convergence segment starts with the current
subset only; never join5k/15k histories or backfill old subsets as new checks.
Histories are comparable only with identical model and analysis hashes.
The combined append-only results/history.json preserves all versions, while
analyze.py filters each convergence/stability segment by exact model identity.

Both user-requested replacements5k->15k(2026-10-26) and15k->20k(2026-10-29)
occurred AFTER earlier Test50 inspection. Retain this provenance for the article;
do not claim blind checkpoint selection. No Test50 training or offset fitting.
Optional training-only alignment coefficients remain frozen and unchanged.
The original export documentation below describes historical5k PBE; this active
section and model_registry.json take precedence. The historical5k LAMMPS test
does not certify a20k LAMMPS export; direct original-PT inference supports metrics.

CURRENT MONITORING MODE (latest user instruction, 2026-10-22)
Default analyze.py uses raw energies only: blue fine-tuned, orange pretrained.
No aligned series or aligned metrics are produced by scheduled monitoring.
Previously frozen coefficients and aligned archives are retained for the article.
For that separate variant use analyze.py --with-alignment; it writes to
results/article-aligned and does not replace monitoring results.
The alignment documentation below describes this optional article variant.

PURPOSE
Refresh every two hours (resumed by explicit user instruction 2026-10-26), using only structure IDs with validated PBE, PBEsol
and r2SCAN DFT outputs. Compare each frozen article DPA4 model against its
own functional on exactly the same common subset. Do not change the models,
test geometries. Preserve raw energies; the separately reported alignment
was explicitly requested by the user and is fitted only to original train splits.

FROZEN MODELS AND PREDICTIONS
The user supplied PBE/PBEsol/r2SCAN checkpoints in the local project's
DPA4-checkpoints directory. Exact hashes are pinned in models/source-manifest.json.
CPU .pt2 exports are in models/<XC>/model_frozen.pt2. The authoritative remote
paths and export hashes are in model_registry.json (status=frozen).
All three passed checkpoint/export E/F/virial comparisons, finite-difference
force and virial checks, and LAMMPS run 0 using the validated native binary.
Fine-tuned predictions for ALL 50 fixed geometries (150 JSON files) are already cached.
Only the common completed DFT subset enters metrics. No new inference jobs
are normally needed: collect.py followed by analyze.py refreshes the figures.
See models/USAGE.txt for runtime constraints and the working LAMMPS setup.

PRETRAINED BASELINE (added 2026-10-22)
The exact article backbone, DPA4-Neo-OMat24-v20260805.pt, was downloaded from
https://huggingface.co/deepmodelingcommunity/DPA4-OMat24
at revision 173bfd5a26f90a756ee891777270ef117078d9be.
Upstream LFS SHA256 and downloaded bytes match:
fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989.
Model identity is pinned in pretrained_registry.json; upstream metadata,
configuration and model card are retained in models/pretrained/.
The article names this checkpoint. Architecture dimensions and the full element
map match the supplied fine-tuned checkpoint; source hashes of the original
article training input were not available for an independent bitwise audit.
Slurm 20958504 evaluated the original checkpoint directly via DeepPot on CPU;
all 50 geometries and force/virial derivative checks passed. No re-freezing,
bias adjustment, training or test-dependent model selection was performed.
One shared cache in predictions/pretrained is compared to each DFT functional,
using exactly the same common IDs as the corresponding fine-tuned model.
Initially there were 200 prediction files: 150 fine-tuned and 50 pretrained.
The 50 additional PBE-15000 files are versioned separately as described above.
All prediction files are covered by prediction-manifest.json.

In monitoring parity.png/pdf: blue = fine-tuned, orange = raw pretrained.
The optional article variant additionally uses green for aligned energies. Energy points for
pretrained are hollow rings so both models remain visible when they overlap.
When several structure families are present, energy marker shapes identify
liquid/hcp/bcc/gamma. Force points are colored by model. Per-family statistics
are saved separately. Both model roles have MAE/RMSE, species and composition
groups, structure-bootstrap intervals, raw energy residuals and history.
results/latest.json preserves metrics for fine-tuned models and adds
pretrained_metrics with the same functional/group schema. Fine-tuned metrics
and intervals were checked for exact equality to the archived six-structure
analysis. Seven scientific tests pass, including comparison of one shared
pretrained prediction to three distinct reference-energy datasets.

Raw absolute energy errors against different functionals include energy-reference
and functional differences. The upstream model card explicitly discusses
per-element energy reference differences between datasets. Such differences
must not be interpreted as purely configuration-dependent model errors.
No shifts are fitted to Test50. Foundation pretraining overlap remains unknown.
The stopping heuristic remains based on the fine-tuned models and coverage;
adding this baseline does not change its thresholds or authorize cancellation.

TRAINING-ONLY ENERGY ALIGNMENT (user-authorized 2026-10-22)
E_aligned = E_pretrained + N_Al*c_Al + N_Ti*c_Ti.
Fit separately for PBE/PBEsol/r2SCAN by unregularized rank-2 least squares
of per-atom residuals, equal weight per training frame. No Test50 or validation
energies enter the fit, and no outliers are removed. Original training_systems.json
lists give 850/849/845 frames. Validation splits (149/149/148 frames) are excluded.
The geometry audit found no train/validation or train/Test50 matches at 5e-5 A
(periodic images, atom permutations and common translations allowed; fixed cell axes).
2544 labelled rows map to 1777 exact distinct geometries evaluated once by the
unchanged pretrained checkpoint. Training labels were not sent to inference jobs.
Inputs, original source hashes and protocol fixed before training inference are
in alignment/train-offsets-v1; Slurm array 20958509 used 8 chunks, at most 4
concurrent tasks with 4 CPUs each. DFT schedules were not modified.
coefficients.json is frozen and pinned by alignment-registry.json. analyze.py
checks its hash, provenance, model identity and Test50 manifest before use.
Never refit during scheduled monitoring. The raw monitor does not load calibration.
For --with-alignment, missing or changed calibration is an error; never silently
fall back to an old figure. Raw prediction files
remain immutable; aligned energies are formed in memory. Forces/stresses are
identical before/after alignment. pretrained_aligned_metrics contains the same
groups, units and structure-bootstrap intervals as the other metric sections.
The extra energy_alignment.png/pdf focuses on fine-tuned versus aligned energies;
The article variant parity.png/pdf retains three energy series and its
convergence includes all variants; raw monitoring keeps only two model roles.
Intervals condition on fixed coefficients and do not include training-calibration
uncertainty. This variant was requested after raw Test50 inspection and must not
be described as a pre-registered blind comparison. Stopping rules remain based
on fine-tuned models only. Use test_alignment.py alongside test_analysis.py.

RUN AUDIT
20958499: GPU export cancelled while pending (zero allocated CPU/GPU time).
20958500: CPU exports succeeded; initial validation then rejected the legitimate
118-element foundation type map because it incorrectly expected only Al/Ti.
20958501: corrected validation reused those exact exports; all checks passed,
and all 150 predictions completed. No source weights were changed.
20958502: bundled LAMMPS executable SIGSEGV before output; no usable test result.
20958503: validated native LAMMPS build passed all three models on one fixed
mixed Al/Ti structure. This is a run-0 interface check, not a long MD test.
Manifests, code, submissions, hashes and retrieved results are retained in runs/.

LOCAL COMMANDS (PowerShell, from the TiAl workspace)
python testset/monitor/collect.py
testset/monitor/.venv/Scripts/python.exe testset/monitor/analyze.py
testset/monitor/.venv/Scripts/python.exe testset/monitor/test_analysis.py

collect.py uses the existing host-verified SSH wrapper in testset/dft.
It does not modify the cluster. It downloads compact labels, not POTCAR or
large VASP files. It checks label SHA256, POSCAR SHA256, unchanged cell and
geometry, SCF success, one ionic calculation, atom count and matching
KPOINTS/POTCAR hashes between all three functional labels. It preserves
immutable bundle snapshots in data/triplets-*.json and updates data/latest.json.
Always inspect scheduler/accounting as well as status.json. Nine historical
FAILED records (PBE 0..3; PBEsol 0..4) are the corrected GGA-validator failure;
their revalidated labels are valid. New scheduler errors must be reported.

INFERENCE CACHE
The complete cache is predictions/<XC>/<structure_id>.json. Its SHA256 hashes
are stored in prediction-manifest.json and checked by analyze.py. Model/POSCAR
hashes and atom ordering are checked as well. collect_outputs.py retrieved and
verified all model and prediction files against remote output manifests.
If recovery is needed, inspect the existing run first; do not silently replace
models, resubmit identical jobs or overwrite a changed cache. predict.py runs
on a Slurm compute node in /home/u3129/miniconda3/envs/lammps_deepmd_dpa4.
The immutable all50-geometries.json contains geometry only; it is never used
as evidence that 50 DFT triplets have completed.

METRICS AND FIGURES
Energy: raw VASP TOTEN versus raw model energy, errors in meV/atom. No offset
is fitted to Test50. The training targets were confirmed to use TOTEN.
Forces: all Cartesian components, eV/angstrom; pooled MAE/RMSE, bias, p95,
maximum, equal-structure-weight alternatives, separate Al/Ti summaries.
Stress: six independent tensor entries in GPa; ASE tensile-positive convention,
model stress = -virial/volume. Check virial convention on real inference.
Groups: pooled, liquid/solid, hcp/bcc/gamma families, five fixed composition bins.
2000 bootstrap draws resample whole structures, never independent force
components. The same deterministic draws are used for matching XC groups.
These are descriptive intervals conditional on the current subset, not
sequentially valid confidence bounds; liquid trajectory correlations are unknown.

results/latest.json: current counts, coverage, metrics or explicit missing
predictions, readiness reasons. coverage.png/pdf is available without models.
With all current predictions available: parity.png/pdf, energy_residuals.png/pdf,
convergence.png/pdf, full metrics and immutable results/nNN-<fingerprint>/ archives.
history.json retains distinct common subsets. Comparisons require unchanged
models and analysis code hash. A completed DFT subset is never silently reduced
to hide absent predictions. Check latest.json before reusing a previous plot.

REPRESENTATIVENESS / STOP DISCUSSION
The frozen ID order is liquid 001..025, hcp 026..033, bcc 034..042,
gamma 043..050. At present the common subset is liquid only. Stable errors on
these first structures cannot establish performance on solid solutions.
Do not reorder the test set, pick an attractive subset, or cancel jobs based
on individual observed errors. Family-aware future scheduling could improve
early coverage but should be fixed independently of error values and recorded.

Predeclared HEURISTIC trigger for discussion, never automatic stopping:
- at least 30 common structures, at least 12 liquid and 4 of each solid family;
- at least three occupied composition bins per family, relative to its planned
  composition range (or all planned bins if fewer than three);
- three comparable checkpoints separated by at least five new common structures;
- across those checkpoints, each XC energy/force MAE and RMSE range <=10% of
  the latest value or an absolute floor of 1 meV/atom and 0.005 eV/angstrom;
- current RMSE bootstrap interval width <=50% of RMSE or the same absolute floor.
These thresholds are pragmatic screening choices, not an accuracy guarantee.
Inspect outliers, per-family metrics and missingness even if the trigger passes.
With the current ID ordering, adequate family coverage arrives late in Test50.
For a publication, completion of the fixed 50-structure holdout is preferred.
If the user elects to stop early, report the adaptive stopping rule and actual
subset/coverage explicitly; do not describe it as the full planned benchmark.

Never cancel, hold, restart or requeue DFT as part of this monitoring workflow.
When evidence is sufficient, show the plots and ask the user to decide about
stopping remaining DFT. Respect subsequent explicit authorization.

SCHEDULER CONTEXT
Original arrays: PBE 20957366, PBEsol 20957369, r2SCAN 20957372.
Throttle remains 5 each. The original five-structure Nice groups from
2026-10-22 were replaced on 2026-10-23 at the user's explicit request:
pending crystals precede pending liquids, with family rounds shared by XC.
Nice=0 for 026/034/043 (hcp/bcc/gamma), 1 for 027/035/044, and so on;
remaining liquids have Nice=20..24. Completed or running tasks were skipped.
This is a one-time scheduling change, not a background priority controller.
Selection used only frozen family and structure ID, independent of model errors,
after inspection of the initial liquid subset. Record this family-based change
when discussing completion-order bias or any adaptive early stopping.
All 50 structures remain planned; no hard dependencies or DFT input changes.
Evidence: ../dft/crystal-priority-20261023/change-intent.json and change-result.json.
Scheduled monitoring must preserve these priorities and must not change Slurm.
Do not alter unrelated phonon or Al32 jobs.

Queue diagnostic 2026-10-22 18:28 (Yekaterinburg): user-authorized reapplication
of the existing PBEsol ArrayTaskThrottle=5 refreshed 26 stale JobArrayTaskLimit
reasons to Priority, verified after several backfill cycles. Running jobs were
unchanged. See ../dft/queue-repair-20261022-pbesol/REPORT.txt. This does not
establish a specific controller bug or authorize automatic queue modifications.

Queue diagnostic 2026-10-24 08:16 (Yekaterinburg): explicit user-authorized
same-limit refresh of PBEsol 20957369 and r2SCAN 20957372 changed 6 stale
JobArrayTaskLimit reasons per array to Priority. All 70 remaining tasks now
show Priority; no Test50 task is running. Confirmed across multiple backfill
cycles. ArrayTaskThrottle=5, crystal-first Nice, resources and all 150 input
manifests preserved. PBE unchanged. The actual queue/priority competition
remains; do not describe the refresh as guaranteeing a start or resetting a
running-task counter. Evidence: ../dft/queue-repair-20261024/REPORT.txt.
This one-time correction does not authorize automatic scheduler mutations.

NOTIFICATIONS
Notify on new analyzed triplets with updated figures/metrics, meaningful
errors, completion or a readiness trigger. Stay quiet if nothing relevant
changed, and do not repeat an already reported missing-model blocker.
After all 50 triplets and final analysis are delivered, disable this heartbeat.
