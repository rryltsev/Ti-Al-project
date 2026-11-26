REFERENCE CELLS CHECK 2026-11-02 23:57 Asia/Yekaterinburg.
PBEsol force000 job20968907 completed2026-10-31 20:55:40; existing audit reconfirmed.
PBE force000 job20973061 completed2026-11-02 13:50:11,10:02:58,28SCF.
16 result files downloaded with SHA256; independent XML/OUTCAR/NPY/input geometry
and SCF audit accepted: results/PBE-force000-validation-20261102.json.
PBE reference Fmax3.30627e-6eV/A,maxstress0.203478kbar; CHGCAR127578746bytes.
PBEsol Fmax2.97379e-6eV/A,maxstress0.204908kbar; stress convergence remains open.
r2SCAN force000 job20973063 PENDING/Priority,not started;20h time limit,
Apollo2nodes64ranks240GiB/node. No calculation directory yet; no failure evidence.
Parameter-check hold remains because r2SCAN reference is still missing.
Snapshot: runs/alpha-ti-xc-phonons-20261021-01/reference-status-20261102.json.

CHECKPOINT PHONONS 2026-11-02: PBEsol and r2SCAN 5k/10k/15k/20k/final compared
against their own DFT reference, exact frozen geometries,4x4x3,+/-0.01A,
Phonopy central-pair/ASR,Gamma50^3,sigma0.05THz,mass47.867,integral3/atom.
CPU Apollo jobs20977440(PBEsol,90s) and20977441(r2SCAN,51s), completed0:0;
eight new model evaluations, two accepted final-model force sets reused.
Results: results/PBEsol-checkpoint-phonons-20261102-02/ and
results/r2SCAN-checkpoint-phonons-20261102-02/; portable ZIPs alongside.
Both DOS L1/L2 and Gamma-K-M-Gamma-A sorted-frequency RMSE favor PBEsol15k,
r2SCAN10k among the five checkpoints tested. Later models have smaller energy
validation errors; no automatic replacement of manuscript or production models.
All hashes/exports/integrals verified; no imaginary modes below-0.001THz;
final-model FC reproduction<1e-15eV/A2 and DOS reproduce previous Figure2.
Revision01 retained with diagnostics for compact/full FC comparison shape fix.
DFT numerical/supercell convergence remains pending. VASP jobs unchanged.

PBE CHECKPOINT DOS 2026-11-02: Matched DFT-PBE vs DPA4 5k/15k/20k.
20k CPU Apollo job20976314 completed34s; all three exact DFT POSCAR forces verified.
Results: results/PBE-checkpoint-DOS-20261102-01/ and portable ZIP alongside.
Overlay, three-panel and separate PNG/PDF/SVG; DOS.csv; force constants and code.
Same DFT geometry,4x4x3,+/-0.01A,central pair/ASR,Gamma50^3,sigma0.05THz,mass47.867.
All DOS integrals3; no modes below-0.001THz; DFT/15k curves reproduce Figure2.
15k has the lowest DOS L1 discrepancy among these three under current settings.
This comparison does not change the manuscript checkpoint choice or establish
DFT numerical/supercell convergence. VASP jobs and held convergence checks unchanged.

FIGURE 2 VARIANT 2026-11-01: User explicitly selected PBE checkpoint15000.
New CPU job20973104 evaluated the three exact PBE DFT force geometries.
Six-panel figure (Gamma-K-M-Gamma-A bands and DOS for all three functionals):
results/figure2-six-panels-20261101-02/; portable ZIP alongside it.
PBE15000/PBEsol23100/r2SCAN23200; use this selection for the article variant.
Earlier PBE5000 comparisons remain historical; do not mix them into this figure.
Copies saved to manuscript/figures/variants/; current manuscript not replaced.
Export hashes, six DOS integrals3, 77 experimental points and visual layout checked.
DFT reference diagnostics and electronic/supercell convergence remain pending.

Фононы alpha-Ti: PBE / PBEsol / r2SCAN
Дата подготовки: 2026-10-21 (Asia/Yekaterinburg)

UPDATE 2026-11-01: PBE and r2SCAN displacement pairs completed.
PBE force002 job20972819 ended07:48:38 local,29SCF,11:02:40; r2SCAN force002
job20972837 ended14:19:38,16SCF,12:24:53. Both independent audits passed;
XML/OUTCAR/NPY, frozen geometries, hashes and both EDIFF criteria checked.
r2SCAN STOPCAR request occurred but final SCF reached EDIFF; normal termination,
fresh CHGCAR/TAUCAR/WAVECAR recorded. Do not mistake the stop request for failure.
New DPA4 CPU jobs20973059(PBE checkpoint5000),20973060(r2SCAN checkpoint23200)
completed. Exact same XC-specific DFT POSCARs,4x4x3,opposite0.01A displacements;
same Phonopy central-pair/ASR processing. Gamma50^3,sigma0.05THz,mass47.867.
Results: results/PBE-matched-DFT-DPA4-20261101-01/ and
results/r2SCAN-matched-DFT-DPA4-20261101-01/; PNG/PDF/CSV/NPZ and portable ZIPs.
Both DFT reference force000 calculations were still missing when plotted.
No invented reference forces: a common F0 cancels exactly in (Fplus-Fminus)/2.
Algebra checked numerically to<1e-15eV/A. Reference residual force/stress diagnostics
remain required, as do cutoff/kmesh/smearing/supercell convergence checks.
No modes below-0.001THz on meshes/paths; all DOS integrals3. Shared qpoints/weights,
finite exports, source/output hashes verified; all four PNGs visually inspected.
Sorted-frequency path RMSE: PBE0.2710THz,r2SCAN0.2632THz (no eigenvector tracking).
Reference continuation policies09/10,driversv9/v10,launcher15LF;20h Apollo64ranks,
KPAR8,NCORE4,240GiB/node. PBE fresh; r2SCAN uses accepted second displacement seed.
Parameter checks remain on hold. See production06 submission records for job IDs.

MATCHED DPA4/DFT COMPARISON 2026-10-31.
User requested DPA4-PBEsol checkpoint23100 evaluated on exact DFT-PBEsol
4x4x3 force000/001/002 POSCAR files, with no relaxation. CPU Apollo job20972853
completed in35s, exit0; three force sets collected with SHA256 verification.
Run records: runs/alpha-ti-xc-phonons-20261021-01/DPA4-PBEsol-matched-20261031-01/.
Final figures/data: results/PBEsol-matched-DFT-DPA4-20261031-02/.
Both spectra use identical Phonopy functions: reference and mean-force removal,
central pair, traditional FC fit, symfc projector and ASR polishing. Exact DFT
FC reproduction; common Gamma50^3 mesh, mass47.867, sigma0.05THz, integral3/atom.
Optical discrepancy persists: upper Gamma optical5.7575THz DFT vs4.9542 DPA4;
high DOS peak7.194THz/0.7815 DFT vs7.328THz/1.0068 DPA4. No modes below-0.001THz
on mesh/path. This does NOT establish DFT cutoff/kmesh/smearing/supercell convergence.
Scripts dpa4_matched_forces.py, dpa4_matched_cluster.py, analyze_dpa4_matched.py,
verify_dpa4_matched.py; portable ZIP and checksum manifest under results.
Earlier comparison01 used different force cells/geometries and remains historical.
Parameter-check hold and all existing VASP jobs remain unchanged.

USER UPDATE 2026-10-31 evening (scheduler-overrides-14.json).
User explicitly requested deferring PBEsol checks atotherparameters until baseline
phononcalculations forallfunctionals finish. PBEsolcutoff20972836 cancelled21:44:36
whilePENDING,elapsed0; noVASPexecuted. Otherjobsnotcancelled: PBEforce00220972819,
r2SCANforce00220972837. DoNOTresubmitparameterchecks now.
Newworkflow-holds-20261031.json enforced incontinue_production.py BEFOREintent
orSSHsubmission: checksdisabled untilaccepted001/002/000existsforall3XC.
Gateverifiedofflinedenial, missingPBE002/000and r2SCAN002/000. Afterreadiness,
reviewremainingchecks anduseNEWimmutableattempt; cancelledintentisnotreusable.

Newrequestedcomparison:results/PBEsol-DFT-vs-DPA4-20261031-01/.
DFT-PBEsolcurrent550eV4x4x3 vsARTICLE DPA4-PBEsolcheckpoint23100,
modelSHA5a5359a713848d98a7746c70b46be65313d7d990c6bf39d68b40908dd9c7c295.
MLinputaudit/oct25/all_xc_audit/results/17/equilibrium/results.npz,9x9x9forcecell,
ownrelaxedcell a2.897577546,c4.586322562A; DFTownrelaxedcell a2.89931448,c4.58937358.
Modelidentity,originaltasksummary/protocol/workerandcollectionSHAallverified.
NoMLinference,newVASP orreplacementofmanuscriptfiguresperformed.
Re-evaluatedMLC/D_NatGamma50^3byFFT; directFouriermatrixcheckerror3.75e-16,
cachedbandsagree3.97e-6THz(differenceinlegacyunitconstant).C/47.867masscheckpasses.
MLretainsoriginalASEFrederiksen/symmetrize3/acousticforceconstantprocessing;
Hermitianresidual5.46e-6inD_N Fouriermatrixrecordedinsummary (legacy uppertriangleeigensolverretained).
BothDOSuseidenticalGamma50^3,mass47.867,sigma.05THz,linearhistdeposition.0005THz
andGaussian8sigma. FrequencygridrefinementL1/mode<1.8e-6;DFTvsoriginalexactGaussian
L1/mode2.55e-7; bothintegrals3. Owngeometriesandunequalforcecellsareexplicit
infigure/summary. DFTstillPRELIMINARY,noisolatedsame-geometryaccuracyclaim.
PNG/PDF/CSV,summary,inputhashes,reproduce.py,inputs andZIP inresults.
Plotvisuallychecked. Cancellationrecords PBEsol-cutoff-cancel-20261031-*.
CommonvalidatedPBE/PBEsol/r2SCANfinalDOSgoalstillpending; basecalculationscontinue.

PRELIMINARY PHONON FIGURES DELIVERED 2026-10-31 21:39 Asia/Yekaterinburg.
PBEsol550eV,96atom4x4x3,Gamma6x6x6electronicmesh. Allthreeforceauditsaccepted.
Localphonopy4.8.2 calculation (isolatedphonons/.analysis-deps; noVASPjobsmodified).
Results:results/PBEsol-phonons-preliminary-20261031-02/.
PNG/PDFcombinedbands+DOS andseparateDOS, CSVs,FCnpy,phonopy_params.yaml,
mesh50.npz,inputs,requirements.lock,reproduce.py,summary/verificationandSHAmanifest.
Self-containedZIP:results/PBEsol-phonons-preliminary-20261031.zip, SHA256
658950502468339f3ad44fc84b0e41d564ce806949e53a830ceb41177b7de2d9.
Processing: subtractreferenceF0;subtractper-setmean;retainantisymmetricplus/minus
centralpair;traditionalfit;spacegroup/permutation/translational symfcprojector;
10translation/permutationcycles polish finiteprecisionASR (maxchange6.408e-11eV/A2).
FinalASRmaxresidual1.149e-14eV/A2,permutationresidual0;fullASRmaxFCchange1.17954e-4.
Firstattempt01 preservedincomplete: strictASRgate caughtprojector3.356e-9residual.
Gammaacousticfrequencies~1e-7THz afterASR (serializedYAML~3e-7,roundoff).
RawGammaacoustic~-.00409,+.00488,+.00488THz retainedindiagnosticfiles.
No frequencies<-.001THz on50^3Gamma-centeredqmesh orGamma-M-K-Gamma-A-L-H-A|L-M|K-Hpath.
Meshmax8.642149THz,bandpathmax8.644542THz;GaussianDOSsigma.05THz,Ti47.867amu,
DOSintegral3.0000000000000004modes/atom. Mesh6084irreducibleq,weights125000.
Allfiguresvisuallyinspected,CSVnorm/sha/YAMLreloadvalidated. NegativeGaussian
DOSfrequencytail is smoothing, notimaginarymodes.
PRELIMINARYONLY: cutoff/kmesh/smearingandsupercellconvergenceunestablished.
SharedPBE/PBEsol/r2SCANfinalDOSgoalstillopen; monitoringmustNOTdisableyet.

CURRENT OBSERVATION 2026-10-31 21:25 Asia/Yekaterinburg (scheduler-overrides-13.json).
PBEsol20968907 referenceforce000 COMPLETED20:55:40,11:09:00,28SCF, EDIFFyes:
dE=-8.7166e-9,banddE=1.9781e-10eV. NoSTOPCAR, finished2h51beforeguard.
r2SCAN20968910 firstdisplacement COMPLETED20:44:37,08:18:35,10SCFafterWAVErestart,
EDIFFyes:dE=-1.1496e-9,banddE=-1.3536e-9. NoSTOPCAR,3h41m25sbeforeguard.
Both independently audited XML/OUTCAR/NPY96x3forces, geometry/input/PAW hashes,
primitiveequilibriumgates and exact phonopydisplacements.15resultfiles each
collectedwithSHA256, noPOTCAR. Reports inresults dated20261031.
PBEsol allbaseforces001/002/000ready. Reference maxF2.9738e-6eV/A;
reference supercellstress.204908kbar is separately flagged forcutoff/kmeshtests,
primitiveequilibriumstress.009825kbarpassed. Pair-even force max.000317eV/A,
maxeven/maxodd.004745; seePBEsol-displacement-reference-audit-20261031.json.
r2newCHGCAR/TAUCAR/WAVEwrittenatcleanexit. WAVE25201037952bytes, headersvalid,
fullSHArecordedbydriver:d80676dd9f8d682f70dc1efdc2b66b2845e23e30f04e7480cefc4fffacc6e795.

Next PBEsolcutoff20972836: production05/check-cutoff, ENCUT650,64ranks/2nodes,
320GiB/node (largerbasis memory estimate),KPAR8NCORE4,20h,policy07/driver7/launch13,
guardv5>=6h calibratedfrommeasuredPBEsol, dynamic2*maxSCF+3h preserved.
Nextr2force00220972837: production05/force002,64ranks/2nodes240GiB/node20h,
policy08/driver8/launch13,guardv2>=8h. Oppositefrozenphonopygeometry; initialize
fromconverged20968910 WAVE+CHGCAR+TAUCAR,ISTART1/ICHARG1,LWAVEtrue.
Driverchecksseedgeometry, displacementdelta.02Aatom1,finite densities/charge,
parentCOMPLETED+EDIFF,fullSHA/copies/quota, and rejectsWAVECARnotread.
No scientificcutoff/smearing/kmesh changes insecond displacement.
BothnewjobsPENDING/Priority21:24:44. PBEsecond20972819alreadyRUNNINGsince20:45:58.

Storage: oldr2restart03WAVE archivalto/mainhomefailedEDQUOT after15.12GB;
originaluntouched, incompletecopyremovedsafely. ChangeddestinationtoLustre/home1.
OldWAVEsuccessfullyarchived21:22:52withfullsource/destinationSHA256andfsync,
oldpathpreservedasSYMLINKto/home1/u3129/tial-phonons-archive/alpha-ti-xc-phonons-20261021-01/r2SCAN/force-001-restart-03/WAVECAR.
SHA=dd986de4213079fc680296be62f57c6e008a5545305625b5897b57a937c2ce78.
Manifests r2SCAN-old-wavecar-archive-20261031-*-result/failure.json retainhistory.
Actualhome6quota~67.55/100GiB, roomfornext23.47GiBcopy+margin; runtimegate repeats.
Lustre quota inheritsdefaults; detailsrequirepermission, doNOTclaimunlimited.
Archivecopyworked; keepcheckingbeforefuturelargeoutputs. NoA100VASP.
Futuresteps: PBE policy06only002; r2policy08only002; PBEsolpolicy07onlycutoff.
Review andversionnextpolicies/drivers beforelatersteps. DoNOTblindlyreuseoldweakdriver.
Supercell-convergence-plan-01.json isDESIGNONLY, nojobs:216atom6x6x3with4x4x6k,
192atom4x4x6with6x6x3k preserveunfoldedprimitive24x24x18. Revisitafterbasekmeshdiagnostic.
No finalDOSqualificationyet; cutoff/kmesh/smearing/supercellconvergence remainopen.

CURRENT OBSERVATION 2026-10-31 20:06 Asia/Yekaterinburg (scheduler-overrides-12.json).
PBE first displacement 20968908 COMPLETED at19:31:36 in09:04:39, 24SCF,
EDIFF reached: dE=-9.0367e-9, band dE=-1.2681e-11 eV; noSTOPCAR/checkpointrequest.
CHGCAR127578746bytes freshly saved19:31:30; WAVEdisabled as configured.
15 result files collected withSHA256, noPOTCAR. Independent audit accepted
XML/OUTCAR/NPYforces, PAW/inputhashes, geometry/displacement+.01A, eqgates.
Reports results/PBE-force001-validation-20261031.json and PBE-force001-forces.csv.
Second displacement PBE20972819 queued Apollo64MPI/2nodes240GiB/node20h,
production-05/force-002, policy06/driverproduction_step_v6/launch_v12/guard_v2.
Sourceinputs unchanged; only parallelism and checkpoint output enabled as before.
No WAVE output (LWAVEfalse). Normal fresh initialization ICHARG2 from frozen inputs.
State20:05:56 PENDING Priority, forecast Nov1 07:58:03 tentative.
continue_production.py now selects reviewed PBEpolicy06 and validates deployed hashes;
queue reconciliation restricted to this phonon remote root (unrelated t50jobs ignored).
Future PBE force000/diagnostic steps must be reviewed before policy06 is expanded.
monitor_production.py discovers production05 submission records.

CURRENT OBSERVATION 2026-10-31 16:13 Asia/Yekaterinburg.
All three jobs RUNNING on Apollo; CPU counters advance, no fatal indicators.
PBE 20968908: 13 SCF, dE=6.8067e-6, band dE=-4.5541e-7 eV; no EDIFF.
PBEsol reference force-000 20968907: 17 SCF, dE=8.0153e-5,
band dE=-1.5393e-5 eV; no EDIFF. Both earlier displacements remain accepted.
r2SCAN 20968910: WAVECAR/CHGCAR/TAUCAR read successfully, 4 SCF,
dE=-1.4305e-6, band dE=-1.1897e-6 eV; no EDIFF. Input WAVE is not new output.
Measured recent SCF: PBE 21-29 min; PBEsol 20-31 min; r2SCAN 38-43 min.
Hard deadlines Nov1: PBEsol05:46:40, PBE06:26:57, r2SCAN08:26:02.
Current guard STOPCAR targets: PBE Oct31 22:26:57, PBEsol Oct31 23:46:40,
r2SCAN Nov1 00:26:02; dynamic reserve can increase if SCF slows down.
No STOPCAR yet; guards armed. Quota ~90.6/100 GiB, ~9.4 GiB free;
another immutable wave copy would require archiving an older verified checkpoint.
PBEsol completion prospects favourable by prior25/27SCF; PBE reasonable but
not guaranteed; r2SCAN uncertain with only4 newSCF, ~12 more before guard.
Evidence: monitor/production-latest.json and status-20261031-1615.json.
No allocation/scientific changes and no new continuation submission in this check.

АКТУАЛЬНОЕ НАБЛЮДЕНИЕ, 2026-10-30 06:28 (Екатеринбург).
Все три новые задачи пока PENDING / Priority; новых итераций нет.
PBEsol force-000 20968907: прогноз 30.10 18:36:39.
PBE restart-05 20968908: прогноз 30.10 19:11:00.
r2SCAN restart-04 20968910: прогноз 30.10 19:46:00.
Это прогнозы Slurm, не гарантированные времена запуска. PartitionTimeLimit нет.
Последнее продолжение r2SCAN 20964090 прочитало WAVECAR и дошло до 13 SCF,
но EDIFF не достигло (dE=-5.0888e-8, d eps=-2.8157e-7 эВ при пороге 1e-8).
Сохранённый новый WAVECAR 25201037952 байт будет использован заданием 20968910.
Квота home6 около 66.36/100 GiB; место под одну следующую копию WAVE есть,
проверка повторяется перед стартом VASP. Apollo, 20h, A100 запрещены.
Полный монитор и компактный статус: status-20261030-0628.json.

АКТУАЛЬНОЕ СОСТОЯНИЕ, 2026-10-29 около23:18 (scheduler-overrides-11.json).
PBEsol force00220962927 сошёлся27SCF и завершился28.10 23:17:44,13:40:49.
Независимый аудит принят: inputs/PAW, displacement -.01A атом1, XML/OUTCAR/NPY,
конечные96x3силы, SCF dE=-2.2847e-9 иd eps=-2.3379e-10 приEDIFF1e-8.
При сравнении XML/POSCAR учитывать периодичность fractional coordinates:
отрицательное смещение XML оборачивает в[0,1), ошибка послеwrap3.77e-9.
Оба смещения PBEsol готовы. CSV/JSONотчёты и pair-audit вresults; метаданные
force002 скачаны сSHA256 вcollected/PBEsol/production-04/force-002.
Следующий эталон force00020968907: Apollo64MPI/2nodes240GiB/node20h,
production_step_v5.py/launch_v11.sbatch/execution-policy-05.json, guardv4min6h.
Guard основан на измеренных post-SCF2002s/2140s, динамический2*maxSCF+3h сохранён.
PBE20964089: NODE_FAIL29.10 13:14:15 наapollo20+22, только7SCF, чистогоcheckpointнет.
apollo22BootTime13:13:26 иslurmdStart13:14:15 совпадают со сбоем. Точная причина
требует админских логов; фактNODE_FAIL подтверждён. Stale workerstate=running
не принимать! Монитор теперь явно сохраняет scheduler_state/effective_state.
PBEвосстановлен изцелогоCHGCAR20959753 как20968908, каталогrestart-05,
64MPI2nodes240GiB/node20h, исключеныA100иapollo22. Прерванные outputs не используются.
r2SCAN20964090 прочиталWAVECAR и дошёл13SCF: dE=-5.0888e-8,d eps=-2.8157e-7eV;
EDIFFещё нет. ЧистыйSTOPCAR, завершён29.10 22:31:22,13:48:54; новыйWAVE25.2GBсохранён.
Продолжениеr2SCAN20968910: ISTART1,свежийWAVE+CHGCAR+TAUCAR,64MPI2nodes20h,
restart_step_v7.py/launch_v11.sbatch, targetproduction-02/force-001-restart-04.
Из-за~89.7/100GiBквоты старыйWAVEизrestart-02 перенесён в/mainhomearchive
/home/u3129/tial-phonons-archive/alpha-ti-xc-phonons-20261021-01/r2SCAN/force-001-restart-02/WAVECAR.
ПолныеSHA256исходника иархивасовпали, archivefsync выполнен. Старыйпуть заменён
symlink только ПОСЛЕсверки. Данные сохранены, home6квота снизиласьдо~66.3GiB.
Архивныйразделmainhome сloginдоступен длязаписи,computeобычноread-only;
тудаНЕписатьVASPoutputs, толькоархивироватьпровереннымпереносом. Не трактовать
непоказаннуюquota mainhomeкакноль: проверятьместо, копироватьс сохранениемисходника
иSHA256/fsyncдоunlink. Вслучаефейла оригиналостаётсянаместе.
Архивныйhelperжёсткоскопированнастарыйrestart02; не перезапускатьего вслепую.
При дальнейшемwaveкопировании сноваоценитьквоту иархивироватьsupersededWave
по аналогичномупротоколу, сохраняяисториюссылок. Не включатьWAVEдругимXCвслепую.
Передcheck-cutoff/check-kmesh оценитьпамятьдлябольшегоbasis/Nk, не повторятьOOM.
НовыхфинальныхDOSещёнет, cutoff/kmesh/smearing иsupercellconvergenceпокаНЕустановлены.
ВсеVASPтолькоApollo,20h;A100запрещены. Кумулятивныхресурсныхбюджетовнет.

АКТУАЛЬНО: КОНТРОЛЬНЫЕ ТОЧКИ И ПРОДОЛЖЕНИЯ, 2026-10-28 около21:35.
PBE20959753: чистый STOPCAR, завершён09:34:14,16SCF, EDIFF не достигнут.
Новый CHGCAR127578746bytes записан09:34, локально скачан сSHA256, сетка полная,
заряд960.00006219e, PAW96; результат годится для restart, не как сошедшиеся силы.
r2SCAN20960773: чистый STOPCAR, завершён19:13:22,9SCF, dE=-.0099453eV, EDIFF нет.
Сохранены CHGCAR/TAUCAR и WAVECAR25201037952bytes (~23.47GiB).
WAVEheader: RECL389952,RTAG45200(complex-single),112k,576bands,550eV;
ожидаемый размер совпадает,112bandheaders и выборочные коэффициенты конечны.
Полный hash WAVE и контроль копирования будут выполнены в allocation продолжения.
PBEsol force002 20962927 запущен09:36:55, около21:35 всё ещё22SCF, dE2.093e-7eV.
Для ЭТОГО задания новый guardv3: max(6h,2*maxSCF+3h) вместо8h. Причина:
первое PBEsol post-SCF2002s, последний maxSCF2481s; шестичасовой запас сохраняется,
давая ещё2h на сходимость. Slurm EndTime прежний29.10 05:36:55.
Старый guardPID4193916 остановлен только после проверки cmdline наapollo11;
VASP не трогали. Документы PBEsol-force002-measured-guard-20261028*.json.
Новые продолжения PBE20964089 и r2SCAN20964090: Apollo64MPI/2nodes,KPAR8,NCORE4,
240GiB/node,20h. Источники — свежие остановленные расчёты. Каталоги:
PBE production02/force001-restart04, r2SCAN production02/force001-restart03
(реальные имена каталогов с дефисами, см.scheduler-overrides-10.json).
r2SCAN ISTART1 сWAVECAR+CHGCAR+TAUCAR; PBE ISTART0 сCHGCAR и ускоренным layout.
restart_step_v6.py проверяет плотности, квоту, source hashes, копии; r2SCAN
прерывает попытку при WAVECAR-not-read вместо молчаливого начала с нуля.
Реальная квота~65.2/100GiB: копия WAVE~23.5GiB+плотности+4GiB помещается;
повторно проверяется в allocation. Не включать WAVE одновременно у других XC.
Текущие jobs/settings: scheduler-overrides-10.json; A100 запрещены.

АКТУАЛЬНОЕ НАБЛЮДЕНИЕ, 2026-10-28 09:15.
PBEsol force-002 20962927 ещё PENDING Resources; прогноз Slurm сегодня 09:42:33.
PBE 20959753: 16 SCF, EDIFF не достигнут, последняя dE=4.9749e-6 eV.
Guard отправил STOPCAR около08:07; stdout подтверждает soft stop, сейчас VASP
финализирует расчёт, normal termination ещё нет. До выхода новую CHGCAR не принимать.
Повторять неизменный 32-rank density-only restart без оценки времени не следует;
PBEsol успешно показал64ranks/KPAR8, рассмотреть этот layout для следующего PBE.
r2SCAN 20960773: RUNNING,3 SCF,4-я в работе; EDIFF ещё нет, LWAVE=T,
но WAVECAR по-прежнему0bytes. Готового checkpoint орбиталей пока нет.
Сошёлся и принят только PBEsol force-001. Все новые запуски Apollo,20h,A100 запрещены.
Доказательства: monitor/production-latest.json и status-20261028-0915.json.

АКТУАЛЬНО: ПЕРВЫЕ ПРОВЕРЕННЫЕ СИЛЫ PBEsol, 2026-10-28 около 09:16.
PBEsol force-001 20959338: COMPLETED 28.10 08:41:39, 08:31:00, SCF25, EDIFF достигнут.
Последние dE=-5.6752e-10 eV, d eps=-1.4822e-9 eV при EDIFF1e-8. STOPCAR не было.
Независимо проверены XML/OUTCAR/forces.npy, конечность всех 96x3 сил,
согласие XML с OUTCAR до 5e-7 eV/A (точность печати), совпадение NPY с XML,
PAW/input hashes, ячейка/порядок атомов и смещение атома1 на .01 A по phonopy YAML.
Максимальная восстанавливающая сила .06680326 eV/A; сумма сил <5e-8 eV/A на компоненту.
Исходная equilibrium-02 проходит force/stress gates. Результат принят как force-001,
но DOS и convergence diagnostics ещё не готовы: остальные смещения не рассчитаны.
Данные collected/PBEsol/production-03/force-001 скачаны с SHA256, включая CHGCAR.
Плотность полная, 180x180x216, 960 electrons; не публиковать POTCAR.
Отчёт results/PBEsol-force001-validation-20261028.json, силы results/PBEsol-force001-forces.csv.
Следующий шаг force-002 поставлен как 20962927, Apollo64 MPI/2nodes, KPAR8/NCORE4,
240GiB/node,20h; production-04 и execution-policy-04.json. A100 запрещены.
continue_production.py теперь выбирает PBEsol production-04; исправлен разбор
sacct -P (без завершающего |). Первая проверка prerequisite остановилась ДО sbatch,
после исправления подан только один job. Монитор включает production-04 records.
Текущие ресурсы и jobs: config/scheduler-overrides-09.json.

АКТУАЛЬНО: ВСЕ ТРИ ЗАПУЩЕНЫ, 2026-10-28 около 07:40.
PBE 20959753: apollo11, старт 27.10 20:07, 15 SCF, идёт 16-я, dE=3.878e-6 eV.
PBEsol 20959338: apollo9+36, старт 28.10 00:10, 22 SCF, идёт 23-я, dE=2.3381e-7 eV.
r2SCAN 20960773: apollo13, старт 28.10 04:30, 2 SCF, идёт 3-я.
Ни один новый force-001 пока не достиг EDIFF=1e-8. OOM/MPI ошибок не обнаружено.
У r2SCAN OUTCAR подтверждает LWAVE=T, однако WAVECAR пока 0 bytes —
готового checkpoint орбиталей ещё нет, проверить после штатного завершения.
У PBE/PBEsol LWAVE=F, WAVECAR пустые; только r2SCAN зарезервирован под WAVE.
Фактическая home6 квота ~41.22/100GiB, запас ~58.78GiB, из них r2SCAN WAVE
может потребовать до 46.94GiB. Не включать дополнительные WAVE вслепую.
Все guard v2 armed; Slurm EndTime: PBE 28.10 16:07:24, PBEsol 28.10 20:10:39,
r2SCAN 29.10 00:30:48. Guard резервирует минимум 8h и читает actual EndTime.
Полный монитор: monitor/production-latest.json; компактно status-20261028-0740.json.

АКТУАЛЬНОЕ СОСТОЯНИЕ, 2026-10-26 около 08:28 (scheduler-overrides-08.json):
r2SCAN 20959337: COMPLETED 26.10 00:36:19 после 15:19:14, 9 электронных итераций.
Это штатный STOPCAR, НЕ EDIFF: последняя dE=0.053903 eV, силы не принимать.
CHGCAR 127578746 bytes и TAUCAR 127374170 bytes сохранены, normal termination есть.
Локальная копия collected/r2SCAN/production-02/force-001-restart-01 проверена SHA256;
обе сетки 180x180x216 полные/конечные, заряд 960.0000625 e, PAW96, XML прочитан.
Аудиты: r2SCAN-restart01-collection-20261026.json и r2SCAN-restart01-density-audit-20261026.json.
Продолжение 20960773: apollo 32 MPI, 240GiB, 20h; restart_step_v5.py/launch_v7.sbatch,
каталог r2SCAN/production-02/force-001-restart-02, источник restart-01.
Изменение операционное: LWAVE=.TRUE. для сохранения орбиталей. Без WAVECAR
первые итерации повторяются (NELMDL=-5 и ISTART=0); простой неизменный restart
не обеспечивает эффективного накопления прогресса между 20h запусками.
Квота сейчас ~37/100 GiB. Консервативная оценка WAVECAR <=~47 GiB по NKPTS112,
NBANDS576, max plane waves48743, 16 bytes/coefficient. Драйвер проверит свободное
место на WAVECAR+плотности+5GiB запаса в allocation. НЕ включать LWAVE у остальных
задач одновременно без нового расчёта квоты. Следующая копия WAVECAR также
требует фактического размера и отдельной проверки места; не дублировать вслепую.
PBE 20959753 и PBEsol 20959338 по-прежнему PENDING Priority, 20h, Apollo.
Прогноз: PBEsol 26.10 18:57:29, PBE/r2SCAN 27.10 00:53:24, не гарантия.
Новых сошедшихся force-001 пока нет. PartitionTimeLimit устранён, A100 запрещены.

АКТУАЛЬНО: ЛИМИТ ВРЕМЕНИ ИСПРАВЛЕН, 2026-10-25 около 07:22.
PBE 20959753, r2SCAN 20959337, PBEsol 20959338 блокировались PartitionTimeLimit:
запрошено 48h, фактический Apollo MaxTime на момент проверки 45h.
Увеличенное окно выходных НЕ является постоянным лимитом partition.
TimeLimit всех трёх изменён IN PLACE на 20:00:00, номера/SubmitTime/EligibleTime сохранены.
После исправления причина PENDING — Priority, PartitionTimeLimit исчезла.
Последняя повторная проверка: Slurm прогнозирует все три задания 26.10 17:30:52.
Это оценка очереди, не гарантия запуска. Дополнительных расчётов не подавали.
Текущие scheduler настройки: config/scheduler-overrides-07.json.
Будущие подачи используют 20h из config/vasp-placement.json; НЕ повторять 48h
по старым историческим overrides/one-off scripts. Накопительный бюджет отсутствует.
Все VASP только Apollo, A100 запрещены. Научные настройки прежние.
Guard v2 получает актуальный EndTime при запуске, минимальный запас на остановку 8h.
Продолжение после нового checkpoint подавать лишь после валидации данных/очереди;
автоматически поставленного следующего restart для этих запусков сейчас нет.

АКТУАЛЬНО: ПРИНУДИТЕЛЬНАЯ ОСТАНОВКА ПО ПРОСЬБЕ ПОЛЬЗОВАТЕЛЯ, 2026-10-24.
PBE 20959336 на tesla-a101 отменён через scancel; сохранения новых итераций не ждём.
Ожидавшее чистого завершения продолжение 20959747 отменено ДО запуска.
Новый PBE 20959753: apollo, 32 MPI, 240GiB, 48h, без зависимости от остановленного job.
Источник — последний проверенный CHGCAR чисто остановленного 20958089,
PBE/production-02/force-001; SHA256 повторно подтверждён перед подачей.
Новый каталог: PBE/production-02/force-001-restart-03. Никакие старые файлы не удалены.
launch_v6.sbatch + restart_step_v4.py: проверены пути, SHA256 и отказ вне Apollo.
Пользователь запретил ВСЕ дальнейшие VASP на A100, включая CPU-only и GPU builds.
Правила: config/vasp-placement.json, scheduler-overrides-06.json.
continue_production.py проверяет Apollo и исключает tesla-a101; старый optimizer отключён.
PBEsol 20959338 и r2SCAN 20959337 остаются в apollo без изменений.
Предыдущие записи ниже — история; soft-stop продолжение больше НЕ актуально.

АКТУАЛЬНОЕ ИСПРАВЛЕНИЕ, 2026-10-24 около 08:04 (scheduler-overrides-05.json):
CPU-only VASP больше НЕ размещать в v100/на tesla-a101: даже без GPU TRES
он блокирует GPU-задачи, занимая почти всю память хоста двух A100.
PBE 20959336 запустился там 24.10 в 01:49; 4 SCF за 6 часов, EDIFF не достигнут.
Первый LOOP 3715 s против прежних 2258 s: ускорения вычислений не получено.
24.10 в 08:01:26 отправлен STOPCAR LABORT для мягкой остановки с плотностью.
На 08:04 узел ещё НЕ освобождён: ждём окончания итерации/финализации VASP.
Продолжение 20959747: apollo, 32 MPI, 240GiB, 48h, afterany:20959336.
Новый каталог PBE/production-02/force-001-restart-02, источник restart-01.
launch_v5.sbatch + restart_step_v3.py проверены; SHA256 и intent сохранены.
Старт VASP разрешён только после проверки чистого завершения и CHGCAR.
PBEsol 20959338 и r2SCAN 20959337 остаются PENDING apollo, без изменений.
Новых сошедшихся сил пока нет. CHGCAR в работающем restart-01 пока является
входной копией старой плотности; новую запись подтверждать только после выхода.
Запрос STOPCAR не означает, что память уже освобождена: проверять squeue/sacct.
Не повторять старый optimize_pending_resources.py (теперь заблокирован).

АКТУАЛЬНОЕ РАЗМЕЩЕНИЕ, 2026-10-23 около 21:33 (scheduler-overrides-04.json):
Slurm теперь сообщает MaxTime=3-07:00:00 (79h) для apollo/v100/tesla/all.
Прежнее утверждение "системный максимум 20h" больше НЕ актуально.
Три ожидающих задания обновлены IN PLACE, номера и время подачи сохранены:
PBE 20959336: v100, CPU-only vasp_std, 1 узел x 32 MPI, 240GiB, 48h.
r2SCAN 20959337: apollo, 1 узел x 32 MPI, 240GiB, 48h.
PBEsol 20959338: apollo, 2 узла x 32 MPI, 240GiB/узел, 48h.
Научные входы и число MPI/KPAR/NCORE не изменены. Тот же SSE4.2 CPU-бинарник.
Guard читает реальный EndTime Slurm, поэтому учитывает новое окно 48h.
Старые metadata ресурсов в замороженных драйверах могут содержать 300GiB/20h;
источник истины для текущей allocation — Slurm + scheduler-overrides-04.json.
Память 240GiB имеет запас над 32*наблюдавшимся MaxRSS (~214.4GiB).
Перенос в v100 ускоряет прогноз запуска, но CPU throughput ещё не измерен:
у tesla-a101 24 физических ядра и 48 аппаратных потоков, у Apollo 36 ядер.
Результаты sbatch --test-only (без подачи дубликатов): queue-options-20261023*.json;
точные before/after: resource-optimization-20261023-applied.json.

АКТУАЛЬНЫЕ ЗАДАНИЯ, 2026-10-23 около 13:12:
PBE 20959336, r2SCAN 20959337 — продолжения сохранённых плотностей;
PBEsol 20959338 — исправленный production-03, 64 ranks / KPAR8.
Предыдущие продолжения 20959188/20959189 завершились до запуска VASP:
launcher делал cd "$SLURM_SUBMIT_DIR", который оказался /home/u3129,
и искал относительный restart_step_v2.py там. Исходные плотности не затронуты.
Ожидающий PBEsol 20959187 отменён до запуска из-за той же ошибки launcher.
Новый launch_v4.sbatch использует абсолютный путь драйвера и явный cd
в корень протокола ПОСЛЕ source env.sh. Проверка launcher из home-каталога
прошла для обоих драйверов без запуска VASP; launch-v4-preflight.json.
История: *restart03* и *production03-force-001-submit03* records.
Все предыдущие номера ниже относятся к истории, не подавать их заново.

ОБНОВЛЕНИЕ 2026-10-23: исходные production-02 force-001 завершились.
r2SCAN 20958091 и PBE 20958089 штатно остановлены по STOPCAR, НЕ сошлись.
Старые status.json/validation.json с validated не принимать за SCF сходимость.
PBE CHGCAR 127578746 bytes; r2SCAN CHGCAR 127578746 и TAUCAR 127374170 bytes.
Продолжения 20958709 (PBE), 20958711 (r2SCAN) заменены ДО запуска:
актуальные 20959188 (PBE), 20959189 (r2SCAN), restart_step_v2.py.
Причина: фактическая запись координат CHGCAR имеет 6 знаков после запятой;
исходный validator отвергал штатное округление. Допуск теперь 3e-5 A по
позициям, 1e-6 A по ячейке, rtol=0; исходный POSCAR остаётся hash-identical.
Также применяется guard v2 с минимум 8h на итерацию и завершающие операции.
Новые job records: *production02-force-001-restart02-submission.json;
каталог результатов остаётся force-001-restart-01 (старые jobs не запускались).
PBEsol 20958090 достиг TIMEOUT во время завершающих операций после STOPCAR:
CHGCAR отсутствует. Его ожидающее продолжение 20958710 отменено как заведомо
неработоспособное. Диагностика сохранена, старый расчёт не удалён.
Исправленный PBEsol: production-03/force-001, 64 ranks на 2 узлах, KPAR=8,
NCORE=4, NBANDS=576, память 300GiB/узел, прежние научные настройки.
Guard v2 резервирует минимум 8h, либо 2*max(8 последних SCF)+3h для
окончания электронной итерации, расчёта сил и сохранения плотностей.
Номер новой подачи хранится в PBEsol-production03-force-001-submit02-submission.json.
Slurm принял исправленный PBEsol как 20959187.
Первая подача production03 была явно отклонена sbatch из-за CRLF (job не создан);
используется отдельный LF-only production-step-v3-unix.sbatch, история сохранена.
Монитор теперь проверяет production-02 и production-03 и все их submission records.
Локальные копии контрольных файлов/диагностики: runs/<id>/collected; SHA256
переноса: stopped-checkpoints-collection.json. Эти плотности ещё НЕ сошедшиеся.

ЗАЩИТА ОТ TIMEOUT — 2026-10-22
Внутри allocations 20958089/20958090/20958091 запущены checkpoint_guard.py
через srun --overlap (1 лёгкий процесс, без нового расчёта VASP).
Проверка каждые 300 секунд. За max(3 часа, 2*максимум последних 8 SCF
итераций + 30 минут) до EndTime записывается STOPCAR: LABORT=.TRUE.
Текущий EndTime: 2026-10-23 04:35:30 Asia/Yekaterinburg.
Это запрос остановки после электронной итерации, не аварийное завершение.
Плотности CHGCAR и для r2SCAN TAUCAR сохраняются при штатном выходе.
WAVECAR у текущих запусков отключён: орбитали/история mixing не сохраняются.
Продолжение поэтому сохраняет плотности, но не точное состояние SCF.
Зависимые afterany задания: PBE 20958709, PBEsol 20958710, r2SCAN 20958711.
Они используют restart_step.py и каталоги force-001-restart-01.
Если исходный расчёт уже сошёлся, продолжение заканчивается без запуска VASP.
Иначе проверяются штатное завершение, запрос checkpoint, неизменность входов,
геометрия, конечность сеток, заряд 960 e, PAW occupancies, TAUCAR и квота.
Копии проверяются SHA256, единственное изменение INCAR — ICHARG=1.
STOPCAR не переносится. Исходные данные сохраняются в прежнем каталоге.
В продолжении также работает защита до 20-часового лимита; повторный
незавершённый расчёт требует анализа, бесконечной цепочки перезапусков нет.
ВАЖНО: старый production_step.py может записать validated после STOPCAR.
Это НЕ доказательство сходимости. Требуются OUTCAR: EDIFF is reached,
normal termination, проверенный XML/силы и COMPLETED в Slurm.
monitor_production.py теперь выводит отдельный признак EDIFF и checkpoint.
continue_production.py запрещает следующий этап без этих проверок.
Исходники/хеши/интенты: checkpoint-restart-source-manifest.json и
*production02-force-001-restart01* в каталоге runs текущего протокола.
Защита и зависимые задания работают на кластере независимо от ноутбука.
При последующем мониторинге проверять и исходные, и зависимые задания;
не подавать дополнительные продолжения при уже существующем intent/job.

АКТУАЛЬНАЯ ПОЛИТИКА РЕСУРСОВ — production-02, 2026-10-21
Пользователь явно отменил установленные агентом ограничения ресурсов.
Общего лимита CPU-часов, стоимости или срока завершения задачи больше нет.
Все описанные ниже бюджеты 576 CPU h, 64 CPU h и остаток 7.4 CPU h относятся
ТОЛЬКО к истории прежних попыток и больше не ограничивают продолжение.
Ограничение продолжения в 80 минут также отменено.
Действуют реальные Slurm/account limits и дисковая квота, которые агент
не может отменить. apollo разрешает до 20 часов на одно задание.
Новая схема: один расчёт сил на один job, 32 MPI ranks, KPAR=4, NCORE=4,
NBANDS=576 (как в исходном расчёте), 300 GiB RAM на узел. Для dense kmesh:
два узла, 64 ranks, 300 GiB на узел. CHGCAR сохраняется; WAVECAR пока
отключён из-за действительной дисковой квоты 100 GiB.
Научные параметры не изменены. Политика: config/execution-policy-02.json.
Первый этап force-001: PBE 20958089, PBEsol 20958090, r2SCAN 20958091.
Каталоги: <remote>/<XC>/production-02/<step>.
Следующие этапы запускать после проверки готовых через:
python phonons/scripts/continue_production.py submit <XC> <step>
Порядок: force-001, force-002, force-000, check-cutoff, check-kmesh,
check-smearing. Перед подачей сохраняются intents и хеши исходных входов.
Старые submit_recovery.py, retry_launcher.py и recover_forces.py — история,
не использовать их для текущей схемы. Готовые этапы не пересчитывать.
Мониторинг ti-dos снова ACTIVE, каждые 5 минут. Его инструкция обновлена:
старые бюджеты отменены; самостоятельно продолжать этапы и проверки
сходимости до общего графика DOS. Ошибки диагностировать до повторов.

ДАЛЕЕ — НАУЧНЫЙ ПРОТОКОЛ И ИСТОРИЯ ПОПЫТОК

Цель
Получить согласованные гармонические DFT-референсы для сравнения с MLIP.
Каждый функционал использует собственную равновесную hcp-ячейку при p=0.
Такое сравнение включает как влияние функционала на равновесную геометрию,
так и на силовые постоянные. Это не сравнение функционалов при фиксированном
объёме и не спектры с температурной ангармонической перенормировкой.

Расчёт
VASP 6.6.1 CPU, один и тот же PBE PAW Ti_pv (10 валентных электронов),
библиотека PAW 6.4; SHA256 зафиксирован в config/protocol.json.
PBE: GGA=PE; PBEsol: GGA=PS; r2SCAN: METAGGA=R2SCAN.
LASPH включён во всех случаях. Немагнитный расчёт, PREC=Accurate,
LREAL=False, ENCUT=550 eV, EDIFF=1e-8 eV, Gaussian sigma=0.1 eV.
Релаксация: 2 атома, Gamma 24x24x18, форма и объём ячейки свободны,
симметрия hcp сохраняется. Повторный запуск с новым плосковолновым базисом.
Статическая проверка: максимальная сила <=1e-4 eV/A, все компоненты
напряжения по модулю <=0.2 kbar. Максимум три прохода релаксации.
Силы: 4x4x3 (96 атомов), Gamma 6x6x6, ISYM=0, смещения +/-0.01 A,
симметрийно независимые направления phonopy. Отдельно идеальная сверхъячейка.
Контрольные расчёты первого смещения: ENCUT=650 eV, Gamma 8x8x8,
sigma=0.05 eV. Остальные параметры при каждой проверке неизменны.
Разница сил <=5e-4 eV/A — диагностический ориентир, а не доказательство
сходимости всего фононного спектра. Исходные силы сохраняются без коррекций.

После расчётов
Проверить все SCF/ionic gates, остаточные силы, диагностические разницы,
симметрию, силовые постоянные, акустические моды и возможные мнимые частоты.
Определить необходимость увеличения сверхъячейки: 4x4x3 пока не проверена
на сходимость. У MLIP в статье использовалась сверхъячейка 5x5x5.
При необходимости отдельно проверить амплитуду смещения и более плотную
k-сетку при уменьшенном smearing. Контроль одного смещения не заменяет
полную проверку сходимости DOS.
Будущая обработка: единая масса Ti=47.867 amu, q-сетка 50x50x50,
Gaussian width=0.05 THz, нормировка интеграла DOS к 3 на атом.
Сохранить результаты до/после коррекции остаточных сил и acoustic sum rule.
DOS и графики в этом запуске автоматически не строятся.

Эксплуатация
Кластер: umt.imm.uran.ru, пользователь u3129.
Каталог: /home6/u3129/tial-phonons/alpha-ti-xc-phonons-20261021-01
Окружение: /home6/u3129/tial-phonons/envs/phonopy-4.8.2
SSH использует существующий ключ и verified known_hosts из MLIP-research.
Ключи не копируются. POTCAR создаётся только на кластере штатным helper
vasp_potentials.py, проверяется по хешу и не включается в Git.
Slurm: apollo, imet-2026-2, normal; три задания по 16 CPU, 32 GB, 12 h.
Начальные задания: PBE 20957239, PBEsol 20957240, r2SCAN 20957241.
Установка: 20957235, COMPLETED, 46 CPU s.
Первая попытка (20957236--20957238) завершилась до запуска VASP: Intel
vars.sh изменял позиционный аргумент оболочки. Исправленный launcher
production-v2.sbatch сохраняет XC_NAME до загрузки окружения.
Неудачная попытка стоила суммарно 96 CPU s; исходные записи сохранены.
С учётом фактической установки и этих ошибок верхняя граница пакета
при полном использовании лимитов действующих заданий — 576.03945 CPU h.

Обновление 2026-10-21, около 17:00 Екатеринбург:
Все три начальных задания остановлены вручную после подтверждения OOM
на force-000. Запрос 32 GiB оказался недостаточным; оставшиеся MPI ranks
продолжали занимать CPU, хотя OUTCAR перестал обновляться. Релаксации,
повторная релаксация и статическая проверка равновесных ячеек завершены
корректно. Фононных сил и DOS из этой попытки нет.
Учтённый расход всех прошлых попыток: 504.826111 CPU h.
Продолжение: PBE 20957996, PBEsol 20957997, r2SCAN 20957998;
каждое 16 CPU, 192 GiB, 01:20:00. Сначала вычисляются две смещённые
структуры, затем несмещённая и электронные диагностики. Научные параметры
не изменены. Каталоги: <XC>/recovery-01/. Исходные неудачные расчёты сохранены.
srun использует --kill-on-bad-exit=1 и --wait=60. Автопереподачи нет.
Дополнительная максимальная резервация 64 CPU h; итоговый верхний предел
568.826111 CPU h, ниже первоначального 576.333334 CPU h.
Лимит продолжения не гарантирует завершения всех проверок; готовые этапы
сохраняются отдельно. Фактическое состояние смотреть в Slurm и в
<XC>/recovery-01/status.json, а не в устаревшем родительском status.json.
Обработка DOS возможна только после получения обеих смещённых структур;
для принятия спектров нужны оставшиеся диагностики и проверка размера ячейки.

Обновление 2026-10-21 18:43 Екатеринбург:
Recovery jobs 20957996/20957997/20957998 завершились TIMEOUT около 18:21.
Новых OOM не выявлено. PBE и PBEsol закончили только первую SCF-итерацию
(2258 и 2213 секунд соответственно); r2SCAN не завершил первую итерацию.
Готовых фононных сил/DOS нет. Причина низкой производительности ещё не
изолирована: 112 k-точек, 576 зон, 16 ranks, NCORE=4, KPAR=1.
Суммарно учтено 568.946111 CPU h, остаток прежнего предела 7.387223 CPU h.
Разница с прежней оценкой обусловлена завершением Slurm на несколько секунд
позже walltime. Фактический терминальный статус Slurm приоритетнее stale status.json.
Подробная диагностика: runs/alpha-ti-xc-phonons-20261021-01/recovery01-terminal-audit.json.
Периодический опрос ti-dos приостановлен: действующих фононных заданий нет.
Перед продолжением нужны измерение производительности и обоснованный новый
план ресурсов; повтор прежней схемы без изменений не выполнять.
Предельная резервация основного пакета — 576 CPU h плюс 1/3 CPU h
на установку. Фактический расход определяется sacct; это не прогноз времени.
Автоматической повторной подачи заданий нет. Этот пакет не расходует
и не изменяет отдельный учёт бюджета Al32 в MLIP-research.
run-id, input hashes, submission intents и job IDs: runs/<run-id>/.
Внутри каждого функционала status.json показывает текущий этап/ошибку.
COMPLETED Slurm не означает научную пригодность результата.

Команда просмотра (из корня TiAl, с доступом к локальному SSH-ключу):
python phonons/scripts/cluster.py exec 'squeue -u u3129'
python phonons/scripts/snapshot.py

snapshot.py сохраняет небольшие входные файлы, хеши, версии и статусы.
Для новых запусков использовать исправленный production-v2.sbatch.
retry_launcher.py предназначен только для зафиксированного исправления
первой попытки; повторно его запускать нельзя.

Воспроизводимость
Не перезаписывать существующие расчёты. Новый протокол — новый run-id.
Для длительного хранения забрать исходные VASP-выходы и входы в private
архив, сверить SHA256 с output-manifest.json; /home6 не считать резервной копией.
До завершения расчётов доступны проверяемые входы, lock окружения и журнал
подачи. Лицензированные данные не публиковать.

Официальные источники параметров и интерфейсов (проверены 2026-10-21):
https://phonopy.github.io/phonopy/vasp.html
https://phonopy.github.io/phonopy/phonopy-module.html
https://vasp.at/wiki/index.php/GGA
https://vasp.at/wiki/index.php/METAGGA

Связь с рукописью
Новый пакет не заменяет старый PBE-референс в manuscript до проверки результатов.
Старый референс: 4x4x3, 500 eV, Gamma 4x4x4, sigma=0.2 eV.
Новый протокол согласован между функционалами и требует собственного аудита.
