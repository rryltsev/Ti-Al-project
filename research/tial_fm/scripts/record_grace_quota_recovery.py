"""Document quota failure, verified archives, preserved results and retry chain."""
import json
from pathlib import Path
from prepare_grace_mask_ablation import ROOT,RUN,REMOTE

def load(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    rec=RUN/'storage-recovery';released=load(rec/'released.json');deps=load(rec/'dependencies.json');status=load(RUN/'latest-status.json');plan=load(RUN/'package/plan.json');archives=load(rec/'verified-archives.json')
    state=load(ROOT/'state.json');matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for p in [ROOT/'state.json',ROOT/'protocol/experiment_matrix.yaml']:
        backup=rec/(p.name+'.before');
        if not backup.exists():backup.write_bytes(p.read_bytes())
    state.update(status='GRACE_mask_ablation_quota_recovered',stage='GRACE_mask_ablation_three_retries',updated_utc=status['utc'],active_training_jobs=3,completed_training_jobs=29)
    state['grace_mask_ablation'].update(quota_recovery=dict(released_bytes=released['bytes_released'],local_archives=len(archives),preserved_success='PBE-restricted_mlp-N200-seed10',restart_indices=[1,2,3],jobs={k:v for k,v in deps.items() if k!='states'},storage_policy='No unused best-validation checkpoint; regular optimizer checkpoints retained; quota checks before start/save/export.'),optimizer='PASS')
    state['grace_mask_ablation']['jobs'].update(storage_probe=deps['storage_probe'],train_quota_retry=deps['retry'])
    state['current_gates']['GRACE_storage_probe']=status['files'].get('storage-probe.json',{}).get('status','PENDING')
    state['next_actions']=['Check storage probe and retry array; preserve the successful PBE+MLP result.','Collect all four phonons and comparison after the three fresh-start retries complete.','Do not reuse interrupted epoch-one optimizer states; scientific settings unchanged.']
    report='reports/24_grace_quota_recovery.md'
    if report not in state['reports']:state['reports'].append(report)
    for i,c in enumerate(plan['cases']):
        d=next(d for d in matrix['training_designs'] if d['architecture']=='GRACE-2L' and d['train_size']==200 and d['functional']==c['XC'] and d['adaptation']==c['method'])
        if i==0:d.update(state='training_and_validation_complete_pending_phonons',blocked_by=['GRACE_mask_harmonics'])
        else:d.update(state='retry_submitted_after_disk_quota_failure',failed_slurm_array_job='20990675',slurm_array_job=deps['retry'],slurm_array_index=i,blocked_by=['GRACE_storage_probe'],restart='fresh aligned source and fresh Adam')
    dump(ROOT/'state.json',state);dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    lines=['# GRACE: восстановление после исчерпания квоты','',f'Снимок: {status["utc"]}.','',
        '20990673_0–3 и optimizer gate20990674 завершились PASS. PBE restricted+MLP (20990675_0) прошёл10000 шагов и final validation/QC PASS. Остальные три элемента массива20990675 остановились после200 шагов при первой записи checkpoint.best_test_loss: ResourceExhaustedError / Disk quota exceeded.','',
        'Причина: квота аккаунта на /home6 —100GiB; свободно было около430MiB, тогда как один checkpoint с состоянием Adam —512–522MB. Общая свободная ёмкость файловой системы не отражает доступную пользовательскую квоту. В первоначальном запуске отсутствовали quota-aware preflight и проверка записи checkpoint в smoke.','',
        f'Промежуточные checkpoint {len(archives)} завершённых GRACE-моделей перенесены в локальные .tar.gz архивы. Для каждого исходного файла проверены размер и SHA-256 внутри локального архива, перед удалением повторно проверены source SHA-256 и сохранность final checkpoint. Освобождено {released["bytes_released"]:,} bytes ({released["bytes_released"]/2**30:.3f}GiB). На UMT в папках checkpoints сохранены LOCAL_ARCHIVE.json с точным местом копии и hashes. Финальные модели, экспорт, исходные веса, данные и результаты не удалялись.','',
        'В повторах отключена только запись неиспользуемого best-validation checkpoint. Протокол изначально выбирает fixed final epoch50; best-reload уже был отключён. Periodic checkpoint с полным состоянием optimizer каждые5 эпох сохраняется. Схема Adam, LR, loss, batch order, seed, split и маски не менялись.','',
        'Добавлены проверки пользовательской квоты: минимум1.5GiB до старта и640MiB перед periodic checkpoint/final export. Отдельный CPU probe требует минимум4GiB, реально записывает512MiB и проверяет fsync; временный probe-файл удаляется. Эти проверки обнаружат новые ограничения квоты, но не резервируют место от других задач аккаунта.','',
        'Неудачные попытки целиком сохранены в recovery/disk-quota-v1/cases. Все три повтора начинают со step0 и fresh Adam, чтобы сохранить тот же протокол; успешная PBE+MLP модель не переобучается.','',
        '| Этап | Job | Условие |','|---|---|---|',f'| Проверка квоты и записи512MiB | {deps["storage_probe"]} | CPU |',f'| Повторы индексов1–3 | {deps["retry"]} | storage probe afterok |',f'| Фононы всех4 моделей | {deps["harmonics"]} | retry afterok; успешный index0 сохранён |',f'| Сравнение с full/restricted | {deps["comparison"]} | фононы afterok |','',
        '```',status['queue'].strip(),'```','',
        'Подробный manifest архивов: storage-recovery/archive-plan.json, verified-archives.json, released.json. Изменения кода и политики: storage-recovery/repair.json и new-files/. Старые код и manifest сохранены в remote recovery/disk-quota-v1/.']
    (ROOT/report).write_text('\n'.join(lines)+'\n',encoding='utf8')
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nGRACE mask tests: PBE+MLP завершён; три запуска восстанавливаются после Disk quota exceeded. Промежуточные checkpoint завершённых моделей архивированы локально с SHA-256. Новая цепочка '+deps['storage_probe']+' → '+deps['retry']+' → '+deps['harmonics']+' → '+deps['comparison']+'.\n\n[Восстановление квоты](24_grace_quota_recovery.md) · [План масок](23_grace_mask_ablation_launch.md) · [Результаты full/restricted](22_grace_n200_results.md).\n',encoding='utf8')
    print(json.dumps(dict(report=report,released_GiB=released['bytes_released']/2**30,jobs={k:v for k,v in deps.items() if k!='states'})))

if __name__=='__main__':main()
