import json,shutil,tomllib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sub=json.loads((ROOT/'submission.json').read_text())
assert sub['job_id']=='20961934'
shutil.copy2(ROOT/'submission.json',HERE/'submission-20961934.json')
shutil.copy2(ROOT/'submission-intent.json',HERE/'submission-intent-20961934.json')
sub.update(job_id='20962153',unix=1793103010.4266553,previous_job_id='20961934',reason='Replacement after corrected preflight assertion; original job purged; requeue refused')
(ROOT/'submission.json').write_text(json.dumps(sub,indent=2)+'\n')
(HERE/'replacement-submission.json').write_text(json.dumps(sub,indent=2)+'\n')
mon=json.loads((ROOT/'monitor.json').read_text())
mon.update(job_id='20962153',previous_failed_job_id='20961934',repair='repairs/type-map-20261027',gpu_preflight='Type-map assertion corrected; full GPU preflight still required in replacement job.')
(ROOT/'monitor.json').write_text(json.dumps(mon,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if (ROOT/'collected/failure.json').exists():
    (ROOT/'collected/failure.json').rename(HERE/'failure-20961934.json')
note='''
27.10.2026: задача 20961934 завершилась FAILED до начала MD из-за ошибочного
требования бинарного type_map. DPA4 содержит 118 элементов; индексы Al=12,
Ti=21 подтверждены. Проверка исправлена на наличие и уникальность требуемых
элементов; сравнение энергий/сил/вириалов на GPU сохранено. Физический протокол
не менялся. Исходный скрипт и ошибка: repairs/type-map-20261027/.
Slurm удалил старый job из активного учёта и отклонил requeue. После проверки
отсутствия активного дубля подана единственная замена 20962153. Текущий job ID
всегда читать из submission.json. Завершённых реплик на момент замены: 0.
'''
with (ROOT/'README_RU.txt').open('a',encoding='utf-8') as fp:fp.write(note)
auto=tomllib.loads(Path('C:/Users/Roman Ryltsev/.codex/automations/pbe-15k-ti-al/automation.toml').read_text(encoding='utf-8'))
prompt=auto['prompt'].replace('Единственная задача Slurm 20961934 использует одну A100;', 'Текущая единственная задача Slurm 20962153 использует одну A100; актуальный ID читать из submission.json. Она заменила завершившуюся FAILED 20961934 после исправления стартовой проверки type_map (исходный job был удалён Slurm и requeue отклонён); физический протокол не менялся;')
payload=dict(mode='update',id=auto['id'],kind=auto['kind'],destination='thread',name=auto['name'],prompt=prompt,status=auto['status'],rrule=auto['rrule'],targetThreadId=auto['target_thread_id'])
(HERE/'automation-update.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Local recovery record and current job ID updated.')
