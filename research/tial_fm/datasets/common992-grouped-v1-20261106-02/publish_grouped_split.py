"""Publish a verified split and update project pointers; keep original history."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
VERSION='common992-grouped-v1-20261106-02'
DATA=ROOT/'datasets'/VERSION
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def md(p,t):p.write_text(t.strip()+'\n',encoding='utf8')

def main():
    v=read(DATA/'verification.json');assert v['status']=='PASS'
    splits=read(DATA/'splits.json');summary=read(DATA/'summary.json');split_hash=digest(DATA/'splits.json')
    dest=ROOT/'manifests'/VERSION;dest.mkdir(exist_ok=False)
    for name in ['splits.json','groups.json','data_manifest.jsonl','split_assignments.csv','verification.json','summary.json','input_hashes.json']:
        shutil.copyfile(DATA/name,dest/name)
    history=read(ROOT/'runs/initial-audit-20261106-01/splits.json')
    history['new_grouped_benchmark']={'state':'frozen_verified_conservative_groups','version':VERSION,'split_sha256':split_hash,'path':str(dest/'splits.json'),'counts':v['counts'],'definition':splits['group_definition'],'qualification':splits['guarantee'],'archive49_policy':splits['archive49_policy']}
    write(ROOT/'manifests/splits.json',history)
    shutil.copyfile(DATA/'data_manifest.jsonl',ROOT/'manifests/data_manifest.jsonl')
    shutil.copyfile(DATA/'split_assignments.csv',ROOT/'tables/split_assignments.csv')
    initial_matrix=ROOT/'protocol/experiment_matrix_initial_20261106.yaml'
    if not initial_matrix.exists():shutil.copyfile(ROOT/'protocol/experiment_matrix.yaml',initial_matrix)
    matrix=read(initial_matrix);matrix['active_split']={'version':VERSION,'sha256':split_hash,'counts':v['counts']}
    for d in matrix['training_designs']:
        d['split_hash']=split_hash;d['split_manifest']=str(dest/'splits.json');d['dataset_version']=VERSION
        d['validation_geometry_ids']=splits['split_ids']['validation']
        d['validation_systems']=str(DATA/'deepmd'/d['functional']/'validation_systems.json')
        d['blocked_by']=[x for x in d['blocked_by'] if x!='grouped_split_and_parent_provenance']
        if d['train_size']=='full_grouped_train':
            d['train_size']=694;d['target_geometry_ids']=splits['split_ids']['train'];d['training_systems']=str(DATA/'deepmd'/d['functional']/'train_systems.json')
        else:d['blocked_by'].append('nested_N200_train_selection')
        d['next_step']='Complete remaining runtime/physics/allocation gates. All methods require retraining from their declared initialization; legacy TiAl FT cannot be an independent baseline on this test.'
        write(ROOT/'configs'/VERSION/(d['run_id']+'.json'),d)
    matrix['scientific_qualification']='Fixed-atom-count parent containment split; exact lost trajectories not reconstructed. Feasible balanced assignment frozen; global MILP optimum not claimed.'
    write(ROOT/'protocol/experiment_matrix.yaml',matrix)
    state=read(ROOT/'state.json');state['stage']='grouped_split_prepared_and_verified';state['status']='data_split_ready_with_documented_parent_containment'
    state['active_dataset']={'version':VERSION,'root':str(DATA),'split_sha256':split_hash,'manifest_sha256':digest(DATA/'data_manifest.jsonl'),'counts':v['counts'],'verification':str(DATA/'verification.json'),'paired_label_records':2976,'guarantee_scope':v['guarantee_scope'],'exact_liquid_trajectory_ids_recovered':False,'old_test49':'archival_diagnostic_only'}
    state['next_actions']=['Select nested N100/N200/N400 subsets within the frozen train694 only','Validate current runtime masks and baseline derivatives/exports','Evaluate original FM harmonic response on frozen DFT geometries','Resolve allocation and profile first bounded training; compare newly trained models on this split']
    state['training_launch_ready']=False;state['new_training_jobs']=0
    state['reports'].append('reports/07_grouped_split.md');write(ROOT/'state.json',state)
    lines=[]
    for part in ['train','validation','test']:
        c=summary['counts'][part];f=c['source_families']
        lines.append(f"| {part} | {c['geometries']} | {c['groups']} | {f['liquid_relax']} | {f['hcp']} | {f['bcc']} | {f['gamma']} |")
    table='\n'.join(lines)
    report=f'''# Единое разбиение Ti–Al — 6 ноября 2026

Версия: `{VERSION}`. Статус: **создано и проверено**, без обучения и новых DFT.
Одинаковые ID используются для PBE, PBEsol и r²SCAN. Предыдущие 49 состояний по решению пользователя оставлены отдельным архивным benchmark и не являются test нового разбиения.

| Часть | Геометрии | Группы | MD/relax liquid source | hcp source | bcc source | γ source |
|---|---:|---:|---:|---:|---:|---:|
{table}

## Что означает «по родителям» при утраченных траекториях

Пользователь подтвердил: жидкостный пул получен чередованием MACE MD и relaxation, исходные траектории, вероятно, утрачены. Числа Ti и Al в MD/relaxation не меняются. Скрипт генерации меняет состав **до** запуска отдельной MD; дальнейшая динамика и FIRE не меняют химические элементы и число атомов.

Поэтому группой служит точная пара **(nTi,nAl)**, одинаковая для всех семейств и XC. Все кадры одной описанной траектории и её релаксационные продолжения неизбежно лежат в одной группе. Одинаковый состав у нескольких разных траекторий приводит к их консервативному объединению. Равенство групп не объявляется доказательством общего точного trajectory ID.

Это **разбиение по группам, гарантированно содержащим родителей**, а не восстановленный реестр утраченных траекторий. Гарантия относится к описанному протоколу с постоянными nTi/nAl. Она не распространяется на недокументированную трансмутацию, вставку/удаление атомов или общее происхождение химически заменённых начальных структур из более раннего предка.

В исходном пуле140 групп, в paired benchmark139: одна группа имеет только неполные метки. У 44 установленных родительских траекторий нет пересечений между частями. На всём пуле1000, включая восемь неполных конфигураций, пересечений групп также нет.

Следствие консервативного подхода: точные пары nTi/nAl не повторяются между train/validation/test. Проверяется также перенос между составами, а не только между траекториями при одном составе. Чистые Al216 и Ti216 помещены в train; все четыре семейства и пять укрупнённых интервалов состава присутствуют в каждой части. Редкий hcp представлен всего двумя геометриями в validation и двумя в test — выводы по нему будут ограниченными.

## Данные и контроль

Из 1000 исходных структур 992 имеют все три функционала; в экспорт вошло **2976 geometry–XC записей**. Восемь неполных структур не удалены: их group и split сохранены, но они исключены из парного benchmark. Harmonic reference в этот split не включён.

Начальные POSCAR проверены в `TiAl-Project/MACE-dataset-gen/MD-liquid+relax/RESULT`: найдено220. Для24 выбранных liquid composition-групп (247 геометрий) соответствующий начальный состав в этом архиве отсутствует. Поэтому ни совпадение состава с POSCAR, ни его отсутствие не используется как выдуманный exact parent ID; безопасная группировка определяется сохранением состава в самом MD/relaxation. Первый подготовительный запуск с излишним требованием полноты initial archive завершился до разбиения/экспорта; запись failure сохранена отдельно.

Распределение построено только по ID, размерам, составам и семействам. Энергии, силы и ошибки моделей при выборе split не использовались. Цели694/149/149 выполнены точно. Seed={SEED if False else 20261106}. MILP остановлен по лимиту30s с допустимым решением и gap около5.04%; глобальная оптимальность баланса не утверждается. Дальше используются замороженные ID, повторная оптимизация не нужна.

Независимая проверка: ноль пересечений geometry ID, conservative groups и known parents; одинаковые списки ID всех XC; **каждая** из2976 экспортированных записей сравнена с исходными массивами; проверены4676 source hashes и хеши всех файлов экспорта. Координаты/ячейки сохранены в исходной численной точности; выполнена только стабильная сортировка Al→Ti с той же перестановкой сил. Энергии и virial не изменялись. `verification.json`: PASS.

## Файлы

- IDs и правила: `manifests/{VERSION}/splits.json`.
- Полная таблица1000 строк: `tables/split_assignments.csv`.
- Группы и известные/candidate parents: `manifests/{VERSION}/groups.json`.
- DeepMD: `datasets/{VERSION}/deepmd/<XC>/<train|validation|test>/`.
- Для каждой части: `<part>_systems.json` со списком систем, в каждой системе `geometry_ids.json` с точным порядком кадров.
- Export lineage и перестановки: `datasets/{VERSION}/export_index.json`.
- Проверка: `datasets/{VERSION}/verification.json`.
- SHA-256 split: `{split_hash}`.

Размер DeepMD export около31.1 MiB. Крупные массивы исключены из Git. Оригинальные datasets, historical splits, checkpoints и результаты статьи не менялись.

## Использование в экспериментах

Полный допустимый train теперь694, development149, test149. Для одного функционала доступ к train+development означает843 DFT-метки; paired обучение трёх XC использует2529 таких меток. Test labels считаются отдельно (149 на XC). N200 будет подмножеством только train694; его выбор ещё не выполнен.

Старые Ti–Al FT checkpoints видели многие геометрии нового test. Их метрики не могут служить независимым baseline этого разбиения. Сравниваемые методы необходимо переобучить от одного исходного pretrained checkpoint (или отдельного random-init контроля) и оценивать на одном новом split. Старые49 могут использоваться только как историческая диагностика; их parent overlap с новым train не скрывается.

Оставшиеся B0/H0/runtime/budget gates не отменяются созданием split. Численная идентичность экспортированных labels проверена; новый аудит SCF каждого первичного VASP-результата этим не заменяется.
'''
    md(ROOT/'reports/07_grouped_split.md',report)
    old=ROOT/'reports/summary_initial_20261106.md'
    if not old.exists():shutil.copyfile(ROOT/'reports/summary.md',old)
    md(ROOT/'reports/summary.md',f'''# Ti–Al FM: текущее состояние

6 ноября2026: создано единое разбиение **694 train / 149 validation / 149 test** для992 общих геометрий PBE/PBEsol/r²SCAN. DeepMD-наборы готовы;2976 записей проверены против исходников, пересечений known parents и консервативных групп нет.

Группировка использует сохранение(nTi,nAl) при MD/relaxation, поскольку exact liquid trajectories, вероятно, утрачены. Она консервативна и не выдаётся за восстановление trajectory/frame IDs. Прежний test49 оставлен архивной диагностикой по выбору пользователя.

[Полный отчёт](07_grouped_split.md) · [Таблица ID](../tables/split_assignments.csv) · [Разбиение](../manifests/{VERSION}/splits.json).

Полный train для новых сравнений:694. Старые FT checkpoints не являются независимым baseline нового test; требуется переобучение. B0/H0, runtime mask, N200 subset и бюджет остаются отдельными этапами. Новых DFT и обучения:0.
''')
    protocol=ROOT/'protocol/benchmark_protocol.md';text=protocol.read_text(encoding='utf8')
    text=text.replace('benchmark protocol v0.1','benchmark protocol v0.2').replace('Статус: предварительный протокол; новый обучающий split ещё не выпущен.',f'Статус: split {VERSION} выпущен и проверен; runtime/physics gates остаются открытыми.')
    text=text.replace('- Новый split строится по родителям и кластерам почти совпадающих геометрий. Сначала исключить родителей внешних структур, затем stratified group split/folds. Не подменять независимость случайным разделением кадров.','- Новый split:694/149/149, по консервативным группам(nTi,nAl), содержащим целиком каждую fixed-species MD/relaxation trajectory. Exact/near duplicate states одинакового числа атомов и состава не могут пересекать части. Точные liquid parent IDs не восстановлены. По решению пользователя прежние49 вынесены из нового test и остаются архивной диагностикой; их родители не принудительно исключаются из train.')
    text=text.replace('N=100/200/400 и полный train вводятся после установления допустимого размера. 850 не является обязательным размером нового train.','Полный train=694; N=100/200/400 строятся только внутри него. N=850 для этого split недоступно.')
    text=text.replace('Test49 — фиксированный ранее изученный benchmark','Test49 — отдельный архивный ранее изученный benchmark')
    protocol.write_text(text,encoding='utf8')
    with (ROOT/'protocol/decision_log.md').open('a',encoding='utf8') as f:f.write('\n12. Пользователь подтвердил MACE MD+relaxation происхождение liquid pool и вероятную утрату исходных trajectories. Выбрал новое разбиение с Test49 как архивным benchmark.\n13. Выпущен split694/149/149 по fixed-species containment groups(nTi,nAl), общий для трёх XC. Точные parent IDs не выдумываются; тест включает перенос между точными составами. Все2976 экспортированных records проверены. Data grouping gate закрыт в этих явно указанных границах; B0/H0 и training budget остаются открытыми.\n')
    # Freeze all version files, including actual source copies, with relative names.
    shutil.copyfile(Path(__file__),DATA/'publish_grouped_split.py')
    shutil.copyfile(ROOT/'scripts/verify_grouped_split.py',DATA/'verify_grouped_split.py')
    manifest={str(p.relative_to(DATA)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in DATA.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
    write(DATA/'FILES_SHA256.json',manifest)
    write(ROOT/'manifests/active_dataset.json',state['active_dataset'])
    print(json.dumps({'status':'published','version':VERSION,'counts':v['counts'],'split_sha256':split_hash,'version_files':len(manifest)},indent=2))

if __name__=='__main__':main()
