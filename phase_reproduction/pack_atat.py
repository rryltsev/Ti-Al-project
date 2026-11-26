"""Bundle native ATAT cells, their MTP inputs, the design matrix and fitted results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("design",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--fit",type=Path,action="append",default=[])
    p.add_argument("--large-source",type=Path,help="directory containing the 64-site ATAT generator and native occupations")
    p.add_argument("--replay", action="store_true",help="include direct MTP labels, native fit checks and code to refit ECI")
    args = p.parse_args()
    design = args.design
    with (design/"design.csv").open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows)!=len({row["id"] for row in rows}):
        raise ValueError("empty design or repeated ID")
    for row in rows:
        cfg = design/row["cell"]
        if digest(cfg.read_bytes())!=row["cell_sha256"]:
            raise ValueError(f"CFG hash mismatch for {row['id']}")
    payload = {}
    for file in sorted(design.rglob("*")):
        if file.is_file():
            payload["design/"+file.relative_to(design).as_posix()] = file
    for directory in args.fit:
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                name = "fit/"+directory.name+"/"+file.relative_to(directory).as_posix()
                if name in payload:
                    raise ValueError(f"duplicate archive member {name}")
                payload[name] = file
    if args.large_source:
        for relative in ["generate.py","job.sh","binaries.sha256","cells64/generation.json"]:
            payload["generation64/"+relative] = args.large_source/relative
        for path in sorted((args.large_source/"cells64").glob("*/*.in")):
            payload["generation64/"+path.relative_to(args.large_source).as_posix()] = path
        for path in sorted((args.large_source/"cells64").glob("*/sqscell.out")):
            payload["generation64/"+path.relative_to(args.large_source).as_posix()] = path
    if args.replay:
        root = Path(__file__).resolve().parent
        for name in ["atomistic/fit_eci.py","atomistic/hcp_orbits.py","atomistic/mtp_cfg.py","atomistic/label_cell.py",
                     "atomistic/atat/export_eci.py","atomistic/atat/select_cells.py","cluster/native-fit/fit_native.py",
                     "table_eci.py","table_labels.py"]:
            payload["code/"+name] = root/name
        for name in ["MTP_labels_414.csv","ECI_TV.csv","eci_vs_S1.csv","transfer_summary.json","static_fit.png"]:
            payload["tables/"+name] = root/"atat_comparison"/name
        for name in ["fitted.mtp","source.json"]:
            payload["potential/"+name] = root/"inputs"/name
        for name in ["eci_thermal_grid.csv","eci_static_unmasked.csv","native_validation.csv","input_archives.json"]:
            payload["native_lsfit/"+name] = root/"cluster/native-fit/tables"/name
        for percent in (94,96,98,100,102,104,106,109,112):
            attempt = "02" if percent==100 else "01"
            payload[f"labels/v{percent:03d}.tar.gz"] = root/f"cluster/runs/atat-v{percent}-{attempt}/outputs/labels.tar.gz"
        for mode in ("static","vref","volumes"):
            payload[f"native_lsfit/{mode}.tar.gz"] = root/f"cluster/runs/atat-verify-{mode}-02/outputs/fits.tar.gz"
    if not any(k.endswith("/str.out") for k in payload) or not any(k.endswith("/clusters.out") for k in payload):
        raise ValueError("missing native ATAT structure or cluster file")
    tmp = args.output.with_suffix(".part")
    manifest = {}
    with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=6) as z:
        for key,path in payload.items():
            raw = path.read_bytes()
            z.writestr(key,raw)
            manifest[key] = digest(raw)
        readme = [
            "Ti–Al • ATAT 3.52 • CPU MTP Qi et al.",
            "",
            "design/lat.in — родительская ГПУ-решётка для ATAT.",
            "design/clusters.out — 20 орбит в собственном порядке ATAT.",
            "design/design.csv — 39 обучающих и 7 контрольных ячеек, признаки нормированы на атом.",
            "design/selected/<id>/str.out — химическая ячейка в координатах ATAT.",
            "design/selected/<id>/corr.out — сырые корреляции ATAT выбранной ячейки.",
            "design/cells/<id>.cfg — та же декорация при физическом опорном объёме для MTP.",
            "design/jobs.csv — сетки объёмов, суперячеек и q-точек.",
            "design/native/S1_round_eV_atom.eci — только ориентир из печатной таблицы S1.",
            "",
        ]
        if args.fit:
            readme.append("Результаты собственной подгонки и экспорта помещены в fit/.")
        if args.replay:
            readme.extend(["","Воспроизведение подгонки при опорном объёме из каталога распакованного архива (Python 3.10+, NumPy):",
                "python code/atomistic/fit_eci.py design labels/v100.tar.gz fitted_vref --ratios 1",
                "python code/atomistic/atat/export_eci.py design fitted_vref/grid.npz native_HA1000 --temperature 1000",
                "Для проверки нативным ATAT добавьте к последней команде --corrdump /absolute/path/to/corrdump.",
                "Таблица tables/MTP_labels_414.csv включает прямую статику, фононный флаг устойчивости и 49 значений Fvib для устойчивых состояний.",
                "Таблица tables/ECI_TV.csv содержит Jstatic, Jvib и их сумму для 20 орбит на девяти объёмах при 49 температурах.",
                "В native_lsfit/eci_thermal_grid.csv собраны коэффициенты из нативной программы ATAT lsfit. Там же сохранены матрицы и проверка corrdump с UMT.",
                "В potential/ находится исходный MTP Qi с контрольной суммой и ссылкой на первоисточник."])
        z.writestr("README_RU.txt","\n".join(readme)+"\n")
        z.writestr("SHA256.json",json.dumps(manifest,indent=2)+"\n")
    with ZipFile(tmp) as z:
        if z.testzip() is not None or any(digest(z.read(k))!=h for k,h in manifest.items()):
            raise ValueError("archive verification failed")
    tmp.replace(args.output)
    print(f"{len(rows)} chemical cells, {len(payload)} files, SHA-256 {digest(args.output.read_bytes())}")


if __name__=="__main__":
    main()
