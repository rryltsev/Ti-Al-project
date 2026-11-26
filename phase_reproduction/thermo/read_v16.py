"""Recover numerical targets and 1000 K harmonic interactions from the supplied v16 DOCX pair."""
import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def word_tables(path):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return [
        [["".join(t.text or "" for t in c.findall(".//w:t", ns))
          for c in row.findall("w:tc", ns)] for row in table.findall("w:tr", ns)]
        for table in root.findall(".//w:tbl", ns)]


def axis_calibration(pixels, tick_x, area, values):
    rows = np.where(np.all(pixels[area, tick_x] < 60, axis=1))[0] + area.start
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    if len(groups) != len(values):
        raise ValueError("Unexpected Figure 2 axis ticks")
    return np.polyfit([g.mean() for g in groups], values, 1)


def bar_height(pixels, x, fit, is_static=False):
    rgb = pixels[195:634, x].astype(int)
    if is_static:
        color = (rgb[:, 0] < 100) & (rgb[:, 1] > 70) & (rgb[:, 2] > 160)
    else:
        color = (rgb[:, 0] > 170) & (rgb[:, 0] > 2*rgb[:, 1]) & (rgb[:, 2] < 160)
    rows = np.where(color)[0] + 195
    if len(rows) == 0:
        return 0.0
    zero = -fit[1]/fit[0]
    edge = rows.min() - 0.5 if rows.mean() < zero else rows.max() + 0.5
    return float(np.polyval(fit, edge))


def extract(pd):
    main, = pd.glob("*v16_статья.docx")
    supplement, = pd.glob("*v16_дополнительные_материалы.docx")
    table = word_tables(supplement)[0]
    names = (["constant", "point"] + [f"p{i}" for i in range(1, 8)] +
             ["t112", "t114", "t117", "t123", "t125", "t136", "t222_1",
              "t222_2", "t226", "t247", "t334"])
    static = np.array([float(row[2]) for row in table[1:]])
    expected_names = ["Постоянный", "Одноузельный", *names[2:15], "222, тип 1",
                      "222, тип 2", *names[17:]]
    if [r[0] for r in table[1:]] != expected_names:
        raise ValueError("Unexpected S1 orbit sequence")
    with ZipFile(main) as archive:
        raw = archive.read("word/media/image2.png")
        fig5raw = archive.read("word/media/image5.png")
    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    if pixels.shape[:2] != (840, 2100):
        raise ValueError("Figure 2 calibration requires the native 2100 x 840 raster")
    pair_fit = axis_calibration(pixels, 199, slice(195, 634), [.04, .02, 0])
    trip_fit = axis_calibration(pixels, 1100, slice(195, 634), [.004, 0, -.004, -.008])
    b_centers = [286, 373, 460, 549, 635, 722, 810, 1200, 1282, 1364, 1446,
                 1528, 1610, 1692, 1774, 1856, 1938, 2020]
    vib = np.zeros(20)
    static_est = static.copy()
    for i, red_x in enumerate(b_centers, 2):
        fit = pair_fit if i <= 8 else trip_fit
        vib[i] = bar_height(pixels, red_x, fit)
        static_est[i] = bar_height(pixels, red_x - 32, fit, True)
    method = dict(orbit_names=names, static_S1_eV=static.tolist(), T_K=1000,
                  fig2_static_pixel_eV=static_est.tolist(), fig2_vib_pixel_eV=vib.tolist(),
                  effective_eV=(static + vib).tolist(),
                  gauge="J0,J1 vibrational terms set to zero. At fixed volume this preserves tangent compositions; absolute free energies and exchange chemical potentials can shift.",
                  eV_per_pixel=dict(pairs=abs(pair_fit[0]), triplets=abs(trip_fit[0])),
                  sources_sha256={main.name: sha(main.read_bytes()), supplement.name: sha(supplement.read_bytes()),
                                  "Figure 2 image2.png": sha(raw), "Figure 5 image5.png": sha(fig5raw)})
    return method


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pd", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(extract(args.pd), ensure_ascii=False, indent=2)+"\n", encoding="utf8")
