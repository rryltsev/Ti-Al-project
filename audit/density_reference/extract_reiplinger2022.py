"""Extract published Fig. 4 symbols/error bars from its PDF vector objects.

These are graph-digitized measurements, not raw experimental data. Path indices
refer only to the archived publisher PDF identified by the recorded SHA256.
"""
from pathlib import Path
import hashlib
import json
import pypdfium2 as pdfium
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
source = HERE / "Reiplinger2022.pdf"
doc = pdfium.PdfDocument(source)
page = doc[5]
paths = [obj for obj in page.get_objects() if obj.type == 2]

def center(index):
    x0, y0, x1, y1 = paths[index].get_bounds()
    return (x0+x1)/2, (y0+y1)/2

x0, x1 = center(283)[0], center(290)[0]
y0, y1 = center(297)[1], center(301)[1]
temperature = lambda x: 1800 + (x-x0)/(x1-x0)*350
density = lambda y: 4.0 + (y-y0)/(y1-y0)*0.4
points = []
image = page.render(scale=3).to_pil()
draw = ImageDraw.Draw(image)
for i in range(25):
    x, y = center(436+i)
    xu, yu = center(304+4*i)
    xl, yl = center(306+4*i)
    assert abs(x-xu) < .05 and abs(x-xl) < .05
    assert yl < y < yu
    points.append(dict(T_K=temperature(x), rho_g_cm3=density(y),
                       lower_g_cm3=density(yl), upper_g_cm3=density(yu),
                       marker_path=436+i, upper_cap_path=304+4*i,
                       lower_cap_path=306+4*i))
    for xx, yy in [(x, y), (xu, yu), (xl, yl)]:
        px, py = 3*xx, 3*(page.get_height()-yy)
        draw.line((px-3, py, px+3, py), fill="red", width=1)
        draw.line((px, py-3, px, py+3), fill="red", width=1)

result = dict(
    citation="Reiplinger and Brillo, J. Mater. Sci. 57, 7954–7964 (2022), Fig. 4",
    doi="10.1007/s10853-022-07090-2",
    url="https://elib.dlr.de/186260/1/Reiplinger-Brillo2022_Article_DensityAndExcessVolumeOfTheLiq.pdf",
    source_file=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    pdf_page=6, printed_page=7959,
    extraction="Centers of filled-square vector paths and horizontal error-bar caps; affine axis calibration",
    uncertainty="Published experimental error bars; no confidence level assigned here. Digitization is additional to the reported measurement uncertainty.",
    calibration=dict(x_pdf=[x0,x1], T_K=[1800,2150], y_pdf=[y0,y1], rho_g_cm3=[4.0,4.4]),
    points=sorted(points, key=lambda p: p["T_K"]))
(HERE / "reiplinger2022_points.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
image.crop((210, 1050, 870, 1580)).save(HERE / "reiplinger2022_vector_check.png")
print(f"Extracted {len(points)} measurements; {sum(p['T_K'] >= 1941 for p in points)} at or above 1941 K.")
print(f"Temperature range: {min(p['T_K'] for p in points):.1f}–{max(p['T_K'] for p in points):.1f} K")
