"""
Validate extracted time-series CSVs against the ONLY ground truth: the images.
For every plot, round-trips points.csv back to pixel space and rings each
point on the original (coloured per well), then runs internal consistency
checks. If the rings sit on the dots, the extraction is faithful.

Mirrors the extractor's layout: CSVs live in  <outdir>/<file_name>/  and the
overlay is written back into that same subfolder.

Usage:
  python validate_timeseries.py                       # default folder + outdir
  python validate_timeseries.py <plots_folder> <outdir>
"""
import cv2, numpy as np, pandas as pd, sys, os
from plot_extraction.parse_time_plots import (load, find_axes_box, calibrate,
                                      date_to_ord, TAB10)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def inverse(fn):                       # invert a linear pixel->value map
    b = fn(0); a = fn(1) - b           # value = a*px + b  ->  px = (value-b)/a
    return lambda val: (val - b) / a

def validate_one(png_path, csv_dir):
    bgr, _ = load(png_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    L, R, T, B = find_axes_box(gray)
    x2ord, y2psi = calibrate(gray, L, R, T, B)
    ord2x, psi2y = inverse(x2ord), inverse(y2psi)

    pts = pd.read_csv(f"{csv_dir}/points.csv")

    ov = bgr.copy()
    for _, r in pts.iterrows():                          # ring each point in its well colour
        px = int(ord2x(date_to_ord(str(r.date))))
        py = int(psi2y(r.pressure_psi))
        rgb = TAB10.get(r.well, [255, 0, 255])
        bgrc = (rgb[2], rgb[1], rgb[0])
        cv2.circle(ov, (px, py), 34, bgrc, 5)
        cv2.drawMarker(ov, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 18, 3)
    out_png = f"{csv_dir}/validated_overlay.png"
    cv2.imwrite(out_png, ov)

    # ---- internal consistency report ----
    dlo, dhi = x2ord(L), x2ord(R)
    plo, phi = y2psi(B), y2psi(T)        # B is bottom (low psi), T is top (high psi)
    ords = pts.date.map(date_to_ord)
    in_x = bool(ords.between(dlo, dhi).all())
    in_y = bool(pts.pressure_psi.between(plo, phi).all())
    per_well = pts.well.value_counts().sort_index().to_dict()
    return dict(file=os.path.basename(png_path), points=len(pts),
                date_min=pd.Series(ords).map(lambda o: f"{int(o)//12:04d}-{int(o)%12+1:02d}").min(),
                date_max=pd.Series(ords).map(lambda o: f"{int(o)//12:04d}-{int(o)%12+1:02d}").max(),
                psi_axis=f"{round(plo)}..{round(phi)}",
                in_date_range=in_x, in_psi_range=in_y,
                per_well=per_well, overlay=out_png)

def validate_folder(folder_path, outdir):
    results = []
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.lower().endswith(IMG_EXT): continue
        csv_dir = os.path.join(outdir, file_name)
        if not os.path.isdir(csv_dir):
            print(f"[skip] {file_name}: no extraction folder at {csv_dir}"); continue
        try:
            res = validate_one(os.path.join(folder_path, file_name), csv_dir)
        except Exception as e:
            print(f"[FAIL] {file_name}: {e}"); continue
        results.append(res)
        print(f"=== {res['file']} ===")
        print(f"  points              : {res['points']}  {res['per_well']}")
        print(f"  date span           : {res['date_min']} .. {res['date_max']}")
        print(f"  psi axis range      : {res['psi_axis']}")
        print(f"  points in date-range: {res['in_date_range']}")
        print(f"  points in psi-range : {res['in_psi_range']}")
        print(f"  overlay -> {res['overlay']}\n")
    if results:
        rep = os.path.join(outdir, "validation_report.csv")
        pd.DataFrame(results).to_csv(rep, index=False)
        print(f"Batch summary written -> {rep}")
    return results

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./pressure_time_plots"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "out"
    validate_folder(folder, outdir)