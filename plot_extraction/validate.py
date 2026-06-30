"""
Validate extracted CSVs against the ONLY ground truth available: the images.
For every plot in a folder, round-trips its CSVs back to pixel space and draws
them on the original plot, then runs internal consistency checks. If the markers
land on the originals, the extraction is faithful.

Mirrors the extractor's layout: CSVs live in  <outdir>/<file_name>/  and the
overlay is written back into that same subfolder.

Usage:
  python validate.py                      # default folder + outdir
  python validate.py <plots_folder> <outdir>
"""
import cv2, numpy as np, pandas as pd, sys, os
from plot_extraction.parse_graph_2 import load, find_axes_box, calibrate

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def inverse(fn):                       # invert a linear pixel->value map
    b = fn(0); a = fn(1) - b           # value = a*px + b  ->  px = (value-b)/a
    return lambda val: (val - b) / a

def validate_one(png_path, csv_dir):
    """Build the overlay + checks for a single plot. Returns a result dict."""
    bgr, _ = load(png_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    L,R,T,B = find_axes_box(gray)
    x2psi, y2depth = calibrate(gray,L,R,T,B)
    psi2x, depth2y = inverse(x2psi), inverse(y2depth)

    meas = pd.read_csv(f"{csv_dir}/measured_points.csv")
    ref  = pd.read_csv(f"{csv_dir}/reference_lines.csv")
    summ = pd.read_csv(f"{csv_dir}/extraction_summary.csv")
    vrow = summ.loc[summ.quantity=="virgin_pressure_psi","value"]
    virgin = float(vrow.iloc[0]) if len(vrow) and pd.notna(vrow.iloc[0]) else None

    ov = bgr.copy()
    for _,r in meas.iterrows():                                   # magenta rings on dots
        cv2.circle(ov,(int(psi2x(r.pressure_psi)),int(depth2y(r.depth_ft))),30,(255,0,255),5)
    for col,bgrcol in [("MIN_psi",(0,0,255)),("BASE_psi",(60,60,60)),("MAX_psi",(255,0,0))]:
        if col not in ref.columns: continue
        for _,r in ref.iterrows():                               # X's tracing each line
            if pd.notna(r[col]):
                cv2.drawMarker(ov,(int(psi2x(r[col])),int(depth2y(r.depth_ft))),
                               bgrcol,cv2.MARKER_TILTED_CROSS,16,3)
    if virgin is not None:                                       # virgin vertical
        cv2.line(ov,(int(psi2x(virgin)),T),(int(psi2x(virgin)),B),(0,255,255),3)
    out_png = f"{csv_dir}/validated_overlay.png"
    cv2.imwrite(out_png, ov)

    # ---- internal consistency report ----
    xlo,xhi = x2psi(L), x2psi(R)
    if {"MIN_psi","BASE_psi","MAX_psi"}.issubset(ref.columns):
        o = ref.dropna(subset=["MIN_psi","BASE_psi","MAX_psi"])
        order_ok = bool(((o.MIN_psi<o.BASE_psi)&(o.BASE_psi<o.MAX_psi)).all())
        violations = int((~((o.MIN_psi<o.BASE_psi)&(o.BASE_psi<o.MAX_psi))).sum())
        env = sum(np.interp(r.depth_ft,ref.depth_ft,ref.MIN_psi) <= r.pressure_psi
                  <= np.interp(r.depth_ft,ref.depth_ft,ref.MAX_psi) for _,r in meas.iterrows())
    else:
        order_ok, violations, env = None, None, None
    in_range = bool(meas.pressure_psi.between(xlo,xhi).all())
    return dict(file=os.path.basename(png_path), points=len(meas),
                x_min=round(xlo), x_max=round(xhi), order_ok=order_ok,
                order_violations=violations, in_x_range=in_range,
                within_envelope=env, virgin_psi=round(virgin,1) if virgin else None,
                overlay=out_png)

def validate_folder(folder_path, outdir):
    results=[]
    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.lower().endswith(IMG_EXT): continue
        csv_dir = os.path.join(outdir, file_name)
        if not os.path.isdir(csv_dir):
            print(f"[skip] {file_name}: no extraction folder at {csv_dir}"); continue
        try:
            res = validate_one(os.path.join(folder_path,file_name), csv_dir)
        except Exception as e:
            print(f"[FAIL] {file_name}: {e}"); continue
        results.append(res)
        print(f"=== {res['file']} ===")
        print(f"  x-axis range        : {res['x_min']} .. {res['x_max']} psi")
        print(f"  MIN<BASE<MAX holds  : {res['order_ok']}  ({res['order_violations']} violations)")
        print(f"  measured in x-range : {res['in_x_range']}")
        print(f"  within envelope     : {res['within_envelope']}/{res['points']} points")
        print(f"  virgin pressure     : {res['virgin_psi']} psi")
        print(f"  overlay -> {res['overlay']}\n")
    if results:
        rep = os.path.join(outdir,"validation_report.csv")
        pd.DataFrame(results).to_csv(rep,index=False)
        print(f"Batch summary written -> {rep}")
    return results

if __name__=="__main__":
    folder = sys.argv[1] if len(sys.argv)>1 else "./pressure_plots"
    outdir = sys.argv[2] if len(sys.argv)>2 else "out"
    validate_folder(folder, outdir)