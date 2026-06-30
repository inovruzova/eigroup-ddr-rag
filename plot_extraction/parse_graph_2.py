"""
Deterministic extractor for matplotlib formation-pressure plots.
No ML, no fine-tuning. Auto-detects the axes box, calibrates from the tick
labels (OCR + robust line fit), then pulls data by colour:
  - black dots  -> measured pressure points
  - red/grey/blue lines -> MIN / BASE / MAX as per SoR
  - purple dashed vertical -> virgin pressure (a constant)
"""
import cv2, numpy as np, pandas as pd, pytesseract, sys, os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ilhama.novruzova\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


def load(path):
    bgr = cv2.imread(path)
    if bgr is None: raise FileNotFoundError(path)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(int)

def find_axes_box(gray):
    """Plot spines = the darkest full-length row/column runs forming a rectangle."""
    dark = gray < 100
    cd, rd = dark.sum(0), dark.sum(1)
    def edges(arr, frac=0.7):
        idx = np.where(arr > frac*arr.max())[0]
        g, s, p = [], idx[0], idx[0]
        for v in idx[1:]:
            if v-p > 5: g.append((s+p)//2); s = v
            p = v
        g.append((s+p)//2); return g
    xs, ys = edges(cd), edges(rd)
    return min(xs), max(xs), min(ys), max(ys)   # L, R, T, B

def tick_centres(mask1d, min_run=8, gap=5):
    idx = np.where(mask1d > min_run)[0]
    if len(idx)==0: return []
    g, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v-p > gap: g.append((s+p)//2); s = v
        p = v
    g.append((s+p)//2); return g

def ocr_num(sub):
    """OCR a number from a sub-image: tightly bound the dark glyphs, pad,
    upscale, threshold, read with psm 8 (single word) and a digit whitelist."""
    dark = sub < 100
    cols, rows = np.where(dark.any(0))[0], np.where(dark.any(1))[0]
    if len(cols)==0 or len(rows)==0: return None
    crop = sub[max(0,rows.min()-4):rows.max()+5, max(0,cols.min()-6):cols.max()+7]
    crop = cv2.copyMakeBorder(crop,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    crop = cv2.resize(crop,(0,0),fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    _,crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t = pytesseract.image_to_string(crop,config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()
    try: return float(t)
    except: return None

def robust_fit(pixels, values):
    """Fit value = a*pixel + b, then drop outliers (bad OCR reads) and refit."""
    p, v = np.array(pixels,float), np.array(values,float)
    for _ in range(3):
        a,b = np.polyfit(p, v, 1)
        res = np.abs(a*p + b - v)
        keep = res < max(3*np.median(res), 1e-6) if np.median(res)>0 else np.ones_like(res,bool)
        if keep.all(): break
        p, v = p[keep], v[keep]
    a,b = np.polyfit(p, v, 1)
    return a, b   # value = a*pixel + b

def calibrate(gray, L,R,T,B):
    xt = tick_centres((gray[B+5:B+25,:]<100).sum(0))
    yt = tick_centres((gray[:,L-25:L-5]<100).sum(1))
    # x labels sit just below the spine. Take ONLY the first contiguous dark-row
    # cluster, so the axis title further down can't pollute the OCR.
    rows = [y for y in range(B+10,B+160) if (gray[y]<100).sum()>50]
    if rows:
        y0 = rows[0]
        y1 = y0
        for y in rows:
            if y - y1 <= 6: y1 = y
            else: break
        y1 += 8
    else:
        y0,y1 = B+30,B+70
    xv = [ocr_num(gray[y0:y1, x-105:x+105]) for x in xt]
    yv = [ocr_num(gray[y-35:y+35, L-170:L-15]) for y in yt]
    xp,xvv = zip(*[(p,v) for p,v in zip(xt,xv) if v is not None])
    yp,yvv = zip(*[(p,v) for p,v in zip(yt,yv) if v is not None])
    ax,bx = robust_fit(xp,xvv)          # PSI   = ax*px + bx
    ay,by = robust_fit(yp,yvv)          # depth = ay*py + by
    return (lambda px: ax*px+bx), (lambda py: ay*py+by)

def colour_mask(rgb, target, tol=60):
    return (np.abs(rgb-np.array(target)).sum(2) < tol).astype(np.uint8)

def measured_points(rgb, L,R,T,B, x2psi, y2depth):
    m = colour_mask(rgb[T:B,L:R], [0,0,0])*255
    n,_,stats,cent = cv2.connectedComponentsWithStats(m,8)
    areas = [stats[i,4] for i in range(1,n)]
    med = np.median([a for a in areas if a>150]) if areas else 0
    pts=[]
    for i in range(1,n):
        x,y,w,h,a = stats[i]
        if 0.5*med < a < 2*med and abs(w-h)<0.4*max(w,h):   # round, dot-sized
            cx,cy = cent[i]
            pts.append((y2depth(cy+T), x2psi(cx+L)))
    return sorted(pts)

def sample_line(rgb, L,R,T,B, target, x2psi, y2depth, step=8):
    out=[]
    for py in range(T+2, B-1, step):
        row = rgb[py, L:R]
        xs = np.where(np.abs(row-np.array(target)).sum(1) < 60)[0]
        if len(xs): out.append((y2depth(py), x2psi(np.median(xs)+L)))
    return out

def virgin_pressure(rgb, L,R,T,B, x2psi):
    m = colour_mask(rgb[T:B,L:R], [128,0,128])
    col = m.sum(0)
    if col.max()==0: return None
    return x2psi(int(np.argmax(col))+L)

def run(path, outdir, file_name):
    bgr, rgb = load(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    L,R,T,B = find_axes_box(gray)
    x2psi, y2depth = calibrate(gray,L,R,T,B)

    meas = measured_points(rgb,L,R,T,B,x2psi,y2depth)
    df_meas = pd.DataFrame(meas, columns=["depth_ft","pressure_psi"]).round(1)

    lines = {"MIN":[255,0,0], "BASE":[128,128,128], "MAX":[0,0,255]}
    series = {k: dict(sample_line(rgb,L,R,T,B,c,x2psi,y2depth)) for k,c in lines.items()}
    depths = sorted({round(d) for s in series.values() for d in s})
    rows=[]
    for d in depths:
        row={"depth_ft":d}
        for k in lines:
            near=[(abs(dd-d),v) for dd,v in series[k].items() if abs(dd-d)<1.5]
            row[f"{k}_psi"]=round(min(near)[1],1) if near else None
        rows.append(row)
    df_lines = pd.DataFrame(rows)
    vp = virgin_pressure(rgb,L,R,T,B,x2psi)

    os.makedirs(outdir + "/" + file_name,exist_ok=True)
    df_meas.to_csv(f"{outdir}/{file_name}/measured_points.csv",index=False)
    df_lines.to_csv(f"{outdir}/{file_name}/reference_lines.csv",index=False)
    # virgin pressure is a single constant per plot -> its own small file
    pd.DataFrame([{"quantity":"virgin_pressure_psi",
                   "value":round(vp,1) if vp is not None else None}]
                 ).to_csv(f"{outdir}/{file_name}/extraction_summary.csv",index=False)
    return df_meas, df_lines, vp, (L,R,T,B)

if __name__=="__main__":
    folder_path = sys.argv[1] if len(sys.argv)>1 else "./pressure_plots"
    
    for file_name in os.listdir(folder_path):
        src = os.path.join(folder_path, file_name)
        
        dm,dl,vp,box = run(src,"out", file_name)
        print("axes box (L,R,T,B):",box,"\n")
        print("MEASURED PRESSURE POINTS\n",dm.to_string(index=False),"\n")
        print(f"VIRGIN PRESSURE: {vp:.1f} psi\n")
        print("REFERENCE LINES (SoR) — sampled\n",dl.to_string(index=False))