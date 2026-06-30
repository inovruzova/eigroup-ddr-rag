"""
Deterministic extractor for matplotlib time-series scatter plots
(pressure-vs-date, one coloured series per well).
No ML, no fine-tuning. Same spine->calibrate->mask->blob skeleton as the
formation-pressure extractor, but:
  - x axis is rotated DATE labels  -> deskew the whole label band once, OCR
    the dates, snap each back to its tick, fit pixel -> ordinal-month (linear).
  - every series is round DOTS in a matplotlib tab10 colour (not pure RGB).
  - the legend sits inside the axes box, so its colour swatches are detected
    and removed (they line up in one vertical column across colours).
One row per detected point:  well, date (YYYY-MM), pressure_psi.
"""
import cv2, numpy as np, pandas as pd, pytesseract, sys, os, re

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ilhama.novruzova\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# matplotlib tab10 defaults, in legend order (RGB)
TAB10 = {"Well_01": [31, 119, 180], "Well_02": [255, 127, 14],
         "Well_03": [44, 160, 44],  "Well_04": [214, 39, 40]}

# ----------------------------------------------------------------- shared bits
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
    if len(idx) == 0: return []
    g, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v-p > gap: g.append((s+p)//2); s = v
        p = v
    g.append((s+p)//2); return g

def ocr_num(sub):
    """OCR a horizontal number: tight glyph bbox, pad, upscale, otsu, psm 8."""
    dark = sub < 100
    cols, rows = np.where(dark.any(0))[0], np.where(dark.any(1))[0]
    if len(cols) == 0 or len(rows) == 0: return None
    crop = sub[max(0,rows.min()-4):rows.max()+5, max(0,cols.min()-6):cols.max()+7]
    crop = cv2.copyMakeBorder(crop,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    crop = cv2.resize(crop,(0,0),fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    _, crop = cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t = pytesseract.image_to_string(crop,config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()
    try: return float(t)
    except: return None

def robust_fit(pixels, values):
    """Fit value = a*pixel + b, drop outliers (bad OCR reads), refit."""
    p, v = np.array(pixels,float), np.array(values,float)
    for _ in range(3):
        a,b = np.polyfit(p, v, 1)
        res = np.abs(a*p + b - v)
        keep = res < max(3*np.median(res), 1e-6) if np.median(res)>0 else np.ones_like(res,bool)
        if keep.all(): break
        p, v = p[keep], v[keep]
    a,b = np.polyfit(p, v, 1)
    return a, b

# --------------------------------------------------------------- date helpers
def date_to_ord(s):                       # "YYYY-MM" -> months since year 0
    y, m = map(int, s.split('-')); return y*12 + (m-1)

def ord_to_date(o):                       # months -> "YYYY-MM"
    o = int(round(o)); return f"{o//12:04d}-{o%12 + 1:02d}"

def _rotate_keep(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(h*sin+w*cos), int(h*cos+w*sin)
    M[0,2] += nw/2-w/2; M[1,2] += nh/2-h/2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=255), M

_DATE = re.compile(r'^(\d{4})-(\d{1,2})$')

def read_date_ticks(gray, L, R, B, xt):
    """Deskew the whole label band once (rotated date labels are 45-deg in
    matplotlib), OCR every date with its position, map each back through the
    rotation and snap to the nearest tick. Returns [(tick_px, ordinal), ...]."""
    col0, row0 = L-60, B+5
    band = gray[row0:B+340, col0:R+60]
    xt = np.array(xt)
    best = []
    for ang in (-45, -42, -40, -38, -48):        # 45 is the matplotlib default
        rot, M = _rotate_keep(band, ang)
        rs = cv2.resize(rot, (0,0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(rs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(
            th, config='--psm 11 -c tessedit_char_whitelist=0123456789-',
            output_type=pytesseract.Output.DICT)
        Minv = cv2.invertAffineTransform(M)
        pairs = []
        for i, t in enumerate(data['text']):
            t = t.strip(); m = _DATE.match(t)
            if not m or not (1 <= int(m.group(2)) <= 12): continue
            xc = (data['left'][i] + data['width'][i]/2) / 1.5
            yc = (data['top'][i]  + data['height'][i]/2) / 1.5
            ox = Minv[0,0]*xc + Minv[0,1]*yc + Minv[0,2] + col0      # -> original x
            tick = int(xt[np.argmin(np.abs(xt - ox))])
            pairs.append((tick, date_to_ord(t)))
        pairs = sorted(set(pairs))
        if len(pairs) > len(best): best = pairs
        if len(best) == len(xt): break
    return best

# ------------------------------------------------------------------ calibrate
def calibrate(gray, L, R, T, B):
    """Returns (x2ord, y2psi) — both linear & invertible.
    x2ord: pixel -> ordinal month.   y2psi: pixel -> pressure."""
    # y axis: plain horizontal numbers (reuse the pressure-plot logic)
    yt = tick_centres((gray[:, L-25:L-5] < 100).sum(1))
    yv = [ocr_num(gray[y-35:y+35, L-200:L-15]) for y in yt]
    yp, yvv = zip(*[(p,v) for p,v in zip(yt, yv) if v is not None])
    ay, by = robust_fit(yp, yvv)

    # x axis: rotated date labels
    xt = tick_centres((gray[B+3:B+22, :] < 100).sum(0))
    pairs = read_date_ticks(gray, L, R, B, xt)
    xp = [p for p,_ in pairs]; xo = [o for _,o in pairs]
    ax, bx = robust_fit(xp, xo)
    return (lambda px: ax*px + bx), (lambda py: ay*py + by)

# --------------------------------------------------------------- dot detection
def _colour_blobs(rgb, L, R, T, B, target, tol=120):
    m = (np.abs(rgb[T:B, L:R] - np.array(target)).sum(2) < tol).astype(np.uint8)*255
    n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
    areas = [st[i,4] for i in range(1,n) if st[i,4] > 150]
    med = np.median(areas) if areas else 0
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if med and 0.4*med < a < 2.2*med and abs(w-h) < 0.45*max(w, h):   # round, dot-sized
            out.append((ct[i][0]+L, ct[i][1]+T))
    return out

def find_legend_box(gray, L, R, T, B, pad=6):
    """Locate the matplotlib legend frame: a hollow light-grey rectangle
    (~180..216) inside the axes, distinct from gridlines (~223) and the white
    background (255). This is what *defines* the legend region, so excluding
    its interior removes the swatches without ever touching real data points —
    even when several wells share the same date (same x-column).
    Returns (x0,y0,x1,y1) in image coords, or None if the legend is frameless."""
    interior = np.zeros_like(gray, bool)
    interior[T+pad:B-pad, L+pad:R-pad] = True
    band = ((gray > 170) & (gray < 218) & interior).astype(np.uint8) * 255
    n, _, st, _ = cv2.connectedComponentsWithStats(band, 8)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 60 or h < 60: continue                      # too small for a frame
        if w > 0.8*(R-L) or h > 0.8*(B-T): continue        # implausibly large
        if a/(w*h) > 0.6: continue                         # must be a hollow border, not a filled blob
        if w*h > best_area: best, best_area = (x, y, w, h), w*h
    if best is None: return None
    x, y, w, h = best
    return (x-3, y-3, x+w+3, y+h+3)

def _strip_legend(series, legend_box):
    """Drop only the blobs that fall inside the detected legend frame.
    Fallback (frameless legend): treat a column as legend ONLY if it stacks
    >=3 distinct colours that are *evenly spaced* in y (real same-date data has
    arbitrary pressures, so its y-gaps are uneven)."""
    if legend_box is not None:
        x0, y0, x1, y1 = legend_box
        inside = lambda x, y: x0 <= x <= x1 and y0 <= y <= y1
        return {k: [(x, y) for (x, y) in pts if not inside(x, y)] for k, pts in series.items()}

    flat = [(k, x, y) for k, pts in series.items() for (x, y) in pts]
    drop = set()
    for k, x, y in flat:
        col = sorted((yy, kk, xx) for kk, xx, yy in flat if abs(xx - x) < 20)
        colours = {kk for _, kk, _ in col}
        if len(colours) < 3: continue
        gaps = np.diff([yy for yy, _, _ in col])
        if len(gaps) >= 2 and gaps.std() < 0.15*gaps.mean():   # evenly spaced -> legend
            drop.update((kk, xx, yy) for yy, kk, xx in col)
    return {k: [(x, y) for (x, y) in pts if (k, x, y) not in drop] for k, pts in series.items()}

# ------------------------------------------------------------------------ run
def run(path, outdir, file_name):
    bgr, rgb = load(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    L, R, T, B = find_axes_box(gray)
    x2ord, y2psi = calibrate(gray, L, R, T, B)

    series = {k: _colour_blobs(rgb, L, R, T, B, c) for k, c in TAB10.items()}
    series = _strip_legend(series, find_legend_box(gray, L, R, T, B))

    rows = []
    for well, pts in series.items():
        for (x, y) in pts:
            rows.append({"well": well,
                         "date": ord_to_date(x2ord(x)),
                         "pressure_psi": round(y2psi(y), 1)})
    df = pd.DataFrame(rows).sort_values(["well", "date"]).reset_index(drop=True)

    os.makedirs(f"{outdir}/{file_name}", exist_ok=True)
    df.to_csv(f"{outdir}/{file_name}/points.csv", index=False)
    summ = (df.groupby("well")
              .agg(n_points=("pressure_psi", "size"),
                   first_date=("date", "min"), last_date=("date", "max"),
                   min_psi=("pressure_psi", "min"), max_psi=("pressure_psi", "max"))
              .reset_index())
    summ.to_csv(f"{outdir}/{file_name}/extraction_summary.csv", index=False)
    return df, (L, R, T, B)

if __name__ == "__main__":
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "./pressure_time_plots"
    outdir      = sys.argv[2] if len(sys.argv) > 2 else "out"
    for file_name in os.listdir(folder_path):
        src = os.path.join(folder_path, file_name)
        df, box = run(src, outdir, file_name)
        print(f"=== {file_name} ===  axes box (L,R,T,B): {box}")
        print(df.to_string(index=False), "\n")