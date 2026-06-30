"""
Extract text, tables, and text-sections from DDR PDFs using pdfplumber.

Tier 0: pure native-text extraction. No OCR, no LLM, no API. Faithful: values
returned verbatim, no sentinel handling or row dropping.

Sections are introduced by bold headings. A heading above a table names that
table; a heading above prose is a text section. Headings are detected by font
(bold, larger than body, not inside a table).

Cross-page handling
-------------------
A section can be split by a page break in two ways:
  (1) its TABLE overflows  -> the next page's first table is a data continuation
  (2) its HEADING is stranded at the bottom of a page -> the next page starts
      with the table's real header row
A first carry-forward handles the common continuation; then a reconciliation
pass fixes stranded headings: if a page's last item is a lone heading (an empty
text section or a one-cell table) and the next page's first table is heading-
less or force-continued, that heading is handed to the table, with continuation
set by whether the table starts with a header row or a data row.

Table entry: {"heading","kind","continuation","bbox","rows"|"text"}.

Usage:
    python extract_ddr.py /path/to/folder_of_pdfs
    python extract_ddr.py file1.pdf file2.pdf
"""

import sys
import json
import re
from collections import Counter
from pathlib import Path

import pdfplumber

HEADING_MAX_GAP = 28
TITLE_TEXT = "summary report"


def _clean(text):
    return " ".join(str(text).split())


def is_bold(ch):
    fn = ch.get("fontname") or ""
    return "Black" in fn or "Bold" in fn


def looks_like_value(cell):
    if cell is None:
        return False
    s = str(cell).strip()
    if not s:
        return False
    return bool(re.match(r"^[+-]?\d*\.?\d+$", s)
                or re.match(r"^\d{1,2}:\d{2}$", s)
                or re.match(r"^\d{4}-\d{2}-\d{2}", s))


def detect_heading_lines(lines, table_spans):
    if not lines:
        return []
    sizes = Counter(round(c.get("size", 0), 1) for ln in lines
                    for c in ln["chars"])
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    heads = []
    for ln in lines:
        chars = ln["chars"]
        if not chars:
            continue
        bold_frac = sum(is_bold(c) for c in chars) / len(chars)
        size = max(round(c.get("size", 0), 1) for c in chars)
        mid = (ln["top"] + ln["bottom"]) / 2
        inside = any(t <= mid <= b for t, b in table_spans)
        if (bold_frac >= 0.9 and size > body_size + 0.05 and not inside
                and _clean(ln["text"]).lower() != TITLE_TEXT):
            heads.append(ln)
    return heads


def assign_table_headings(tables, heading_lines):
    headings, claimed = [], set()
    for t in tables:
        tx0, ttop, tx1, _ = t.bbox
        best_i, best = None, None
        for i, ln in enumerate(heading_lines):
            if ln["bottom"] > ttop + 1:
                continue
            if ln["x1"] < tx0 or ln["x0"] > tx1:
                continue
            if best is None or ln["bottom"] > best["bottom"]:
                best_i, best = i, ln
        if best is not None and (ttop - best["bottom"]) <= HEADING_MAX_GAP:
            headings.append(_clean(best["text"]))
            claimed.add(best_i)
        else:
            headings.append(None)
    return headings, claimed


def build_text_sections(lines, heading_lines, claimed, tables):
    boundaries = sorted([ln["top"] for ln in heading_lines]
                        + [t.bbox[1] for t in tables])
    sections = []
    for i, ln in enumerate(heading_lines):
        if i in claimed:
            continue
        htop = ln["top"]
        below = [b for b in boundaries if b > htop + 0.1]
        stop = min(below) if below else float("inf")
        body, x0, x1, bottom = [], ln["x0"], ln["x1"], ln["bottom"]
        for l2 in lines:
            if htop + 0.1 < l2["top"] < stop - 0.1:
                body.append(l2["text"])
                x0, x1 = min(x0, l2["x0"]), max(x1, l2["x1"])
                bottom = max(bottom, l2["bottom"])
        sections.append({
            "heading": _clean(ln["text"]),
            "kind": "text",
            "continuation": False,
            "bbox": [round(ln["x0"], 1), round(htop, 1),
                     round(x1, 1), round(bottom, 1)],
            "text": "\n".join(body),
        })
    return sections


def _stranded_heading_text(rec):
    """A lone heading: an empty text section, or a one-cell table (e.g. a gray
    heading bar that find_tables captured as a table). Guards against the title
    and against stray numeric cells."""
    text = None
    if (rec.get("kind") == "text" and rec.get("heading")
            and not (rec.get("text") or "").strip()):
        text = rec["heading"]
    elif rec.get("kind") == "table":
        cells = [c for r in (rec.get("rows") or [])
                 for c in r if c is not None and str(c).strip()]
        if len(cells) == 1:
            text = cells[0]
        elif len(cells) == 0 and rec.get("heading"):
            # a gray heading-bar grabbed as a table whose cells came out empty,
            # but the heading text was stamped into its heading field
            text = rec["heading"]
    if not text:
        return None
    t = _clean(text)
    if t.lower() == TITLE_TEXT or looks_like_value(t):
        return None
    return t


def reconcile_headings(pages):
    """Hand a lone heading (empty text section or one-cell heading-bar table) to
    the next heading-less / force-continued table. Works within a page (a gray
    heading bar grabbed as its own table, pushing its data to the next table) and
    across a page break (a heading stranded at the bottom of a page)."""
    flat = [rec for pg in pages for rec in pg["tables"]]
    for i in range(len(flat) - 1):
        heading = _stranded_heading_text(flat[i])
        if not heading:
            continue
        nxt = flat[i + 1]
        if nxt.get("kind") != "table":
            continue
        if not (nxt.get("heading") is None or nxt.get("continuation")):
            continue
        rows = nxt.get("rows") or []
        if not rows:
            continue
        nxt["heading"] = heading
        # header row -> real table start; data row -> overflow continuation
        nxt["continuation"] = any(looks_like_value(c) for c in rows[0])
        flat[i]["_drop"] = True
    for pg in pages:
        pg["tables"] = [r for r in pg["tables"] if not r.get("_drop")]


def extract_pdf(path):
    result = {"file": Path(path).name, "pages": []}
    with pdfplumber.open(path) as pdf:
        last_heading = None
        for page_index, page in enumerate(pdf.pages):
            page = page.dedupe_chars(tolerance=1)
            tables = page.find_tables()
            table_spans = [(t.bbox[1], t.bbox[3]) for t in tables]
            lines = page.extract_text_lines(return_chars=True)
            heading_lines = detect_heading_lines(lines, table_spans)
            headings, claimed = assign_table_headings(tables, heading_lines)

            continued = [False] * len(tables)
            if (tables and headings[0] is None and page_index > 0
                    and last_heading is not None):
                headings[0] = last_heading
                continued[0] = True

            records = []
            for t, heading, is_cont in zip(tables, headings, continued):
                if heading:
                    last_heading = heading
                records.append({
                    "heading": heading, "kind": "table",
                    "continuation": is_cont,
                    "bbox": [round(v, 1) for v in t.bbox],
                    "rows": t.extract(),
                })
            records += build_text_sections(lines, heading_lines, claimed, tables)
            records.sort(key=lambda r: r["bbox"][1])
            result["pages"].append({
                "page_number": page_index,
                "text": page.extract_text() or "",
                "tables": records,
            })

    reconcile_headings(result["pages"])
    return result


def collect_pdf_paths(args):
    paths = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            paths.append(p)
    return paths


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdf_paths = collect_pdf_paths(sys.argv[1:])
    if not pdf_paths:
        print("No PDF files found.")
        sys.exit(1)
    all_results = []
    for path in pdf_paths:
        print(f"Processing {path.name} ...")
        data = extract_pdf(path)
        all_results.append(data)
        out = Path(path).with_suffix(".extracted.json")
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        n_tab = sum(1 for pg in data["pages"] for t in pg["tables"]
                    if t["kind"] == "table")
        n_txt = sum(1 for pg in data["pages"] for t in pg["tables"]
                    if t["kind"] == "text")
        print(f"  -> {len(data['pages'])} page(s), {n_tab} table(s), "
              f"{n_txt} text-section(s) -> {out.name}")
    Path("all_extracted.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False))
    print("\nCombined output -> all_extracted.json")


if __name__ == "__main__":
    main()