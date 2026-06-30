"""
tool_plot.py - interpret_plot: turns the extracted plot tables into facts.

Deterministic on purpose: envelope membership is numpy.interp over the
MIN/BASE/MAX curves, not Text2SQL. An LLM only picks the sub-kind and target
well; the numbers are computed here.

  formation  -> measured points vs MIN/BASE/MAX envelope + virgin pressure
  timeseries -> offset-well pressure vs month for a subject well
"""
import numpy as np

import db
import llm


def _dicts(sql, params=()):
    conn = db.connect()
    try:
        return db.rows_as_dicts(conn, sql, params)
    finally:
        conn.close()


def _series(ref, key):
    """Build sorted (depth, value) arrays for one SoR curve, dropping Nones."""
    xs, ys = [], []
    for r in ref:
        d, v = r.get("depth_ft"), r.get(key)
        if d is not None and v is not None:
            xs.append(float(d)); ys.append(float(v))
    if not xs:
        return None, None
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]


def list_formation_wells():
    return [r["well_label"] for r in _dicts(
        "SELECT DISTINCT well_label FROM formation_pressure_plots "
        "WHERE well_label IS NOT NULL ORDER BY well_label")]


# ----------------------------------------------------------- formation plots
def envelope_check(well_label):
    plot = _dicts(
        "SELECT plot_id, virgin_pressure_psi, source_file "
        "FROM formation_pressure_plots WHERE well_label = ? "
        "ORDER BY plot_id LIMIT 1", (well_label,))
    if not plot:
        return {"ok": False, "error": f"no formation plot for {well_label}"}
    pid = plot[0]["plot_id"]
    vp = plot[0]["virgin_pressure_psi"]

    pts = _dicts(
        "SELECT depth_ft, pressure_psi FROM formation_measured_points "
        "WHERE plot_id = ? AND depth_ft IS NOT NULL "
        "AND pressure_psi IS NOT NULL ORDER BY depth_ft", (pid,))
    ref = _dicts(
        "SELECT depth_ft, min_psi, base_psi, max_psi "
        "FROM formation_reference_lines WHERE plot_id = ? ORDER BY depth_ft",
        (pid,))
    if not pts or not ref:
        return {"ok": False, "error": f"incomplete plot data for {well_label}"}

    mnx, mny = _series(ref, "min_psi")
    bsx, bsy = _series(ref, "base_psi")
    mxx, mxy = _series(ref, "max_psi")

    classified = []
    n_above_max = n_below_min = n_exceeds_virgin = 0
    for p in pts:
        d, val = float(p["depth_ft"]), float(p["pressure_psi"])
        mn = float(np.interp(d, mnx, mny)) if mnx is not None else None
        bs = float(np.interp(d, bsx, bsy)) if bsx is not None else None
        mx = float(np.interp(d, mxx, mxy)) if mxx is not None else None

        if mx is not None and val > mx:
            band = "above MAX (over the SoR envelope)"; n_above_max += 1
        elif mn is not None and val < mn:
            band = "below MIN (under the SoR envelope)"; n_below_min += 1
        elif bs is not None and val <= bs:
            band = "between MIN and BASE"
        else:
            band = "between BASE and MAX"

        exceeds_virgin = (vp is not None and val > float(vp))
        if exceeds_virgin:
            n_exceeds_virgin += 1
        classified.append({
            "depth_ft": round(d, 1), "pressure_psi": round(val, 1),
            "min": round(mn, 1) if mn is not None else None,
            "base": round(bs, 1) if bs is not None else None,
            "max": round(mx, 1) if mx is not None else None,
            "band": band, "exceeds_virgin": exceeds_virgin,
        })

    return {
        "ok": True, "well_label": well_label,
        "virgin_pressure_psi": vp, "n_points": len(pts),
        "n_above_max": n_above_max, "n_below_min": n_below_min,
        "n_exceeds_virgin": n_exceeds_virgin,
        "all_inside_envelope": (n_above_max == 0 and n_below_min == 0),
        "points": classified,
    }


def describe_formation(well_label):
    chk = envelope_check(well_label)
    if not chk.get("ok"):
        return chk
    pts = chk["points"]
    depths = [p["depth_ft"] for p in pts]
    press = [p["pressure_psi"] for p in pts]
    trend = "n/a"
    if len(pts) >= 2:
        slope = float(np.polyfit(depths, press, 1)[0])
        trend = ("increases with depth" if slope > 0.05
                 else "decreases with depth" if slope < -0.05
                 else "roughly constant with depth")
    chk.update({
        "kind": "formation", "depth_range_ft": [min(depths), max(depths)],
        "pressure_range_psi": [min(press), max(press)],
        "trend": trend,
    })
    return chk


# ---------------------------------------------------------- timeseries plots
def timeseries_wells():
    return [r["well_label"] for r in _dicts(
        "SELECT DISTINCT well_label FROM timeseries_pressure_plots "
        "WHERE well_label IS NOT NULL ORDER BY well_label")]


def describe_timeseries(well_label=None):
    """Offset-well pressure-over-time for one subject well (or all if None)."""
    where = "WHERE p.pressure_psi IS NOT NULL AND p.obs_month IS NOT NULL"
    params = ()
    if well_label:
        where += " AND t.well_label = ?"
        params = (well_label,)
    pts = _dicts(
        "SELECT t.well_label AS subject, p.offset_label, p.obs_month, "
        "p.pressure_psi FROM timeseries_pressure_points p "
        "JOIN timeseries_pressure_plots t ON p.ts_plot_id = t.ts_plot_id "
        f"{where} ORDER BY t.well_label, p.offset_label, p.obs_month", params)
    if not pts:
        return {"ok": False, "error": "no time-series points found"}

    groups = {}
    for p in pts:
        groups.setdefault((p["subject"], p["offset_label"]), []).append(p)
    offsets = []
    for (subject, offset), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["obs_month"])
        vals = [float(r["pressure_psi"]) for r in rows]
        first, last = vals[0], vals[-1]
        direction = ("rising" if last - first > 20
                     else "falling" if last - first < -20 else "flat")
        offsets.append({
            "subject_well": subject, "offset_label": offset,
            "n_points": len(rows), "first_month": rows[0]["obs_month"],
            "last_month": rows[-1]["obs_month"],
            "min_psi": round(min(vals), 1), "max_psi": round(max(vals), 1),
            "first_psi": round(first, 1), "last_psi": round(last, 1),
            "overall_direction": direction,
        })
    return {"ok": True, "kind": "timeseries",
            "subject_well": well_label or "all", "offsets": offsets}


# --------------------------------------------------------------- dispatch
_ROUTE_SYSTEM = (
    "Classify a pressure-plot question. Return JSON "
    '{"kind": "formation|timeseries", "well": "Well_NN or null"}.\n'
    "formation = a well's measured pressure vs depth, the SoR MIN/BASE/MAX "
    "envelope, or virgin pressure.\n"
    "timeseries = offset-well pressure over time/by month for a subject well.\n"
    "Set well to the subject well if the question names one, else null."
)


def interpret_plot(question):
    """LLM picks sub-kind + target well; facts are computed deterministically."""
    wells = sorted(set(list_formation_wells()) | set(timeseries_wells()))
    out = llm.chat_json(
        _ROUTE_SYSTEM, f"Wells available: {wells}\nQuestion: {question}")
    kind = (out.get("kind") or "formation").strip().lower()
    well = out.get("well") if out.get("well") in wells else None

    if kind == "timeseries":
        facts = describe_timeseries(well)
        return {"ok": facts.get("ok", False), "tool": "plot",
                "kind": "timeseries", "target": well or "all wells",
                "facts": facts}

    if well:
        facts = describe_formation(well)
        return {"ok": facts.get("ok", False), "tool": "plot",
                "kind": "formation", "target": well, "facts": facts}

    fw = list_formation_wells()
    return {"ok": bool(fw), "tool": "plot", "kind": "formation",
            "target": "all wells",
            "facts": {"ok": bool(fw),
                      "wells": [describe_formation(w) for w in fw]}}
