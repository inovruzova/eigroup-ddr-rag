"""
data_dictionary.py - the semantics the schema alone does not carry.

schema_reference.json gives column NAMES; this file gives column MEANING plus
the handful of rules that keep Text2SQL honest (sentinels, placeholder dates,
the two-island no-join boundary). This is the highest-leverage file for SQL
accuracy: column names like equ_mud_weight / bhst / iadc_code are opaque
without it.
"""
import db

# Rules every generated SQL query must respect. Injected verbatim into the
# Text2SQL prompt.
GLOBAL_NOTES = """\
RULES (obey all):
1. Never join DDR tables to PLOT tables (no shared key). Within the PLOT island,
   formation_pressure_plots.well_label = timeseries_pressure_plots.well_label
   (subject well, e.g. 'Well_05') may be joined; offset_label is local to its
   ts_plot_id (not a subject well, not comparable across plots).
2. -999.99 is a missing-value sentinel: exclude it before aggregating/comparing
   (col != -999.99 AND col IS NOT NULL).
3. NULL/empty = not recorded; do not invent a value.
4. report_number is NOT unique (restarts per campaign); identify wells by
   wellbore_id, not report_number.
5. Some period dates are placeholders (e.g. 1979-12-31), so MIN(period_start)
   may not be the earliest real report.
6. Units: DDR depths metres, PLOT depths feet, pressures PSI. Never mix.
7. SQLite; one read-only SELECT; LIKE for wellbore matching; CAST(col AS REAL)
   if a numeric is stored as text.
"""

ABBREVIATIONS = {
    "mMD / MD": "measured depth along the wellbore, in metres",
    "mTVD / TVD": "true vertical depth, in metres",
    "TVD Sub Sea": "true vertical depth below sea level, in feet (plots)",
    "RKB": "rotary kelly bushing (the rig-floor depth reference)",
    "MSL": "mean sea level",
    "SoR": "Scheme of Requirements (the allowed pressure envelope: MIN/BASE/MAX)",
    "HPHT": "high pressure / high temperature well",
    "ROP": "rate of penetration",
    "BHA": "bottom hole assembly",
    "TIH": "trip in hole",
    "BHST": "bottom hole static temperature",
    "IADC": "bit dull/classification code",
    "GOR": "gas-oil ratio",
    "BHP": "bottom hole pressure",
}

# Plain-language notes for the columns whose names are not self-explanatory.
# Keyed (table, column). Only what adds signal - obvious columns are omitted.
COLUMN_NOTES = {
    ("reports", "wellbore_id"): "well identifier, e.g. '15/9-19 A' (DDR island)",
    ("reports", "period_start"): "start of the report's 24h period (may be placeholder)",
    ("reports", "period_end"): "end of the report's 24h period (may be placeholder)",
    ("reports", "report_creation_time"): "when the DIGITAL report was generated (often 2018), NOT the drilling date",
    ("reports", "spud_date"): "date drilling began",
    ("reports", "operator"): "operating company, e.g. 'Statoil' (often blank)",
    ("reports", "rig_name"): "drilling rig, e.g. 'BYFORD DOLPHIN' (often blank)",
    ("reports", "water_depth_msl"): "water depth below mean sea level, metres",
    ("reports", "elevation_rkb_msl"): "rig-floor (RKB) elevation above MSL, metres",
    ("reports", "tight_well"): "Y/N confidential-well flag",
    ("reports", "hpht"): "Y/N high pressure / high temperature flag",
    ("reports", "dist_drilled"): "metres drilled this period (-999.99 = missing)",
    ("reports", "penetration_rate"): "ROP, m/h (-999.99 = missing)",
    ("reports", "depth_at_kick_off_mmd"): "sidetrack kick-off point, metres MD",
    ("reports", "plug_back_depth_mmd"): "depth cement/plug set back to, metres MD",
    ("survey_station", "inclination"): "hole inclination, degrees",
    ("survey_station", "azimuth"): "hole azimuth, degrees",
    ("operations", "main_sub_activity"): "operation performed (free text)",
    ("operations", "remark"): "free-text note on the operation",
    ("drilling_fluid", "fluid_density"): "mud weight / density",
    ("drilling_fluid", "funnel_visc"): "funnel viscosity of the mud",
    ("gas_reading_information", "c1"): "methane fraction (C1); c2..ic5 are heavier gas components",
    ("bit_record", "form_rop"): "formation rate of penetration for the bit run",
    ("welltest_information", "bhp"): "bottom hole pressure",
    ("welltest_information", "gor"): "gas-oil ratio",
    # plot island
    ("formation_pressure_plots", "well_label"): "plot well id, e.g. 'Well_01' (PLOT island)",
    ("formation_pressure_plots", "virgin_pressure_psi"): "undisturbed reservoir pressure, a single constant per plot, PSI",
    ("formation_measured_points", "depth_ft"): "TVD sub sea, feet",
    ("formation_measured_points", "pressure_psi"): "a measured formation pressure point, PSI",
    ("formation_reference_lines", "min_psi"): "MIN allowed pressure per SoR at that depth, PSI",
    ("formation_reference_lines", "base_psi"): "BASE (expected) pressure per SoR, PSI",
    ("formation_reference_lines", "max_psi"): "MAX allowed pressure per SoR, PSI",
    ("timeseries_pressure_plots", "well_label"): "subject well, e.g. 'Well_05' (joins to formation_pressure_plots.well_label)",
    ("timeseries_pressure_points", "offset_label"): "reference (offset) well within this plot, 'Offset_1'..'Offset_4' (local to ts_plot_id)",
    ("timeseries_pressure_points", "obs_month"): "observation month, 'YYYY-MM'",
    ("timeseries_pressure_points", "pressure_psi"): "pressure reading at that month, PSI",
}


def dictionary_text(conn):
    """Compact schema (one line per table) + notes for opaque columns + global
    rules. Kept terse on purpose - it ships on every Text2SQL call."""
    tables = db.list_tables(conn)
    ddr = [t for t in tables if t not in db.PLOT_TABLES]
    plot = [t for t in tables if t in db.PLOT_TABLES]

    def schema(names):
        return "\n".join(
            f"{t}(" + ", ".join(n for n, _ in db.table_columns(conn, t)) + ")"
            for t in names)

    notes = "\n".join(f"{t}.{c}: {note}"
                      for (t, c), note in COLUMN_NOTES.items() if t in tables)
    abbr = ", ".join(f"{k}={v}" for k, v in ABBREVIATIONS.items())

    parts = ["DDR ISLAND tables (key report_id/wellbore_id):\n" + schema(ddr)]
    if plot:
        parts.append("PLOT ISLAND tables (key plot_id/well_label):\n" + schema(plot))
    parts.append("COLUMN NOTES (opaque columns only):\n" + notes)
    parts.append("ABBREVIATIONS: " + abbr)
    parts.append(GLOBAL_NOTES)
    return "\n\n".join(parts)


def capabilities_text():
    """Used to answer meta questions ('what can you do?')."""
    return (
        "I answer questions about Daily Drilling Reports (DDRs) and the "
        "formation/time-series pressure plots stored in ddr.db.\n"
        "- Structured lookups & counts (operators, rigs, depths, water depth, "
        "HPHT flags, survey stations, bit records, casing, etc.) via SQL.\n"
        "- Free-text search over activity summaries and operation remarks "
        "(e.g. 'reports mentioning a whipstock').\n"
        "- Pressure-plot interpretation: whether a well's measured points sit "
        "inside the SoR MIN/BASE/MAX envelope and how they compare to virgin "
        "pressure (formation), and offset-well pressure over time (time-series).\n"
        "Note: DDR wells (e.g. '15/9-19 A') and plot wells ('Well_05') are "
        "separate datasets and cannot be cross-referenced."
    )