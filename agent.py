"""
agent.py - a small ReAct loop: the model thinks, calls a tool, observes the
result, and repeats (up to MAX_TOOL_CALLS) before answering. Tools and dispatch
stay explicit so every step shows up in the UI.
"""
import json

import db
import data_dictionary
import llm
import tool_sql
import tool_semantic
import tool_plot

MAX_TOOL_CALLS = 4
TOOLS = {"sql", "semantic", "plot", "meta"}

_SYSTEM = (
    "You answer questions about drilling reports by calling tools, then giving a "
    "final answer. Reply with ONE JSON object per step.\n"
    'Call a tool: {"thought":"...","action":"sql|semantic|plot|meta","input":"<sub-question>"}\n'
    'Finish:      {"thought":"...","action":"final","answer":"..."}\n'
    "Tools:\n"
    "- sql: structured facts, counts, specific values and exact filters "
    "(operator, rig, depths, HPHT, activities for a wellbore on a date); also "
    "raw plot values.\n"
    "- semantic: fuzzy free-text search over activity summaries and operation "
    "remarks (whipstock, kick, losses, 'what was done when X').\n"
    "- plot: interpret pressure plots - SoR MIN/BASE/MAX envelope, virgin "
    "pressure, formation pressure-vs-depth, or offset-well pressure over time.\n"
    "- meta: what data exists / what you can do / which wells are available.\n"
    "Call several tools if a question needs them; use the fewest that suffice.\n"
    "The 'Recent turns' block is the conversation so far. Use it to resolve "
    "follow-ups like 'that well', 'it', 'those reports', then call the tool that "
    "fetches the specifics. Prefer calling a tool over saying you cannot answer.\n"
    "Final-answer rules: use ONLY tool evidence; never invent values; if a field "
    "is null or evidence is empty, say it is not recorded; -999.99 means missing; "
    "state units (metres for DDR depths, feet for plot depths, PSI for pressure); "
    "DDR wells and plot wells are different datasets - never imply a link."
)

# Plain-prose system prompt for the forced final answer (no JSON, or it leaks
# the envelope into the chat bubble).
_FINAL_SYSTEM = (
    "Answer the question in plain prose using ONLY the evidence in the steps "
    "above. Never invent values; if a field is null or evidence is empty, say it "
    "is not recorded; -999.99 means missing; state units (metres for DDR depths, "
    "feet for plot depths, PSI for pressure); DDR wells and plot wells are "
    "different datasets. Be concise."
)


def _unwrap(text):
    """Defensive: if the model handed back a {...,'answer':...} blob as text,
    show the answer, not the raw JSON."""
    t = (text or "").strip()
    if t.startswith("{") and '"answer"' in t:
        try:
            return str(json.loads(t).get("answer", t)).strip()
        except Exception:
            pass
    return t


def _context():
    conn = db.connect()
    try:
        tables = set(db.list_tables(conn))
        wb = []
        if "reports" in tables:
            wb = [r[0] for r in conn.execute(
                "SELECT DISTINCT wellbore_id FROM reports WHERE wellbore_id "
                "IS NOT NULL ORDER BY wellbore_id LIMIT 50").fetchall()]
        wells = []
        if "formation_pressure_plots" in tables:
            wells = [r[0] for r in conn.execute(
                "SELECT DISTINCT well_label FROM formation_pressure_plots "
                "WHERE well_label IS NOT NULL ORDER BY well_label").fetchall()]
        return {"wellbores": wb, "wells": wells}
    finally:
        conn.close()


def _dispatch(action, tool_input, ctx):
    if action == "sql":
        return tool_sql.construct_sql(tool_input)
    if action == "semantic":
        return tool_semantic.semantic_search(tool_input)
    if action == "plot":
        return tool_plot.interpret_plot(tool_input)
    return {"ok": True, "tool": "meta",
            "capabilities": data_dictionary.capabilities_text(),
            "wellbores": ctx["wellbores"], "wells": ctx["wells"]}


def _prompt(question, ctx, scratch, history):
    parts = []
    if history:
        parts.append(f"Recent turns:\n{history}")
    parts.append(f"DDR wellbores: {ctx['wellbores']}")
    parts.append(f"Plot wells: {ctx['wells']}")
    if scratch:
        parts.append("Steps so far:\n" + "\n".join(scratch))
    parts.append(f"Question: {question}\n\nNext JSON step.")
    return "\n\n".join(parts)


def _force_final(question, ctx, scratch, history):
    return _unwrap(llm.chat(
        _FINAL_SYSTEM, _prompt(question, ctx, scratch, history)
        + "\n\nGive the final answer now, grounded only in the steps above."))


def _result(text, reason, steps):
    tools = []
    for s in steps:
        if s["action"] not in tools:
            tools.append(s["action"])
    return {"answer": text, "tool": "+".join(tools) or "none",
            "reason": reason, "steps": steps,
            "evidence": [s["result"] for s in steps]}


def answer(question, history=""):
    """Run the ReAct loop and return the answer plus the full evidence trail."""
    ctx = _context()
    steps = []          # {action, input, result} for the UI
    scratch = []        # text observations fed back to the model

    for _ in range(MAX_TOOL_CALLS):
        out = llm.chat_json(_SYSTEM, _prompt(question, ctx, scratch, history))
        action = (out.get("action") or "final").strip().lower()
        if action not in TOOLS:                       # final (or anything else)
            text = _unwrap(out.get("answer") or "") \
                or _force_final(question, ctx, scratch, history)
            return _result(text, out.get("thought", ""), steps)

        tool_input = out.get("input") or question
        result = _dispatch(action, tool_input, ctx)
        steps.append({"action": action, "input": tool_input, "result": result})
        scratch.append(f"Action: {action}({tool_input})\nObservation: "
                       + json.dumps(result, default=str)[:3000])

    return _result(_force_final(question, ctx, scratch, history),
                   "tool budget reached", steps)
