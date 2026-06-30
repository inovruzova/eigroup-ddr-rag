"""
app.py - the Streamlit chat UI. Run with:  streamlit run app.py

Keeps a chat history, sends each question through agent.answer(), shows the
reply plus an expander revealing the tools / SQL / rows / snippets used.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()              # read .env in the working directory, if present
except Exception:
    pass

import pandas as pd
import streamlit as st

import agent
import db
import tool_semantic

st.set_page_config(page_title="DDR-RAG", page_icon="O", layout="centered")

# let the LLM key come from Streamlit secrets if present (safe if no secrets file)
try:
    for _k in ("DDR_LLM_API_KEY", "DDR_LLM_BASE_URL", "DDR_LLM_MODEL",
               "DDR_FALLBACK_BASE_URL", "DDR_FALLBACK_MODEL",
               "GROQ_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY",
               "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
               "GOOGLE_API_KEY"):
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass


@st.cache_resource(show_spinner="Loading search index...")
def _warm_index():
    """Build the semantic index ONCE per process (loads precomputed vectors +
    BM25), shared across reruns and sessions, so no query pays the build cost.
    Also primes the query encoder so the first real question is instant too."""
    idx = tool_semantic.get_index()
    if idx.get("embeddings") is not None:
        import embedder
        embedder.encode(["warm up"])      # one-time fastembed model load
    return idx


_warm_index()

# --- light blue/white styling -------------------------------------------------
st.markdown("""
<style>
  /* Only accent colours here. Background and body text are left to the active
     (light or dark) theme, so the app stays readable in either mode. */
  .ddr-title { color:#1f6feb; font-weight:700; font-size:2.0rem; margin-bottom:0; }
  .ddr-sub  { color:#6b8cae; margin-top:0; }
  .ddr-foot { color:#8aa0bc; font-size:0.8rem; text-align:center; margin-top:2.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="ddr-title">DDR-RAG</p>', unsafe_allow_html=True)
st.markdown('<p class="ddr-sub">Ask about the drilling reports and pressure plots.</p>',
            unsafe_allow_html=True)

# clear, early warning if no LLM key is set for any supported provider
import llm as _llm
if not _llm.has_key():
    st.warning(
        "No API key detected. Set a provider key in a `.env` next to app.py "
        "(e.g. `GEMINI_API_KEY=your_key`) or in `.streamlit/secrets.toml`, then "
        "rerun. Presets are listed at the top of llm.py.")


# --- evidence renderer (defined before any use) -------------------------------
def _render_tool(tool, ev):
    if tool == "sql":
        if ev.get("sql"):
            st.code(ev["sql"], language="sql")
        if ev.get("rows"):
            st.dataframe(pd.DataFrame(ev["rows"], columns=ev.get("columns")),
                         width='stretch')
        if ev.get("error"):
            st.warning(ev["error"])
    elif tool == "semantic":
        st.caption(f"retriever: {ev.get('mode', '?')}")
        for h in ev.get("hits", []):
            st.markdown(
                f"**{h.get('wellbore_id')}** - {h.get('period_start')} - "
                f"`{h.get('table')}.{h.get('column')}`\n\n> {h.get('text')}")
        if not ev.get("hits"):
            st.caption("no matching text found")
    elif tool == "plot":
        st.json(ev.get("facts", ev))
    else:
        st.json(ev)


def render_evidence(meta):
    with st.expander(f"How I answered this - tools: {meta['tool']}"):
        if meta.get("reason"):
            st.caption(f"Reasoning: {meta['reason']}")
        steps = meta.get("steps", [])
        if not steps:
            st.caption("Answered directly, no tools called.")
        for i, s in enumerate(steps, 1):
            st.markdown(f"**Step {i} - {s['action']}** · `{s['input']}`")
            _render_tool(s["action"], s["result"])


# --- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.caption("Two retrieval tools (SQL + hybrid semantic search) plus a "
               "deterministic plot-interpretation tool, behind one ReAct agent.")
    try:
        _conn = db.connect()
        _tables = db.list_tables(_conn)
        _conn.close()
        st.success(f"Connected to {db.DB_PATH} ({len(_tables)} tables)")
    except Exception as e:
        st.error(f"No database: {e}")

    st.subheader("Try asking")
    examples = [
        "Which operator and rig were used on wellbore 15/9-19 A?",
        "How many reports are flagged HPHT?",
        "Show reports mentioning a whipstock",
        "Is Well_01 inside the SoR pressure envelope?",
        "Does any measured point exceed virgin pressure for Well_02?",
        "What can you answer?",
    ]
    for ex in examples:
        if st.button(ex, width='stretch'):
            st.session_state.pending = ex

    st.caption("Ingestion of new DDRs/plots is a separate offline step: run "
               "extract_ddr.py + build_database.py + build_plot_tables.py.")


# --- chat state ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# replay history (with evidence)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            render_evidence(msg["meta"])


def handle(question):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # last 5 user/assistant pairs (10 msgs) before the current question
            history = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in st.session_state.messages[-11:-1])
            _llm.reset_fallback()
            result = agent.answer(question, history)
        if _llm.took_fallback():
            st.info(f"Primary model hit its rate limit - answered with the "
                    f"fallback model ({_llm.FALLBACK_MODEL}).")
        st.markdown(result["answer"])
        meta = {"tool": result["tool"], "reason": result["reason"],
                "steps": result["steps"]}
        render_evidence(meta)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "meta": meta})


pending = st.session_state.pop("pending", None)
typed = st.chat_input("Ask a question about the DDRs or pressure plots...")
question = typed or pending
if question:
    handle(question)
