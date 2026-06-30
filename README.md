# DDR-RAG

A stupid-simple chatbot over Daily Drilling Reports (DDRs) and pressure plots.
One chat box, three tools behind it.

## What it does

A question goes through three explicit steps (see `agent.py`):

1. **route** - one LLM call picks ONE tool.
2. **dispatch** - run the tool, collect evidence.
3. **synthesize** - one LLM call writes the answer grounded only in that evidence.

The tools:

- **`tool_sql.py` (construct_sql)** - Text2SQL for anything answerable from
  columns: counts, specific values, per-well rollups, exact-filter lookups.
  Grounded in `data_dictionary.py` so it respects the `-999.99` sentinel, the
  no-join boundary between DDR and plot tables, placeholder dates, etc.
  Output is validated to be a single read-only `SELECT`; the DB is opened
  read-only as well.
- **`tool_semantic.py` (semantic_search)** - hybrid retrieval over free-text
  fields (activity summaries, operation remarks, ...). BM25 + dense vectors,
  fused with Reciprocal Rank Fusion (RRF). Indexing unit is one row, carrying
  `report_id` / `wellbore_id` / period as metadata.
- **`tool_plot.py` (interpret_plot)** - deterministic plot interpretation:
  interpolates the MIN/BASE/MAX SoR curves at each measured depth, classifies
  each point, and compares to virgin pressure. No LLM, no guessing.

## The two data islands

DDR tables (`reports` + children) are keyed by `report_id` / `wellbore_id`
("15/9-19 A"). Plot tables (`formation_*`, `timeseries_*`) are keyed by
`plot_id` / `well_label` ("Well_01".."Well_04"). They share no key and are
**never joined**. The data dictionary states this and the router treats a
question as belonging to one island or the other.

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key           # free, no card: console.groq.com/keys
# make sure ddr.db is in this folder (or: export DDR_DB_PATH=/path/to/ddr.db)
streamlit run app.py
```

The dense embedding model (`all-MiniLM-L6-v2`) downloads once on first
semantic search; if it cannot load, semantic search degrades to BM25-only so
the demo still runs.

## Swapping the LLM

Everything LLM goes through `llm.py`, which uses the OpenAI-compatible chat
API. Switch provider with env vars only - no code change:

```
DDR_LLM_BASE_URL   provider endpoint
DDR_LLM_MODEL      model name
DDR_LLM_API_KEY    your key  (or a provider key like GROQ_API_KEY / HF_TOKEN)
```

Default is Groq (`https://api.groq.com/openai/v1`, `llama-3.3-70b-versatile`) -
free, no credit card, and it hosts open-weight models. Presets for Qwen
(`qwen/qwen3-32b` on Groq), the Hugging Face router, OpenRouter free DeepSeek,
and local Ollama are listed at the top of `llm.py`.

## A note on deployment

The LLM must be a **hosted API reachable over the network** (Groq, HF router,
OpenRouter, ...). A model running locally via Ollama works in development but
NOT once the app is deployed to Streamlit Community Cloud, because the app then
runs on Streamlit's CPU-only server, which can neither reach your laptop nor
run a multi-GB model itself. The embeddings (small MiniLM) run in-process and
are fine on Streamlit Cloud; if memory is tight there, semantic search falls
back to BM25-only automatically.

## Files

```
app.py              Streamlit chat UI (blue/white)
agent.py            route -> dispatch -> synthesize
tool_sql.py         construct_sql (Text2SQL, read-only)
tool_semantic.py    semantic_search (BM25 + vectors + RRF)
tool_plot.py        interpret_plot (deterministic envelope/virgin checks)
data_dictionary.py  column meanings + sentinel/island/placeholder rules
db.py               read-only sqlite helpers
llm.py              swappable LLM wrapper (Gemini default)
.streamlit/config.toml   theme
requirements.txt
```