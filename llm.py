"""
llm.py - the ONLY place that talks to an LLM. One plain-text call, one JSON
call. Swap provider by setting env vars; no code changes needed.

OpenAI-compatible chat API, which almost every provider speaks, so the same
wrapper works with Gemini, Groq, DeepSeek, Qwen, OpenRouter, or a local Ollama.

Set in your .env (or .streamlit/secrets.toml):
    DDR_LLM_BASE_URL   provider endpoint
    DDR_LLM_MODEL      model name
    DDR_LLM_API_KEY    your key (or a provider-specific key below)

FALLBACK: if the primary provider returns a rate-limit (429), we retry the same
call once on a Groq model and the UI shows a "switched" notice. The default
fallback is a SMALLER, higher-daily-limit Groq model (llama-3.1-8b-instant) so
it has headroom even if the big model is exhausted. Configure with
DDR_FALLBACK_BASE_URL / DDR_FALLBACK_MODEL.

PRESETS (base url / model / key env):
  Gemini  (FREE, ~1500 req/day)  https://generativelanguage.googleapis.com/v1beta/openai/
                           gemini-2.0-flash               GEMINI_API_KEY
  Groq    (FREE, 100k tok/day)  https://api.groq.com/openai/v1
                           llama-3.3-70b-versatile        GROQ_API_KEY
     -> want Qwen instead?  set DDR_LLM_MODEL=qwen/qwen3-32b  (same Groq key)
  OpenRouter (free models) https://openrouter.ai/api/v1
                           deepseek/deepseek-v4-flash:free OPENROUTER_API_KEY
  Ollama  (LOCAL, dev only) http://localhost:11434/v1
                           qwen2.5  (key can be any text)  DDR_LLM_API_KEY=ollama

Default below is Gemini 2.0 Flash (free, high daily request limit).
"""
import json
import os
import re

BASE_URL = os.environ.get(
    "DDR_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ.get("DDR_LLM_MODEL", "gemini-2.0-flash")

# fallback provider, used when the primary hits a rate limit
FALLBACK_BASE_URL = os.environ.get(
    "DDR_FALLBACK_BASE_URL", "https://api.groq.com/openai/v1")
FALLBACK_MODEL = os.environ.get("DDR_FALLBACK_MODEL", "llama-3.1-8b-instant")

# provider (matched by a substring of the base url) -> its key env vars
_PROVIDER_KEYS = [
    ("generativelanguage", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    ("groq", ("GROQ_API_KEY",)),
    ("huggingface", ("HF_TOKEN",)),
    ("openrouter", ("OPENROUTER_API_KEY",)),
    ("deepseek", ("DEEPSEEK_API_KEY",)),
    ("openai", ("OPENAI_API_KEY",)),
]
_ALL_KEYS = ("GROQ_API_KEY", "HF_TOKEN", "OPENROUTER_API_KEY",
             "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
             "GOOGLE_API_KEY")

_client = None
_fallback_client = None
_used_fallback = False                              # this query switched providers
_THINK = re.compile(r"<think>.*?</think>", re.S)   # strip reasoning-model traces


def _provider_key(base_url):
    for needle, vars_ in _PROVIDER_KEYS:
        if needle in base_url:
            for k in vars_:
                v = os.environ.get(k)
                if v:
                    return v
    return None


def _key():
    """DDR_LLM_API_KEY overrides everything; otherwise pick the key that matches
    the active BASE_URL's provider, so an unrelated key can't shadow it."""
    override = os.environ.get("DDR_LLM_API_KEY")
    if override:
        return override
    pk = _provider_key(BASE_URL)
    if pk:
        return pk
    for k in _ALL_KEYS:
        v = os.environ.get(k)
        if v:
            return v
    return None


def has_key():
    return _key() is not None


def took_fallback():
    return _used_fallback


def reset_fallback():
    global _used_fallback
    _used_fallback = False


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI            # lazy import
        key = _key()
        if not key:
            raise RuntimeError(
                "No API key found. Put one in .env, e.g. GEMINI_API_KEY=... "
                "(or DDR_LLM_API_KEY=... for any provider).")
        _client = OpenAI(api_key=key, base_url=BASE_URL)
    return _client


def _get_fallback_client():
    """Groq client for the rate-limit fallback, or None if no Groq key / it is
    the same provider+model as the primary (then fallback would be pointless)."""
    global _fallback_client
    if FALLBACK_BASE_URL == BASE_URL and FALLBACK_MODEL == MODEL:
        return None
    if _fallback_client is None:
        key = _provider_key(FALLBACK_BASE_URL)
        if not key:
            return None
        from openai import OpenAI
        _fallback_client = OpenAI(api_key=key, base_url=FALLBACK_BASE_URL)
    return _fallback_client


def _is_rate_limit(e):
    try:
        from openai import RateLimitError
        if isinstance(e, RateLimitError):
            return True
    except Exception:
        pass
    return getattr(e, "status_code", None) == 429 or "429" in str(e) \
        or "rate limit" in str(e).lower()


def _try_create(client, model, messages, response_format, temperature):
    try:
        return client.chat.completions.create(
            model=model, temperature=temperature, messages=messages,
            **({"response_format": response_format} if response_format else {}))
    except Exception as e:
        if response_format is None or _is_rate_limit(e):
            raise
        # provider didn't accept response_format -> retry without it
        return client.chat.completions.create(
            model=model, temperature=temperature, messages=messages)


def _create(messages, response_format=None, temperature=0.0):
    """Primary call; on a rate limit, switch to the Groq fallback once."""
    global _used_fallback
    try:
        return _try_create(_get_client(), MODEL, messages,
                           response_format, temperature)
    except Exception as e:
        if not _is_rate_limit(e):
            raise
        fb = _get_fallback_client()
        if fb is None:
            raise                                   # no fallback -> surface 429
        _used_fallback = True
        return _try_create(fb, FALLBACK_MODEL, messages,
                           response_format, temperature)


def _clean(text):
    return _THINK.sub("", text or "").strip()


def chat(system, user, temperature=0.0):
    resp = _create([{"role": "system", "content": system},
                    {"role": "user", "content": user}], None, temperature)
    return _clean(resp.choices[0].message.content)


def chat_json(system, user, temperature=0.0):
    """JSON completion -> dict. Asks for JSON mode; the create helper retries
    without response_format if the provider/model doesn't support it."""
    msgs = [{"role": "system",
             "content": system + " Respond with ONE JSON object, nothing else."},
            {"role": "user", "content": user}]
    resp = _create(msgs, {"type": "json_object"}, temperature)
    txt = _clean(resp.choices[0].message.content)
    if txt.startswith("```"):
        txt = txt.strip("`")
        i = txt.find("{")
        if i != -1:
            txt = txt[i:]
    if not txt.startswith("{"):
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            txt = m.group(0)
    return json.loads(txt)
