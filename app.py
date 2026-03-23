"""
Inference-only. Build index first: python build_index.py --docs ./Docs
"""

import streamlit as st
import asyncio
import threading
import time
from lightrag import LightRAG, QueryParam

from config import ollama_config, groq_config, rag_config, app_config
from utils import initialize_lightrag, check_ollama_connection
from groq_client import (
    stream_groq_inference,
    check_groq_connection,
    extract_references_from_answer,
    extract_page_refs_from_context
)
from groq_client import (
    stream_groq_inference,
    check_groq_connection,
)

st.set_page_config(
    page_title=app_config.page_title,
    layout="wide",
    initial_sidebar_state="expanded",
)

import os

st.markdown("""
<style>
/* ─── Google Fonts ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── CRITICAL FIX: Material Symbols Rounded font for Streamlit expander icons ── */
/* Without this, "keyboard_arrow_right" / "keyboard_arrow_down" show as raw text  */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

/* Fix ALL Streamlit material icons globally */
.material-symbols-rounded,
span[class*="material-symbol"] {
    font-family: "Material Symbols Rounded" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 18px !important;
    line-height: 1 !important;
}

/* ─── Base ──────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
}

.stApp {
    background: linear-gradient(160deg, #0d1117 0%, #111520 55%, #0d1117 100%);
    color: #d4dce8;
}

/* ─── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1422 0%, #111829 100%);
    border-right: 1px solid #1e2d42;
    width: 330px !important;
}

/* Apply Inter to sidebar but NOT to icon spans — keeps collapse button rendering correctly */
section[data-testid="stSidebar"] *:not(.material-symbols-rounded):not(span[class*="material-symbol"]) {
    font-family: 'Inter', sans-serif;
}

/* Sidebar collapse / expand button — force Material Symbols so it shows ☰ / ✕, not "key" */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span,
button[data-testid="baseButton-headerNoPadding"] span {
    font-family: "Material Symbols Rounded" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 20px !important;
    line-height: 1 !important;
}

/* ─── Hide Streamlit keyboard shortcut hints ─────────────── */
[data-testid="InputInstructions"] { display: none !important; }

/* ─── Hero ──────────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #131d30 0%, #1a2840 50%, #131d30 100%);
    border: 1px solid #253650;
    border-radius: 14px;
    padding: 36px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -5%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(74,144,226,0.08) 0%, transparent 65%);
    animation: breathe 6s ease-in-out infinite;
}

@keyframes breathe {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.08); opacity: 1; }
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem; font-weight: 700;
    color: #eaf0fb; margin: 0 0 10px 0;
    letter-spacing: -0.7px; line-height: 1.2;
}

.hero-subtitle { font-size: 1.0rem; color: #7a9bc0; line-height: 1.5; }

.hero-tagline {
    font-size: 0.78rem; color: #6a90b8; margin-top: 12px;
    letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600;
}

/* ─── Status badges ────────────────────────────────────── */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 22px;
    font-size: 0.82rem; font-weight: 500; margin: 3px 0;
    width: 100%; box-sizing: border-box;
}

.status-online { background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.25); color: #34d399; }
.status-offline { background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.25); color: #f87171; }

.status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dot-online { background: #34d399; box-shadow: 0 0 6px #34d399; animation: dotglow 2.5s ease-in-out infinite; }
@keyframes dotglow { 0%, 100% { box-shadow: 0 0 4px #34d399; } 50% { box-shadow: 0 0 10px #34d399; } }
.dot-offline { background: #ef4444; }

/* ─── Sidebar text ─────────────────────────────────────── */
.sb-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700; color: #7eb8f7; letter-spacing: -0.3px;
}
.sb-sub { font-size: 0.78rem; color: #6a8aaa; margin-top: 2px; margin-bottom: 18px; }
.sb-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem; font-weight: 700; color: #4a90e2;
    text-transform: uppercase; letter-spacing: 1.3px; margin: 4px 0 10px 0;
}

/* ─── Metric cards ─────────────────────────────────────── */
.metric-card {
    background: rgba(18, 27, 44, 0.65);
    border: 1px solid #1e2d42; border-radius: 12px;
    padding: 18px 20px; text-align: center;
    transition: border-color 0.22s ease, transform 0.2s ease;
}

.metric-card:hover {
    border-color: #4a90e2;
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(74,144,226,0.18);
}

.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; color: #7eb8f7; }
.metric-label { font-size: 0.72rem; color: #6a8aaa; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; font-weight: 600; }
.mode-badge {
    background: rgba(74,144,226,0.13); border: 1px solid rgba(74,144,226,0.28); color: #7eb8f7;
    padding: 4px 12px; border-radius: 14px; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
}

/* ─── Reference panel ───────────────────────────────── */
.ref-item {
    background: rgba(30, 45, 66, 0.5); border-left: 3px solid #4a90e2;
    border-radius: 4px; padding: 7px 10px; margin-bottom: 6px;
    font-size: 0.82rem; color: #7eb8f7; line-height: 1.4; word-break: break-word;
}
.ref-item-page { border-left-color: #10b981; color: #6ee7b7; }

.context-raw {
    background: rgba(8, 12, 22, 0.9); border: 1px solid #1e2d42;
    border-radius: 7px; padding: 12px;
    font-family: 'Courier New', monospace; font-size: 0.74rem; color: #4a6080;
    max-height: 240px; overflow-y: auto; white-space: pre-wrap; word-break: break-word;
}

/* ─── Divider ──────────────────────────────────────── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2d42 30%, #1e2d42 70%, transparent);
    margin: 18px 0;
}

/* ─── Model info rows ──────────────────────────────── */
.model-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #111d2e; gap: 8px; }
.model-row:last-child { border-bottom: none; }
.model-lbl { font-size: 0.78rem; color: #7a9bc0; font-weight: 600; flex-shrink: 0; }
.model-val { font-size: 0.78rem; color: #7eb8f7; text-align: right; word-break: break-all; }

/* ─── Mode guide rows ──────────────────────────────── */
.mode-row { padding: 9px 0; border-bottom: 1px solid #111d2e; }
.mode-row:last-child { border-bottom: none; }
.mode-name { font-size: 0.82rem; font-weight: 600; color: #7eb8f7; }
.mode-desc { font-size: 0.76rem; color: #7a9bc0; margin-top: 3px; line-height: 1.45; }

/* ─── Expanders ────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: rgba(14, 22, 36, 0.45) !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 9px !important;
    margin-bottom: 10px !important;
}
div[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    color: #7a9bc0 !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
}

/* Fix Streamlit expander arrow icons */
div[data-testid="stExpander"] summary span {
    font-family: "Material Symbols Rounded" !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-size: 18px !important;
}

/* ─── Buttons ──────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1a3560, #1f4070) !important;
    color: #d0e4ff !important; border: 1px solid #253e5e !important;
    border-radius: 9px !important; font-weight: 500 !important;
    font-size: 0.88rem !important; transition: all 0.2s ease !important;
    padding: 10px 16px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1f4070, #2a5090) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(74,144,226,0.22) !important;
    border-color: #4a90e2 !important;
}

/* ─── Form elements ────────────────────────────────── */
.stSelectbox label, .stSlider label {
    font-size: 0.85rem !important; color: #7a9bc0 !important; font-weight: 500 !important;
}

/* ─── Chat messages ────────────────────────────────── */
.stChatMessage {
    background: rgba(10, 18, 32, 0.7) !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 11px !important;
    padding: 14px 18px !important; margin-bottom: 10px !important;
}
.stChatMessage p { font-size: 1.02rem !important; line-height: 1.8 !important; color: #d4dce8 !important; }

/* ─── Chat input box ───────────────────────────────── */
/*
  MANUAL STYLING GUIDE for the chat input bar:
  Primary container:  div[data-testid="stChatInput"]
  Textarea element:   div[data-testid="stChatInput"] textarea
  Send button:        div[data-testid="stChatInput"] button
  To change height:   set min-height on the textarea
  To change width:    the container fills the column width; control via st.columns()
*/
div[data-testid="stChatInput"] {
    border: 1px solid #253650 !important;
    border-radius: 12px !important;
    background: rgba(10, 18, 32, 0.9) !important;
    position: sticky !important;
    bottom: 0 !important;
    z-index: 999 !important;
}
div[data-testid="stChatInput"] textarea {
    font-size: 0.97rem !important;
    color: #d4dce8 !important;
    min-height: 52px !important;
    height: 52px !important;
    padding: 14px 52px 14px 18px !important;
    line-height: 1.5 !important;
    resize: none !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stChatInput"] button {
    color: #4a90e2 !important;
}
div[data-testid="stChatInput"] button:hover { color: #7eb8f7 !important; }

/* ─── Alerts + Spinner ─────────────────────────────── */
.stAlert { border-radius: 9px !important; font-size: 0.88rem !important; }
.stSpinner p { font-size: 0.88rem !important; color: #7eb8f7 !important; }

/* ─── Scrollable main content area ─────────────────── */
section.main > div {
    max-height: calc(100vh - 260px);
    overflow-y: auto;
}

/* ─── Scrollbars ───────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e2d42; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4a90e2; }
</style>
""", unsafe_allow_html=True)

# ─── Background async loop ────────────────────────────────────────────────────
if "bg_loop" not in st.session_state:
    bg_loop = asyncio.new_event_loop()
    def _runner(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
    bg_thread = threading.Thread(target=_runner, args=(bg_loop,), daemon=True)
    bg_thread.start()
    st.session_state.bg_loop = bg_loop


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, st.session_state.bg_loop).result()


@st.cache_resource
def get_rag_instance():
    return run_async(initialize_lightrag())


# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("total_queries", 0),
    ("last_ctx_sources", []),
    ("last_llm_refs", []),
    ("last_raw_ctx", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

ollama_ok = check_ollama_connection()
groq_ok = check_groq_connection()
rag_instance = get_rag_instance()
# ─── DEBUG BLOCK ────────────────────────────────────────────────
with st.sidebar.expander("DEBUG: Neo4j Connection Status", expanded=True):
    try:
        from neo4j import GraphDatabase
        import os
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        pwd = os.getenv("NEO4J_PASSWORD")
        db = os.getenv("NEO4J_DATABASE", "neo4j")
        
        st.write("**Connection Parameters Used**")
        st.write(f"URI: {uri}")
        st.write(f"Username: {user}")
        st.write(f"Database: {db}")
        st.write(f"Password length: {len(pwd) if pwd else 0}")
        
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        
        def test_query(tx):
            result = tx.run("MATCH (n) RETURN count(n) AS cnt")
            return result.single()["cnt"]
        
        with driver.session(database=db) as session:
            count = session.execute_read(test_query)
            st.success(f"✅ Connected! Aura has {count} nodes")
            st.write("If count is ~42 → data is there")
            
        driver.close()
        
    except Exception as e:
        st.error(f"❌ Connection FAILED")
        st.code(str(e), language="text")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">F-16 Intelligence Bot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">System Control Panel</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-title">System Status</div>', unsafe_allow_html=True)
    for label, ok, ok_txt in [
        ("Ollama (Build LLM)", ollama_ok, "Connected"),
        ("Groq LLaMA 70B (Inference)", groq_ok, "Ready"),
        ("Knowledge Graph Index", bool(rag_instance), "Loaded"),
    ]:
        cls = "online" if ok else "offline"
        txt = ok_txt if ok else "Offline / Missing"
        st.markdown(f"""
        <div class="status-badge status-{cls}">
            <span class="status-dot dot-{cls}"></span>
            <span>{label} &nbsp;—&nbsp; {txt}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-title">Configuration Parameters</div>', unsafe_allow_html=True)

    # ── Custom Groq API Key & Model ──────────────────────────────────
    with st.expander("Custom Groq Key & Model", expanded=False):
        st.markdown(
            '<div style="font-size:0.76rem;color:#7a9bc0;margin-bottom:8px;">'
            'Override the default API key and model from <code>.env</code>. ',
            unsafe_allow_html=True,
        )
        custom_api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.get("custom_api_key", ""),
            type="password",
            placeholder="gsk_...",
            key="custom_api_key_input",
        )
        custom_model = st.text_input(
            "Model Name",
            value=st.session_state.get("custom_model", ""),
            placeholder="e.g. llama-3.3-70b-versatile",
            key="custom_model_input",
        )
        if st.button("Apply Key & Model", use_container_width=True):
            if custom_api_key:
                st.session_state["custom_api_key"] = custom_api_key
            if custom_model:
                st.session_state["custom_model"] = custom_model
            st.success("Applied! Will use on next query.")

    with st.expander("Retrieval", expanded=True):
        search_mode = st.selectbox(
            "Retrieval Mode",
            options=rag_config.search_modes,
            index=rag_config.search_modes.index(rag_config.default_search_mode),
        )
        if search_mode == "naive":
            st.warning("Naive mode may not work on cloud")
            search_mode = "local"
        top_k = st.slider("Top-K Entities", 1, 10, rag_config.top_k,
                          help="Entities + relationships retrieved from KG per query")

    with st.expander("Mode Guide", expanded=False):
        for name, desc in [
            ("mix", "Local + Global KG combined - Multi-hop F-16 queries."),
            ("hybrid", "Local KG entities + vector similarity - Balanced precision & recall."),
            ("local", "Entity-centric graph walk - Specific system specs or precise facts."),
            ("global", "Concept-level cluster retrieval - Broad topic summaries."),
            ("naive", "Pure vector similarity - General factual lookups"),
        ]:
            st.markdown(f"""
            <div class="mode-row">
                <div class="mode-name">{name}</div>
                <div class="mode-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Pipeline Models ", expanded=False):
        st.markdown(f"""
        <div>
            <div class="model-row">
                <span class="model-lbl">Build LLM</span>
                <span class="model-val">{ollama_config.llm_model}</span>
            </div>
            <div class="model-row">
                <span class="model-lbl">Inference LLM</span>
                <span class="model-val">{groq_config.model}</span>
            </div>
            <div class="model-row">
                <span class="model-lbl">Embeddings</span>
                <span class="model-val">{ollama_config.embed_model}</span>
            </div>
            <div class="model-row">
                <span class="model-lbl">Chunk Size</span>
                <span class="model-val">{ollama_config.chunk_size} tokens</span>
            </div>
            <div class="model-row">
                <span class="model-lbl">Context Window</span>
                <span class="model-val">{ollama_config.llm_model_kwargs["options"]["num_ctx"]:,} tokens</span>
            </div>
            <div class="model-row">
                <span class="model-lbl">Groq Max Output</span>
                <span class="model-val">{groq_config.max_tokens:,} tokens</span>
            </div>
            <div class="model-row">
                <span class="model-lbl">Groq Temperature</span>
                <span class="model-val">{groq_config.temperature}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_ctx_sources = []
        st.session_state.last_llm_refs = []
        st.session_state.last_raw_ctx = ""
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("""
    <div style="margin-top:16px; padding:12px 14px; background:rgba(14,22,36,0.6);
                border-radius:9px; border:1px solid #1e2d42;">
        <div style="font-size:0.72rem; color:#2e4060; text-align:center; line-height:1.9;">
            LightRAG Knowledge Graph RAG<br>        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">F-16 Defence Intelligence Bot</div>
    <div class="hero-subtitle">Knowledge Graph-Augmented Intelligence System</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")
for col, value, label in [
    (col1, str(st.session_state.total_queries), "Queries Processed"),
    (col2, str(len(st.session_state.messages) // 2), "Conversation Turns"),
    (col3, str(len(st.session_state.last_ctx_sources)), "Sources Last Query"),
    (col4, f'<span class="mode-badge">{search_mode}</span>', "Active Mode"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if not rag_instance:
    st.error("Knowledge graph not loaded. Run: `python build_index.py --docs ./Docs` then restart.")
if not groq_ok:
    st.error("No inference backend configured. Add GROQ_API_KEY to .env and restart.")

chat_col, ref_col = st.columns([3, 1], gap="large")

with chat_col:
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    all_refs = list(dict.fromkeys(
                        (msg.get("ctx_sources") or []) + (msg.get("llm_refs") or [])
                    ))
                    if all_refs:
                        with st.expander(f"Source Attribution ({len(all_refs)})", expanded=False):
                            for r in all_refs:
                                cls = "ref-item ref-item-page" if ("pg." in r or "Page" in r) else "ref-item"
                                st.markdown(f'<div class="{cls}">{r}</div>', unsafe_allow_html=True)

with ref_col:
    st.markdown('<div style="font-size:0.82rem;font-weight:700;color:#4a90e2;text-transform:uppercase;letter-spacing:1.1px;margin-bottom:14px;">Last Query Sources</div>', unsafe_allow_html=True)

    if st.session_state.last_ctx_sources or st.session_state.last_llm_refs:
        if st.session_state.last_ctx_sources:
            st.markdown('<div style="font-size:0.72rem;color:#10b981;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;font-weight:600;">Document Pages</div>', unsafe_allow_html=True)
            for s in st.session_state.last_ctx_sources:
                st.markdown(f'<div class="ref-item ref-item-page">{s}</div>', unsafe_allow_html=True)
        if st.session_state.last_llm_refs:
            st.markdown('<div style="font-size:0.72rem;color:#7eb8f7;text-transform:uppercase;letter-spacing:0.8px;margin:10px 0 6px 0;font-weight:600;">Model Citations</div>', unsafe_allow_html=True)
            for r in st.session_state.last_llm_refs:
                st.markdown(f'<div class="ref-item">{r}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:0.82rem; color:#4a90e2; padding:14px;
                    border:1px dashed #1e2d42; border-radius:8px; text-align:center; line-height:1.7;">
            Sources and page references will appear here after a query.
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.last_raw_ctx:
        with st.expander("Raw KG Context", expanded=False):
            preview = st.session_state.last_raw_ctx[:3000]
            if len(st.session_state.last_raw_ctx) > 3000:
                preview += "\n... [truncated]"
            st.markdown(f'<div class="context-raw">{preview}</div>', unsafe_allow_html=True)


# ─── Chat input + streaming inference ────────────────────────────────────────
if user_prompt := st.chat_input("Ask about F-16 systems, features, specifications..."):
    _backend_ready = groq_ok
    # Also accept a custom key override as "ready"
    if st.session_state.get("custom_api_key"):
        _backend_ready = True

    if not _backend_ready:
        st.error("Groq API key not configured. Add key to .env or use 'Custom Groq Key & Model' in the sidebar.")
    elif not rag_instance:
        st.error("Knowledge base not loaded. Build the index first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        with chat_col:
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                # ── Phase 1: KG retrieval ─────────────────────────────────────
                with st.spinner("Retrieving from knowledge graph..."):
                    try:
                        raw_ctx = run_async(
                            rag_instance.aquery(
                                user_prompt,
                                param=QueryParam(
                                    mode=search_mode,
                                    top_k=top_k,
                                    chunk_top_k=top_k,
                                    enable_rerank=False,
                                    only_need_context=True,
                                )
                            )
                        )
                    except Exception as e:
                        st.error(f"Knowledge graph retrieval failed: {e}")
                        st.stop()

                # ── Phase 2: Backend streaming generation ────────────────────
                try:
                    _api_key = st.session_state.get("custom_api_key") or None
                    _model   = st.session_state.get("custom_model")   or None

                    _spinner_msg = f"Generating via Groq ({_model or groq_config.model})..."
                    _stream_fn = lambda: stream_groq_inference(
                        rag_context=raw_ctx,
                        user_query=user_prompt,
                        conversation_history=st.session_state.messages[:-1],
                        api_key_override=_api_key,
                        model_override=_model,
                    )

                    full_response = st.write_stream(_stream_fn()) or ""
                    raw_ctx = raw_ctx or ""

                    answer, llm_refs = extract_references_from_answer(full_response)
                    ctx_sources = extract_page_refs_from_context(raw_ctx)
                    answer = answer or ""
                    
                    # Prevent listing useless random sources for out-of-box queries
                    if "I cannot answer that question" in answer:
                        ctx_sources = []
                        llm_refs = []

                    # Show source attribution inline under the streamed message
                    all_refs = list(dict.fromkeys(ctx_sources + llm_refs))
                    if all_refs:
                        with st.expander(f"Source Attribution ({len(all_refs)})", expanded=False):
                            for r in all_refs:
                                cls = "ref-item ref-item-page" if ("pg." in r or "Page" in r) else "ref-item"
                                st.markdown(f'<div class="{cls}">{r}</div>', unsafe_allow_html=True)

                    # Persist to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "ctx_sources": ctx_sources,
                        "llm_refs": llm_refs,
                    })
                    st.session_state.last_ctx_sources = ctx_sources
                    st.session_state.last_llm_refs = llm_refs
                    st.session_state.last_raw_ctx = raw_ctx
                    st.session_state.total_queries += 1
                    st.rerun()

                except Exception as e:
                    err = f"Generation error: {e}"
                    st.error(err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err,
                        "ctx_sources": [],
                        "llm_refs": [],
                    })