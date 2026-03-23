
import asyncio
import argparse
import sys
import time
from pathlib import Path

# Always resolve imports relative to project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from lightrag import QueryParam
from utils import initialize_lightrag
from groq_client import run_groq_inference


DEFAULT_QUERIES = [
    "What is the F-16's flight control system?",
    "Explain the APG-68 radar capabilities.",
    "What are the F-16 engine specifications?",
    "Describe the F-16 weapons delivery system.",
]

MODES = ["local", "hybrid", "mix", "global", "naive"]


async def run_test(query: str, mode: str, top_k: int, show_answer: bool):
    print(f"\n{'='*70}")
    print(f"  QUERY : {query}")
    print(f"  MODE  : {mode}  |  TOP-K : {top_k}")
    print(f"{'='*70}")

    print("\n[1/3] Initializing LightRAG (connecting to Neo4j + FAISS index)...")
    t0 = time.perf_counter()
    rag = await initialize_lightrag()
    if not rag:
        print("ERROR: LightRAG initialization failed.")
        print("  → Is Neo4j running?  Check NEO4J_URI / credentials in .env")
        print("  → Did you run:  python build_index.py --docs ./Docs")
        sys.exit(1)
    init_ms = round((time.perf_counter() - t0) * 1000)
    print(f"  ✓ Initialized in {init_ms} ms")

    print("\n[2/3] Retrieving context from knowledge graph...")
    t1 = time.perf_counter()
    try:
        raw_ctx = await rag.aquery(
            query,
            param=QueryParam(
                mode=mode,
                top_k=top_k,
                chunk_top_k=top_k,
                enable_rerank=False,
                only_need_context=True,
            ),
        )
    except Exception as e:
        print(f"  ✗ Retrieval failed: {e}")
        sys.exit(1)
    retrieval_ms = round((time.perf_counter() - t1) * 1000)

    ctx_str = raw_ctx if isinstance(raw_ctx, str) else str(raw_ctx)
    print(f"  ✓ Retrieved in {retrieval_ms} ms")
    print(f"  Context length : {len(ctx_str):,} chars")

    # Try to count sections inside context (entities / relations / chunks)
    sections = {
        "Entities"      : ctx_str.count("-----Entities-----"),
        "Relations"     : ctx_str.count("-----Relationships-----"),
        "Text chunks"   : ctx_str.count("-----Sources-----"),
    }
    for k, v in sections.items():
        if v:
            print(f"  {k} sections found : {v}")

    ctx_preview = ctx_str[:600].replace("\n", " ")
    print(f"\n  Context preview:\n  {ctx_preview}{'...' if len(ctx_str) > 600 else ''}")

    if show_answer:
        print("\n[3/3] Generating answer via Groq...")
        t2 = time.perf_counter()
        try:
            answer, llm_refs, ctx_sources = run_groq_inference(
                rag_context=ctx_str,
                user_query=query,
            )
            gen_ms = round((time.perf_counter() - t2) * 1000)
            print(f"  ✓ Generated in {gen_ms} ms")
            print(f"\n  Answer:\n{'─'*60}")
            print(f"  {answer[:1200]}")
            if len(answer) > 1200:
                print("  ...[truncated]")
            if llm_refs:
                print(f"\n  LLM References: {llm_refs}")
            if ctx_sources:
                print(f"  Context sources: {ctx_sources}")
        except Exception as e:
            print(f"  ✗ Generation failed: {e}")
            print("    → Check GROQ_API_KEY in .env")
    else:
        print("\n[3/3] Skipped answer generation (use --answer to enable)")

    print(f"\n{'='*70}")
    print(f"  Done. Total retrieval: {retrieval_ms} ms | Init: {init_ms} ms")
    print(f"{'='*70}\n")


async def run_all_modes(query: str, top_k: int):
    """Run the same query across all retrieval modes for comparison."""
    print(f"\n{'#'*70}")
    print(f"  MULTI-MODE COMPARISON")
    print(f"  Query: {query}")
    print(f"{'#'*70}")

    rag = await initialize_lightrag()
    if not rag:
        print("ERROR: LightRAG initialization failed.")
        sys.exit(1)

    rows = []
    for mode in MODES:
        t = time.perf_counter()
        try:
            ctx = await rag.aquery(
                query,
                param=QueryParam(mode=mode, top_k=top_k, chunk_top_k=top_k, enable_rerank=False, only_need_context=True),
            )
            ctx_str = ctx if isinstance(ctx, str) else str(ctx)
            elapsed = round((time.perf_counter() - t) * 1000)
            rows.append((mode, len(ctx_str), elapsed, "✓"))
        except Exception as e:
            elapsed = round((time.perf_counter() - t) * 1000)
            rows.append((mode, 0, elapsed, f"✗ {e}"))

    print(f"\n  {'Mode':<10} {'Context (chars)':>16} {'Latency (ms)':>14}  Status")
    print(f"  {'─'*10} {'─'*16} {'─'*14}  {'─'*20}")
    for mode, chars, ms, status in rows:
        print(f"  {mode:<10} {chars:>16,} {ms:>14}  {status}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test LightRAG pre-indexed retrieval")
    p.add_argument("--query",   default="",    help="Query to run (default: runs built-in list)")
    p.add_argument("--mode",    default="local", choices=MODES, help="Retrieval mode")
    p.add_argument("--top-k",   type=int, default=5, help="Top-K entities from KG")
    p.add_argument("--answer",  action="store_true", help="Also generate answer via Groq")
    p.add_argument("--all-modes", action="store_true", help="Compare all modes for a single query")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.all_modes:
        q = args.query or DEFAULT_QUERIES[0]
        asyncio.run(run_all_modes(q, args.top_k))
    elif args.query:
        asyncio.run(run_test(args.query, args.mode, args.top_k, args.answer))
    else:
        # Run all default queries sequentially
        for q in DEFAULT_QUERIES:
            asyncio.run(run_test(q, args.mode, args.top_k, args.answer))