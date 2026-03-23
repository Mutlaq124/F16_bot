"""
build_index.py

Standalone CLI to build the LightRAG Knowledge Graph index from source documents.
Run ONCE before launching the Streamlit app. Supports structured (heading-based) chunking.
"""

import asyncio
import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import List

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.prompt import PROMPTS

from config import ollama_config, rag_config
from prompt_template import KG_EXTRACTION_PROMPT, DEFENCE_ENTITY_TYPES
from extractor import extract_document, TextChunk

PROMPTS["entity_extraction_system_prompt"] = KG_EXTRACTION_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# F-16 specific boilerplate patterns (regex-resilient with [A-Z/\-]* variant matching)
F16_EXTRA_PATTERNS = [
    r"TO 1F-16[A-Z/\-]*\s+BMS\n\d+\nCHANGE [\d\.]+",  # any manual version
    r"BMS F-16[A-Z/\-]*\s+FLIGHT MANUAL[^\n]*",
]


def save_chunks(chunks: List[TextChunk], doc_name: str, output_dir: Path):
    """Save extracted chunks to JSON for inspection before indexing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(doc_name).stem
    out_file = output_dir / f"{stem}_chunks.json"
    data = [
        {
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
            "source_file": c.source_file,
            "heading": c.heading.strip("# "),
            "char_count": len(c.content),
            "word_count": len(c.content.split()),
            "content_preview": c.content[:300] + "..." if len(c.content) > 300 else c.content,
            "full_content": c.content,
        }
        for c in chunks
    ]
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"  Chunks saved to: {out_file} ({len(chunks)} chunks)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build LightRAG Knowledge Graph index from local documents."
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="./Docs",
        help="Path to directory containing source documents (default: ./Docs)"
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        default=["pdf", "txt", "md"],
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete existing index and rebuild from scratch"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=rag_config.working_dir,
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=150,
        help="Minimum characters for a heading section to be kept as a chunk (default: 150)"
    )
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        default=False,
        help="Save extracted chunks to ./chunks/ folder for inspection before indexing"
    )
    return parser.parse_args()


async def build_index(
    docs_path: Path,
    file_types: List[str],
    working_dir: str,
    reset: bool,
    min_chunk_chars: int,
    save_chunks_flag: bool = False,
):
    wd = Path(working_dir)

    if reset and wd.exists():
        logger.warning(f"Resetting existing index at {wd}")
        shutil.rmtree(wd)

    wd.mkdir(parents=True, exist_ok=True)
    logger.info(f"Index directory: {wd.resolve()}")

    embedding_func = EmbeddingFunc(
        embedding_dim=ollama_config.embedding_dim,
        max_token_size=ollama_config.max_tokens_embed,
        func=lambda texts: ollama_embed(
            texts,
            embed_model=ollama_config.embed_model,
            host=ollama_config.host
        )
    )

    rag = LightRAG(
        working_dir=str(wd),
        embedding_func=embedding_func,
        llm_model_func=ollama_model_complete,
        llm_model_name=ollama_config.llm_model,
        llm_model_kwargs=ollama_config.llm_model_kwargs,
        default_llm_timeout=300,                                          # 5 min per LLM call (sequential Ollama)
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
        chunk_token_size=ollama_config.chunk_size,            # 800 tokens
        chunk_overlap_token_size=ollama_config.chunk_overlap, # 50 tokens
        llm_model_max_async=ollama_config.llm_model_max_async,            # 1 (sequential for local Ollama)
        embedding_func_max_async=ollama_config.embedding_func_max_async,  # 2 (embed is fast)
        entity_extract_max_gleaning=rag_config.entity_extract_max_gleaning,
        enable_llm_cache=rag_config.enable_llm_cache,
        addon_params={"entity_types": DEFENCE_ENTITY_TYPES},
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    logger.info("LightRAG initialized — starting document ingestion")

    target_ext = {f".{e.lstrip('.')}" for e in file_types}
    doc_files = sorted([
        p for p in docs_path.rglob("*")
        if p.is_file() and p.suffix.lower() in target_ext
    ])

    if not doc_files:
        logger.error(f"No documents found in {docs_path} with extensions {file_types}")
        return

    logger.info(f"Found {len(doc_files)} document(s)")
    total_chunks = 0
    success_docs = 0

    for i, doc_path in enumerate(doc_files, 1):
        logger.info(f"[{i}/{len(doc_files)}] {doc_path.name}")

        chunks: List[TextChunk] = extract_document(
            path=doc_path,
            extra_clean_patterns=F16_EXTRA_PATTERNS,
        )

        if not chunks:
            logger.warning(f"  No chunks produced from {doc_path.name}, skipping")
            continue

        # Optional: save chunks to disk for inspection
        if save_chunks_flag:
            save_chunks(chunks, doc_path.name, Path("./chunks"))

        # Batched insertion: compile all texts then call ainsert ONCE per document.
        # LightRAG's internal task queue handles concurrency — much faster than per-chunk calls.
        all_texts = [c.to_indexed_text() for c in chunks if c.to_indexed_text().strip()]
        if not all_texts:
            logger.warning(f"  All chunks empty after formatting for {doc_path.name}, skipping")
            continue

        try:
            await rag.ainsert(
                all_texts,
                file_paths=[doc_path.name] * len(all_texts)
            )
            doc_success = len(all_texts)
            logger.info(f"  Batch-inserted {doc_success} chunks from {doc_path.name}")
        except Exception as e:
            logger.error(f"  Batch insert failed for {doc_path.name}: {e}")
            doc_success = 0

        total_chunks += doc_success
        if doc_success > 0:
            success_docs += 1

    logger.info(f"Build complete. {success_docs}/{len(doc_files)} docs, {total_chunks} total chunks indexed.")
    logger.info(f"Index: {wd.resolve()}")
    logger.info("Launch app: streamlit run app.py")


def main():
    args = parse_args()
    docs_path = Path(args.docs)

    if not docs_path.exists():
        logger.error(f"Documents path does not exist: {docs_path.resolve()}")
        return

    asyncio.run(build_index(
        docs_path=docs_path,
        file_types=args.file_types,
        working_dir=args.working_dir,
        reset=args.reset,
        min_chunk_chars=args.min_chunk_chars,
        save_chunks_flag=args.save_chunks,
    ))


if __name__ == "__main__":
    main()
