import logging
import re
import requests
from pathlib import Path
from typing import Optional
import os

from lightrag import LightRAG
from config import ollama_config, rag_config, USE_OLLAMA

# ─────────────────────────────────────────────────────────────
# LLM / EMBEDDING IMPORTS
# ─────────────────────────────────────────────────────────────
if USE_OLLAMA:
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
else:
    async def ollama_model_complete(*args, **kwargs):
        raise NotImplementedError("Ollama disabled in cloud.")

# ─────────────────────────────────────────────────────────────
# CORE IMPORTS
# ─────────────────────────────────────────────────────────────
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.prompt import PROMPTS
from prompt_template import KG_EXTRACTION_PROMPT, DEFENCE_ENTITY_TYPES

PROMPTS["entity_extraction_system_prompt"] = KG_EXTRACTION_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL (CACHED)

try:
    import streamlit as st

    @st.cache_resource
    def get_embedding_model():
        from sentence_transformers import SentenceTransformer
        logger.info("Loading HuggingFace nomic model...")
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cpu"
        )

except ImportError:
    def get_embedding_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cpu"
        )
        
# ─────────────────────────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────────────────────────
def hf_embed(texts, *args, **kwargs):
    model = get_embedding_model()

    # Optional safety: truncate long texts
    texts = [t[:2000] for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False
    )

    return embeddings.tolist()


def get_embedding_func() -> EmbeddingFunc:
    if USE_OLLAMA:
        logger.info(f"Embedding function: {ollama_config.embed_model}")
        return EmbeddingFunc(
            embedding_dim=ollama_config.embedding_dim,
            max_token_size=ollama_config.max_tokens_embed,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=ollama_config.embed_model,
                host=ollama_config.host
            )
        )
    else:
        logger.info("Embedding function: HuggingFace Nomic (768 dim)")
        return EmbeddingFunc(
            embedding_dim=768,  # MUST match your index
            max_token_size=2048,
            func=hf_embed
        )


# ─────────────────────────────────────────────────────────────
# INITIALIZE LIGHTRAG
# ─────────────────────────────────────────────────────────────
async def initialize_lightrag() -> Optional[LightRAG]:
    try:
        if USE_OLLAMA:
            Path(rag_config.working_dir).mkdir(parents=True, exist_ok=True)

            rag = LightRAG(
                working_dir=rag_config.working_dir,
                embedding_func=get_embedding_func(),
                llm_model_func=ollama_model_complete,
                llm_model_name=ollama_config.llm_model,
                llm_model_kwargs=ollama_config.llm_model_kwargs,
                default_llm_timeout=300,
                graph_storage="Neo4JStorage",
                vector_storage="FaissVectorDBStorage",
                chunk_token_size=ollama_config.chunk_size,
                chunk_overlap_token_size=ollama_config.chunk_overlap,
                llm_model_max_async=ollama_config.llm_model_max_async,
                embedding_func_max_async=ollama_config.embedding_func_max_async,
                entity_extract_max_gleaning=rag_config.entity_extract_max_gleaning,
                enable_llm_cache=rag_config.enable_llm_cache,
                addon_params={"entity_types": DEFENCE_ENTITY_TYPES},
            )
        else:
            rag = LightRAG(
                working_dir="./lightrag_storage",
                workspace="f16_bot",
                graph_storage="Neo4JStorage",
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                embedding_func=get_embedding_func(),
                llm_model_func=ollama_model_complete,
                chunk_token_size=900,
                enable_llm_cache=False,
                addon_params={"entity_types": DEFENCE_ENTITY_TYPES},
            )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        logger.info(f"✅ LightRAG initialized. (Cloud: {not USE_OLLAMA})")
        return rag

    except Exception as e:
        logger.error(f"❌ LightRAG init failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# CONNECTION CHECK
# ─────────────────────────────────────────────────────────────
def check_ollama_connection() -> bool:
    if not USE_OLLAMA:
        return True
    try:
        response = requests.get(f"{ollama_config.host}/api/tags", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ─────────────────────────────────────────────────────────────
# CONTEXT PARSER
# ─────────────────────────────────────────────────────────────
def parse_context_sources(context_str: str) -> list:
    sources = []
    seen = set()

    page_pattern = re.compile(r"===\s*Page\s*(\d+)\s*\|\s*([^=\n]+?)\s*===")
    for match in page_pattern.finditer(context_str):
        page_num = match.group(1)
        filename = match.group(2).strip()
        key = f"{filename} (pg. {page_num})"
        if key not in seen:
            sources.append(key)
            seen.add(key)

    file_pattern = re.compile(r"\b([\w\-]+\.(?:pdf|txt|md))\b", re.IGNORECASE)
    for match in file_pattern.finditer(context_str):
        fname = match.group(1)
        if fname not in seen:
            sources.append(fname)
            seen.add(fname)

    return sources