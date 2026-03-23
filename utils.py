import logging
import re
import requests
from pathlib import Path
from typing import Optional

from lightrag import LightRAG
from config import ollama_config, rag_config, USE_OLLAMA

if USE_OLLAMA:
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
else:
    # Dummy async functions to satisfy LightRAG when deployed on Streamlit without local daemon
    async def ollama_model_complete(*args, **kwargs):
        raise NotImplementedError("Ollama is disabled in Cloud. Use Groq/OpenRouter.")
    async def ollama_embed(texts, *args, **kwargs):
        return [[0.0] * ollama_config.embedding_dim for _ in texts]
        
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.prompt import PROMPTS

from lightrag.prompt import PROMPTS

from prompt_template import KG_EXTRACTION_PROMPT, DEFENCE_ENTITY_TYPES

PROMPTS["entity_extraction_system_prompt"] = KG_EXTRACTION_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def get_embedding_func() -> EmbeddingFunc:
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


import os

async def initialize_lightrag() -> Optional[LightRAG]:
    try:
        if USE_OLLAMA:
            # Local mode with FAISS & Embedder
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
            # User provided GRAPH-ONLY cloud deployment snippet
            rag = LightRAG(
                working_dir="./lightrag_storage",           # dummy folder (won't be used much)
                workspace="f16_bot",
                
                # Force Neo4j for graph + KV
                graph_storage="Neo4JStorage",
                kv_storage="JsonKVStorage", #in-memory fine for cloud
                
                # Disable local vector storage at query time (we only need graph)
                vector_storage="NanoVectorDBStorage", # in-memory vector storage(no files)
                
                # no llm at query time 
                embedding_func=None,          
                llm_model_func=None,          
                
                chunk_token_size=900,
                enable_llm_cache=False,
                addon_params={"entity_types": DEFENCE_ENTITY_TYPES},
            )

        await rag.initialize_storages()
        await initialize_pipeline_status()
        logger.info(f"✅ LightRAG initialized. (Graph-Only Mode: {not USE_OLLAMA})")
        return rag

    except Exception as e:
        logger.error(f"❌ LightRAG init failed: {e}")
        return None


def check_ollama_connection() -> bool:
    if not USE_OLLAMA:
        return True # Pretend it's ok to bypass UI blocks
    try:
        response = requests.get(f"{ollama_config.host}/api/tags", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def parse_context_sources(context_str: str) -> list:
    """
    Parse source file names and page numbers from LightRAG raw context strings.
    Looks for patterns like '=== Page 3 | F16_manual.pdf ===' embedded in chunk text.
    """
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