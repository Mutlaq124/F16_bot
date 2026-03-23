import os
from dataclasses import dataclass, field
from typing import List, Dict, Literal
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "True").lower() in ("true", "1")


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "")
    username: str = os.getenv("NEO4J_USERNAME", "")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "")


@dataclass
class OllamaConfig:
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q8_0")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embedding_dim: int = 768 # 768 for maximum quality
    max_tokens_embed: int = 8192
    max_tokens_gen: int = 8192
    # IMPORTANT: Ollama is single-threaded — set async=1 for LLM to avoid timeout cascades.
    # Embedding can handle 2 concurrent since it's much faster than generation.
    llm_model_max_async: int = 1   # 1 = sequential LLM calls (safe for local Ollama)
    embedding_func_max_async: int = 2  # 2 concurrent embedding calls
    chunk_size: int = 800          # internal token chunk size
    chunk_overlap: int = 50
    llm_model_kwargs: Dict = field(default_factory=lambda: {
        "options": {
            "num_ctx": 8192,     # reduced from 32768 — entity extraction needs ~4K max
            "temperature": 0.2,  # low for factual extraction
            "num_predict": 512   # entity JSON output is short, 512 is plenty
        }
    })


@dataclass
class OpenRouterConfig:
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    # qwen/qwen-2.5-coder-32b-instruct:free was retired — use this instead:
    model: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free")
    max_tokens: int = 1024   # keep low to conserve free-tier TPM
    temperature: float = 0.4
    base_url: str = "https://openrouter.ai/api/v1"


@dataclass
class GroqConfig:
    api_key: str = os.getenv("GROQ_API_KEY", "")
    # llama-3.3-70b-versatile: 6,000 TPM / 30 RPM / 14,400 TPD on free tier
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    max_tokens: int = 1024   # ~1K output keeps TPM budget healthy at 6K TPM
    temperature: float = 0.4
    base_url: str = "https://api.groq.com/openai/v1"


@dataclass
class RAGConfig:
    working_dir: str = os.getenv("RAG_WORKING_DIR", "./lightrag_neo4j_storage")
    search_modes: List[Literal["mix", "hybrid", "local", "global", "naive"]] = field(
        default_factory=lambda: ["mix", "hybrid", "local", "global", "naive"]
    )
    default_search_mode: str = "local"
    top_k: int = 3
    entity_extract_max_gleaning: int = 1  # refinement passes per chunk
    enable_llm_cache: bool = True         # cache LLM responses to speed up re-indexing


@dataclass
class AppConfig:
    page_title: str = "F-16 Defence Intelligence Bot"
    allowed_file_types: List[str] = field(default_factory=lambda: ["pdf", "txt", "md"])
    max_history_turns: int = 3


ollama_config = OllamaConfig()
openrouter_config = OpenRouterConfig()
groq_config = GroqConfig()
rag_config = RAGConfig()
app_config = AppConfig()
neo4j_config = Neo4jConfig()

model_config = ollama_config