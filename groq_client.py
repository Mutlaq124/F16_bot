"""
"""

import logging
from typing import Optional, Tuple, List, Generator

from openai import OpenAI

from config import groq_config
from prompt_template import get_qa_system_prompt, get_generator_prompt
import re

logger = logging.getLogger(__name__)

def extract_references_from_answer(raw_answer: str) -> Tuple[str, List[str]]:
    """Finds [1], [2] type references in the response if the LLM provided them."""
    refs = list(set(re.findall(r'\[\d+\]', raw_answer)))
    return raw_answer, refs

def extract_page_refs_from_context(context: str) -> List[str]:
    """Finds page/source references from LightRAG context."""
    sources = []
    if "-----Sources-----" in context:
        sources_part = context.split("-----Sources-----")[-1]
        for line in sources_part.splitlines():
            line = line.strip()
            if not line or line.startswith("===") or line.startswith("---"):
                continue
            # Usually LightRAG CSV or TSV format: id, text, etc.
            # We'll just grab everything before a tab or comma if it looks like a file/page.
            if ".pdf" in line.lower() or "page" in line.lower() or "pg" in line.lower():
                sources.append(line[:60])
                
    # fallback regex for page numbers
    if not sources:
        sources = list(set(re.findall(r'(?i)(page\s*\d+|pg\.?\s*\d+)', context)))
        
    return list(set(sources))


def get_groq_client(api_key_override: Optional[str] = None) -> OpenAI:
    """Return an OpenAI-compatible client pointed at Groq."""
    return OpenAI(
        api_key=api_key_override or groq_config.api_key,
        base_url=groq_config.base_url,
    )


def check_groq_connection() -> bool:
    """Return True if GROQ_API_KEY is set and reachable."""
    if not groq_config.api_key:
        return False
    try:
        client = get_groq_client()
        client.models.list()
        return True
    except Exception:
        return False


def _build_messages(rag_context: str, user_query: str, conversation_history: Optional[list]) -> list:
    """Shared message builder (same structure as openrouter_client)."""
    messages = [{"role": "system", "content": get_qa_system_prompt()}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": get_generator_prompt(context=rag_context, query=user_query)})
    return messages


# ─── Non-streaming inference ──────────────────────────────────────────────────
def run_groq_inference(
    rag_context: str,
    user_query: str,
    conversation_history: Optional[list] = None,
    api_key_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Tuple[str, List[str], List[str]]:
    """
    Full (non-streaming) Groq inference.
    Returns: (answer, llm_references, context_sources)
    Used by eval_script.py and test.py --answer flag.

    Rate limit tip: max_tokens=1024 with 6K TPM means ~6 concurrent calls max.
    """
    # Using local helper functions

    client = get_groq_client(api_key_override=api_key_override)
    _model = model_override or groq_config.model
    messages = _build_messages(rag_context, user_query, conversation_history)

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=messages,
            max_tokens=groq_config.max_tokens,
            temperature=groq_config.temperature,
        )
        raw = response.choices[0].message.content or ""
        answer, llm_refs = extract_references_from_answer(raw)
        ctx_sources = extract_page_refs_from_context(rag_context)
        return answer, llm_refs, ctx_sources
    except Exception as e:
        logger.error(f"Groq inference failed: {e}")
        raise RuntimeError(f"Groq API error: {e}") from e


# ─── Streaming inference (for st.write_stream) ────────────────────────────────
def stream_groq_inference(
    rag_context: str,
    user_query: str,
    conversation_history: Optional[list] = None,
    api_key_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streaming Groq inference. Yields text delta chunks.
    Use with st.write_stream() in Streamlit.

    Groq streams at ~300 tokens/s — noticeably faster than OpenRouter free tier.

    Usage:
        full_text = st.write_stream(
            stream_groq_inference(raw_ctx, user_prompt, history)
        )
        answer, llm_refs = extract_references_from_answer(full_text)
    """
    client = get_groq_client(api_key_override=api_key_override)
    _model = model_override or groq_config.model
    messages = _build_messages(rag_context, user_query, conversation_history)

    try:
        stream = client.chat.completions.create(
            model=_model,
            messages=messages,
            max_tokens=groq_config.max_tokens,
            temperature=groq_config.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        logger.error(f"Groq streaming failed: {e}")
        yield f"\n\n[Groq streaming error: {e}]"
