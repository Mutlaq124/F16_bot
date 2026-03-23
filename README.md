# F-16 Defence Intelligence Bot

LightRAG-powered knowledge graph chatbot for F-16 technical manuals.
Uses **Ollama** (local) for index building and **Groq LLaMA 70B** (API) for inference.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Neo4j and Groq credentials
```

Required:
- **Ollama** running locally with `llama3.1:8b-instruct-q8_0` and `nomic-embed-text` , General Tip -> ollama is single-threaded, one request at a time - Lightrag launch parallel workers, all in queue waiting for there turn (Timeout configure)
- **Neo4j** instance (local or [Neo4j Aura free](https://neo4j.com/cloud/aura/))
- **Groq API key** from [console.groq.com](https://console.groq.com)

### 3. Build the knowledge index (run once)

```bash
pip install "mineru[all]>=2.0.0" # Instead of all dependencies, can install specific ones as well 
pip install "magic-pdf[all]"
python build_index.py --docs ./Docs --save-chunks
```

Options:
```
--docs ./Docs            Source documents directory (default: ./Docs)
--file-types pdf txt md  File types to process
--reset                  Delete and rebuild index from scratch
--min-chunk-chars 150    Minimum chars to keep a heading section as a chunk
--save-chunks            Save chunks to disk  (debugging)
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Architecture

```
Source Docs (./Docs) -> MinerU/PyMuPDF extraction -> Post-processing -> Heading-based chunks
Chunks -> Ollama LLaMA 3.1 8B -> Entity/Relation extraction -> Neo4j + FAISS index

User Query -> LightRAG (only_need_context=True) -> Raw KG context
Raw context -> Groq LLaMA 3.3 70B -> Answer + Page references + Citations
```

---

## Project Structure

```
app.py                       Streamlit UI (inference only)
build_index.py               Standalone index builder (run once)
extractor.py                 MinerU + PyMuPDF extraction + heading chunking
config.py                    Configuration dataclasses
groq_client.py               Groq inference + reference extraction
prompt_template.py           KG extraction + QA + generator prompts
utils.py                     LightRAG init, context source parsing
Dockerfile                   Container definition
.env.example                 Environment variable template
.streamlit/config.toml       Streamlit theme config
eval/
  eval_dataset.json          20 ground-truth Q&A pairs (F-16 manual)
  eval_script.py             LLM-as-judge evaluation runner
```

---

## Evaluation

Run the evaluation suite against the pre-built index:

```bash
python eval/eval_script.py
python eval/eval_script.py --mode hybrid --top-k 8
python eval/eval_script.py --limit 5       # quick 5-query sanity check
```

### Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? Measures hallucination risk. | > 0.8 |
| **Context Recall** | Does the retrieved context contain what's needed to answer correctly? | > 0.7 |
| **Context Precision** | Is the retrieved context relevant, not noisy? | > 0.6 |
| **Answer Relevance** | Does the answer actually address the question asked? | > 0.75 |
| **Answer Correctness** | How factually accurate is the answer vs. ground truth? | > 0.65 |

### Iteration Guide — What to Do When Scores Are Low

**Low Faithfulness (< 0.6)**
The LLM is adding information not in the retrieved context (hallucinating).
- Tighten the `GENERATOR_PROMPT_TEMPLATE` instruction: "Answer ONLY from the context above."
- Reduce Groq `temperature` to 0.1 in `config.py`.
- Verify the context is actually reaching the LLM (check Raw KG Context in UI).

**Low Context Recall (< 0.5)**
The knowledge graph is missing relevant information for the queries.
- Increase `top_k` slider in app (or `--top-k` in eval) to retrieve more context.
- Switch retrieval mode from `local` to `mix` or `hybrid`.
- Rebuild with smaller `chunk_size` (e.g. 512) — smaller chunks = more entities extracted.
- Increase `entity_extract_max_gleaning` from 1 to 2 for more thorough extraction.
- Check if `DEFENCE_ENTITY_TYPES` covers the query domain.

**Low Context Precision (< 0.5)**
Retrieved context is too noisy — irrelevant chunks are polluting the prompt.
- Decrease `top_k` to retrieve fewer but more targeted results.
- Try `local` mode (entity-centric, less broad).
- Review entity type definitions — overly broad types lead to fuzzy matches.

**Low Answer Relevance (< 0.6)**
The answer talks around the question without directly addressing it.
- Improve `DEFAULT_QA_SYSTEM_PROMPT` — add "Be direct and specific."
- Ensure the `GENERATOR_PROMPT_TEMPLATE` user-turn clearly states the question.
- Check `only_need_context` is working — if context is empty, the LLM has nothing to work with.

**Low Answer Correctness (< 0.5)**
Factual errors in the output. This combines retrieval and generation problems.
- First fix Context Recall (ensure the right information is retrieved).
- Then fix Faithfulness (ensure the LLM uses what was retrieved).
- For procedure queries: ensure the QA prompt instructs numbered-list format.

---

## Tech Stack

| Component | Technology |
|---|---|
| User Interface | Streamlit |
| RAG Framework | LightRAG |
| Graph Database | Neo4j |
| Vector Store | FAISS |
| PDF Extraction | MinerU (magic-pdf) / PyMuPDF fallback |
| Build LLM | Ollama LLaMA 3.1 8B |
| Inference LLM | Groq LLaMA 3.3 70B |
| Embeddings | nomic-embed-text |
| Evaluation | LLM-as-judge (OpenRouter LLM) |


## License
This project is licensed under the terms of the MIT license. 