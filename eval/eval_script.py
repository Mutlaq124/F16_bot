"""
Metrics:
    faithfulness        -- Is the answer grounded in the retrieved context?
    context_recall      -- Does the context contain the expected answer's information?
    context_precision   -- Is the retrieved context relevant (not noisy)?
    answer_relevance    -- Does the answer address the user's question?
    answer_correctness  -- How factually close is the answer to the expected answer?

Usage:
    # From project root:
    python eval/eval_script.py
    python eval/eval_script.py --mode hybrid --top-k 8
    python eval/eval_script.py --limit 5          # quick sanity check with first 5 items
    python eval/eval_script.py --output eval/my_run.json
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from tqdm import tqdm
from tabulate import tabulate

from config import groq_config, rag_config, ollama_config
from utils import initialize_lightrag
from lightrag import QueryParam
from prompt_template import get_qa_system_prompt, get_generator_prompt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def get_groq_client() -> OpenAI:
    return OpenAI(api_key=groq_config.api_key, base_url=groq_config.base_url)


def get_openrouter_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENROUTER_API_KEY", ""), base_url="https://openrouter.ai/api/v1")


def _judge(client: OpenAI, prompt: str) -> float:
    """Call OpenRouter as an LLM judge. Returns a float score 0.0-1.0."""
    time.sleep(0.5) # Since it's paid API, we don't need intense Rlimit sleeps
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, objective evaluator. "
                        "Respond with ONLY a single decimal number between 0.0 and 1.0. "
                        "No explanation, no text — only the score."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        raw_content = resp.choices[0].message.content
        if raw_content is None:
            logger.warning("Judge returned None content (possible safety filter or API block).")
            return 0.0
        
        raw = raw_content.strip()
        score = float(raw)
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return 0.0


def score_faithfulness(client: OpenAI, context: str, answer: str) -> float:
    prompt = f"""Retrieved Context:
{context}

Generated Answer:
{answer}

Score 0.0 to 1.0: How much of the generated answer is directly supported by or derivable from the retrieved context?
1.0 = every claim in the answer is grounded in the context.
0.0 = the answer contradicts or ignores the context entirely."""
    return _judge(client, prompt)


def score_context_recall(client: OpenAI, expected: str, context: str) -> float:
    prompt = f"""Expected Answer (ground truth):
{expected}

Retrieved Context:
{context}

Score 0.0 to 1.0: How much of the key information from the expected answer is present in the retrieved context?
1.0 = all critical facts from the expected answer are in the context.
0.0 = none of the relevant information is present."""
    return _judge(client, prompt)


def score_context_precision(client: OpenAI, query: str, context: str) -> float:
    prompt = f"""User Query:
{query}

Retrieved Context:
{context}

Score 0.0 to 1.0: How relevant and focused is the retrieved context for answering this specific query?
1.0 = context is highly targeted with minimal irrelevant noise.
0.0 = context is entirely off-topic or retrieves wrong information."""
    return _judge(client, prompt)


def score_answer_relevance(client: OpenAI, query: str, answer: str) -> float:
    prompt = f"""User Query:
{query}

Generated Answer:
{answer}

Score 0.0 to 1.0: How directly and completely does the answer address what the user asked?
1.0 = answer fully addresses the question with no irrelevant content.
0.0 = answer misses the question entirely or is evasive."""
    return _judge(client, prompt)


def score_answer_correctness(client: OpenAI, expected: str, answer: str) -> float:
    prompt = f"""Expected Answer (ground truth):
{expected}

Generated Answer:
{answer}

Score 0.0 to 1.0: How factually accurate is the generated answer compared to the ground truth?
1.0 = all key facts match the expected answer exactly.
0.0 = generated answer contains factual errors or omits all critical information."""
    return _judge(client, prompt)


def generate_answer(client: OpenAI, context: str, query: str, history: list = None, model="meta-llama/llama-3.3-70b-instruct") -> str:
    time.sleep(2) 
    
    # Safe fallback if retrieval failed and returned None
    context = context or ""
    
    # No context truncation needed for paid OpenRouter
    truncated_context = context
    
    messages = [{"role": "system", "content": get_qa_system_prompt()}]
    if history:
        for turn in history[-4:]:
            if turn.get("role") in ("user", "assistant"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": get_generator_prompt(context=truncated_context, query=query)})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=groq_config.max_tokens,
            temperature=groq_config.temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[Generation error: {e}]"


async def evaluate_dataset(
    mode: str,
    top_k: int,
    # max_graph_nodes: int,
    limit: int,
    output_path: Path,
):
    print(f"\nInitializing LightRAG...")
    rag = await initialize_lightrag()
    if not rag:
        print("ERROR: LightRAG initialization failed. Is Neo4j running and index built?")
        sys.exit(1)

    generate_client = get_openrouter_client()
    judge_client = get_openrouter_client()

    dataset = json.loads(DATASET_PATH.read_text())
    if limit:
        dataset = dataset[:limit]

    print(f"Running evaluation: {len(dataset)} queries | mode={mode} | top_k={top_k}")
    print("-" * 70)

    results = []
    metric_totals = {
        "faithfulness": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "answer_relevance": 0.0,
        "answer_correctness": 0.0,
    }
    category_scores = {}

    for item in tqdm(dataset, desc="Evaluating"):
        query = item["query"]
        expected = item["expected_answer"]
        category = item.get("category", "General")

        try:
            time.sleep(2) 

            context = await rag.aquery(
                query,
                param=QueryParam(
                    mode=mode,
                    top_k=top_k, # top-k (vector store)
                    chunk_top_k=top_k, # top-k (graph entities/relations)
                    enable_rerank=False, #enable for better results
                    only_need_context=True, # must be true for production 
                )
            )
        except Exception as e:
            context = ""
            logger.warning(f"Retrieval failed for '{query[:40]}...': {e}")
            
        context = context or ""

        answer = generate_answer(generate_client, context, query)

        scores = {
            "faithfulness":       score_faithfulness(judge_client, context, answer),
            "context_recall":     score_context_recall(judge_client, expected, context),
            "context_precision":  score_context_precision(judge_client, query, context),
            "answer_relevance":   score_answer_relevance(judge_client, query, answer),
            "answer_correctness": score_answer_correctness(judge_client, expected, answer),
        }

        avg = sum(scores.values()) / len(scores)

        result_entry = {
            "query": query,
            "category": category,
            "expected_answer": expected,
            "generated_answer": answer,
            "retrieved_context_length": len(context),
            "scores": scores,
            "average_score": round(avg, 3),
        }
        results.append(result_entry)

        for k, v in scores.items():
            metric_totals[k] += v

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(avg)

    n = len(results)
    summary = {
        "run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "top_k": top_k,
            "total_queries": n,
            "generator_model": "meta-llama/llama-3.3-70b-instruct",
            "judge_model": "openai/gpt-4o-mini",
        },
        "overall_metrics": {k: round(v / n, 4) for k, v in metric_totals.items()},
        "overall_average": round(sum(metric_totals.values()) / (len(metric_totals) * n), 4),
        "category_averages": {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in category_scores.items()
        },
        "per_query_results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    overall_table = [
        ["Faithfulness",       f"{summary['overall_metrics']['faithfulness']:.3f}",
         "Answer grounded in context?"],
        ["Context Recall",     f"{summary['overall_metrics']['context_recall']:.3f}",
         "Context covers expected answer?"],
        ["Context Precision",  f"{summary['overall_metrics']['context_precision']:.3f}",
         "Context is relevant, not noisy?"],
        ["Answer Relevance",   f"{summary['overall_metrics']['answer_relevance']:.3f}",
         "Answer addresses the question?"],
        ["Answer Correctness", f"{summary['overall_metrics']['answer_correctness']:.3f}",
         "Answer matches ground truth?"],
        ["──────────────", "───────", "──────────────────────────────────"],
        ["Overall Average",    f"{summary['overall_average']:.3f}", ""],
    ]
    print(tabulate(overall_table, headers=["Metric", "Score", "Description"], tablefmt="rounded_outline"))

    print("\nBy Category:")
    cat_table = [
        [cat, f"{score:.3f}", f"{sum(1 for r in results if r['category']==cat)} queries"]
        for cat, score in sorted(summary["category_averages"].items(), key=lambda x: x[1], reverse=True)
    ]
    print(tabulate(cat_table, headers=["Category", "Avg Score", "Queries"], tablefmt="rounded_outline"))

    print(f"\nDetailed results saved to: {output_path}")
    print("=" * 70)

    print("\nDiagnosis & Iteration Guide:")
    met = summary["overall_metrics"]
    if met["context_recall"] < 0.5:
        print("  LOW Context Recall -- KG may be missing relevant edges. Try: increase top_k, use 'mix' mode, or rebuild with smaller chunk_size.")
    if met["context_precision"] < 0.5:
        print("  LOW Context Precision -- Too much noisy context. Try: reduce top_k, use 'local' mode, or improve entity type specificity in KG_EXTRACTION_PROMPT.")
    if met["faithfulness"] < 0.6:
        print("  LOW Faithfulness -- LLM hallucinating beyond context. Check GENERATOR_PROMPT_TEMPLATE instructions on grounding.")
    if met["answer_relevance"] < 0.6:
        print("  LOW Answer Relevance -- Answer is off-topic. Check DEFAULT_QA_SYSTEM_PROMPT and GENERATOR_PROMPT_TEMPLATE.")
    if met["answer_correctness"] < 0.5:
        print("  LOW Answer Correctness -- Factual gaps. Either retrieval is poor (raise top_k) or extraction missed entities (check gleaning).")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate F-16 Bot RAG quality")
    parser.add_argument("--mode", default="hybrid", choices=["mix", "hybrid", "local", "global", "naive"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=20, help="Limit to first N items (default=5 for free tier APIs)")
    parser.add_argument("--output", type=str,
                        default=f"eval/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(evaluate_dataset(
        mode=args.mode,
        top_k=args.top_k,
        limit=args.limit,

        output_path=Path(args.output),
    ))
