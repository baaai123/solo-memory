"""LongMemEval benchmark evaluation for Memory Skill.

Reads LongMemEval test parquet, ingests dialogue history into MemorySkill,
weaves context, and generates answers via DeepSeek API.

Usage:
    venv/bin/python -m scripts.eval_longmem_eval --limit 10
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_data(parquet_path: str, limit: int | None = None) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def ingest_history(skill, documents) -> int:
    """Ingest all dialogue turns from LongMemEval documents."""
    from memory_skill.contracts import DialogueTurn

    count = 0
    for i, turn in enumerate(documents):
        if isinstance(turn, dict):
            text = turn.get("text", "") or turn.get("content", "")
            ts = turn.get("timestamp", None)
        elif isinstance(turn, str):
            text = turn
            ts = None
        else:
            continue
        if not text or not text.strip():
            continue
        from datetime import datetime, UTC
        turn_obj = DialogueTurn(
            id=f"lme_{i}",
            role="user" if i % 2 == 0 else "assistant",
            content=text[:500],
            timestamp=ts or datetime.now(UTC),
        )
        skill.ingest(turn_obj)
        count += 1
    return count


def generate_answer(skill, question: str) -> str:
    """Generate an answer using the memory system + LLM."""
    from memory_skill._llm_utils import call_llm

    ctx = skill.weave(question)
    block = ctx.to_prompt_block()
    prompt = (
        f"{block}\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely based on the above context. If unknown, say 'I don't know.'"
    )

    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-REPLACED-BY-SECURITY-SCAN")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    raw = call_llm(api_base, api_key, model, prompt, max_tokens=200, temperature=0.0)
    return raw.strip() if raw else "I don't know."


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=str, default="/tmp/lme_results.jsonl")
    parser.add_argument("--parquet", type=str,
                       default="/home/pc/projects/memory/test/LongMemEval/test.parquet")
    parser.add_argument("--db", type=str, default="/tmp/lme_eval.db")
    args = parser.parse_args()

    # Setup
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-REPLACED-BY-SECURITY-SCAN")
    os.environ.setdefault("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")
    os.environ.setdefault("IMPORTANCE_API_KEY", os.environ["DEEPSEEK_API_KEY"])
    os.environ.setdefault("IMPORTANCE_API_BASE", os.environ["DEEPSEEK_API_BASE"])
    os.environ.setdefault("IMPORTANCE_MODEL", os.environ["DEEPSEEK_MODEL"])

    # Clean db
    import shutil
    for p in [args.db, args.db + "_chroma"]:
        if os.path.exists(p):
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)

    from memory_skill import MemorySkill, MemorySkillConfig
    skill = MemorySkill(MemorySkillConfig(db_path=args.db, agent_name="lme_eval"))

    data = load_data(args.parquet, limit=args.limit)
    results = []

    for i, row in enumerate(data):
        qid = row.get("question_id", f"q{i}")
        question = row["question"]
        answer = row["answer"]
        qtype = row.get("question_type", "unknown")
        docs = row.get("documents", [])

        # Ingest
        n = ingest_history(skill, docs)
        # Generate
        t0 = time.time()
        hyp = generate_answer(skill, question)
        elapsed = time.time() - t0

        results.append({
            "question_id": qid,
            "question": question,
            "answer": answer,
            "question_type": qtype,
            "response": hyp,
        })
        print(f"[{i+1}/{len(data)}] {qid[:20]} | ingested={n} | {elapsed:.1f}s | {hyp[:60]}")

    # Write results
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults: {args.output} ({len(results)} questions)")


if __name__ == "__main__":
    main()
