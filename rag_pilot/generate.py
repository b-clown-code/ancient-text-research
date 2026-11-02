#!/usr/bin/env python3
"""Grounded generation adapter for ancient literature RAG pipeline.

Formats retrieved passage windows into citation-anchored prompts and verifies
citation groundedness against retrieved context passages.
"""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense import Encoder
from system import search_unified

SYSTEM_INSTRUCTION = """You are a scholarly research assistant specializing in ancient literature across Latin, Greek, Pali, Sanskrit, Syriac, and Coptic traditions.
Answer the user's question accurately using ONLY the provided primary source passages.
Every substantive claim must be explicitly cited with its passage identifier in brackets, e.g. [passage:<id>].
If the provided passages do not contain sufficient evidence to answer the question, state clearly that the corpus does not contain the required evidence."""


def format_grounded_context(search_results):
    """Format unified search results into a clean, citation-anchored context block."""
    context_blocks = []
    seen_passages = set()

    for item_idx, item in enumerate(search_results, 1):
        window = item.get("window", {})
        passages = item.get("passages", [])
        retrieval_meta = item.get("retrieval", {})

        for p in passages:
            pid = p.get("id") or p.get("record", {}).get("passage_id")
            if not pid or pid in seen_passages:
                continue
            seen_passages.add(pid)
            rec = p.get("record", {})
            work_id = rec.get("work_id", "unknown")
            locator = rec.get("locator", f"line:{p.get('line')}")
            lang = rec.get("language", "unknown")
            text = (rec.get("text") or "").strip()

            header = f"[passage:{pid}] (Work: {work_id} | Locator: {locator} | Lang: {lang})"
            context_blocks.append(f"{header}\n{text}\n")

    return "\n---\n".join(context_blocks), list(seen_passages)


def build_rag_prompt(query, search_results):
    """Build complete grounded prompt payload for LLM completion."""
    context_text, passage_ids = format_grounded_context(search_results)
    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXT PASSAGES:
{context_text}

USER QUERY:
{query}

GROUNDED RESPONSE (with [passage:<id>] citations):"""
    return {
        "system_instruction": SYSTEM_INSTRUCTION,
        "query": query,
        "context_passage_count": len(passage_ids),
        "available_passage_ids": passage_ids,
        "full_prompt": prompt
    }


def verify_response_citations(response_text, available_passage_ids):
    """Check that citations in generated text match retrieved context."""
    citation_pattern = re.compile(r'\[passage:([a-zA-Z0-9_\-]+)\]')
    cited_ids = citation_pattern.findall(response_text)
    available_set = set(available_passage_ids)

    valid_citations = [cid for cid in cited_ids if cid in available_set]
    hallucinated_citations = [cid for cid in cited_ids if cid not in available_set]

    return {
        "total_citations": len(cited_ids),
        "unique_citations": len(set(cited_ids)),
        "valid_citations": valid_citations,
        "hallucinated_citations": hallucinated_citations,
        "grounded_ratio": len(valid_citations) / max(1, len(cited_ids)),
        "unsupported_claim_risk": len(hallucinated_citations) > 0
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prompt", help="Retrieve context and format grounded generation prompt")
    pr.add_argument("--db", default="rag_pilot/work/system.sqlite")
    pr.add_argument("--model", default="rag_pilot/work/model")
    pr.add_argument("--revision", default="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
    pr.add_argument("--pooling", choices=["mean", "last_token", "cls"])
    pr.add_argument("--dimension", type=int)
    pr.add_argument("--trust-remote-code", action="store_true",
                     help="Allow custom Python code from the selected model repository")
    pr.add_argument("--cache", default="rag_pilot/work/embeddings.sqlite")
    pr.add_argument("--graph", default="rag_pilot/work/graph.sqlite")
    pr.add_argument("--mode", choices=["lexical", "dense", "hybrid", "hybrid_graph"], default="hybrid")
    pr.add_argument("--limit", type=int, default=5)
    pr.add_argument("--language")
    pr.add_argument("query")

    ver = sub.add_parser("verify", help="Verify citations in an existing response")
    ver.add_argument("--passages", nargs="+", required=True, help="List of available passage IDs")
    ver.add_argument("--response", required=True, help="Generated answer text")

    args = p.parse_args()

    if args.cmd == "prompt":
        db = sqlite3.connect(f"file:{Path(args.db).resolve()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        graph_db = None
        if args.mode == "hybrid_graph" and args.graph:
            graph_db = sqlite3.connect(f"file:{Path(args.graph).resolve()}?mode=ro", uri=True)

        encoder = None
        if args.mode in ("dense", "hybrid", "hybrid_graph"):
            encoder = Encoder(args.model, args.revision, args.cache,
                              pooling=args.pooling, dimension=args.dimension,
                              trust_remote_code=args.trust_remote_code)

        res = search_unified(
            db=db,
            query=args.query,
            mode=args.mode,
            encoder=encoder,
            graph_db=graph_db,
            limit=args.limit,
            language=args.language,
            pooling=args.pooling,
            dimension=args.dimension,
            trust_remote_code=args.trust_remote_code
        )
        prompt_data = build_rag_prompt(args.query, res.get("results", []))
        print(prompt_data["full_prompt"])

        if encoder:
            encoder.close()
        if graph_db:
            graph_db.close()
        db.close()

    elif args.cmd == "verify":
        res = verify_response_citations(args.response, args.passages)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
