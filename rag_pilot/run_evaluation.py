#!/usr/bin/env python3
"""Execute full evaluation benchmark across all retrieval modes and record metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense import Encoder
from rerank import Reranker
from system import VectorMatrixCache, search_unified
from evaluate import read_rows, evaluate


def run_benchmark(args):
    judgments_path = Path(args.judgments)
    if not judgments_path.exists():
        raise FileNotFoundError(f"Judgments file not found: {judgments_path}")

    all_judgments = list(read_rows(judgments_path))
    if args.languages:
        target_langs = set(args.languages)
        judgments = [j for j in all_judgments if j.get("language") in target_langs]
        print(f"Filtered to {len(judgments)} benchmark queries for languages: {sorted(target_langs)} (from {len(all_judgments)})")
    else:
        judgments = all_judgments
        print(f"Loaded {len(judgments)} benchmark queries from {judgments_path}")

    if not judgments:
        raise ValueError("No benchmark queries match the selected filters")

    db_path = Path(args.db).resolve()
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    graph_path = Path(args.graph).resolve()
    graph_db = sqlite3.connect(f"file:{graph_path}?mode=ro", uri=True)

    print(f"Initializing neural encoder ({args.model})...")
    t0 = time.time()
    encoder = Encoder(
        model_path=args.model,
        revision=args.revision,
        cache_path=args.cache,
        batch_size=args.batch,
        pooling=args.pooling,
        dimension=args.dimension,
        default_query_instruction=args.instruction,
        trust_remote_code=args.trust_remote_code
    )
    print(f"Encoder initialized in {time.time()-t0:.2f}s (dim: {encoder.dimension}, pooling: {encoder.pooling_method}, weight file: {encoder.weight_file})")

    # Initialize reranker if requested
    reranker = None
    any_rerank = any("rerank" in m for m in args.modes)
    if any_rerank:
        print(f"Initializing reranker ({args.reranker or 'rule-based fallback'})...")
        t0 = time.time()
        reranker = Reranker(model_path=args.reranker, revision=args.reranker_revision,
                            trust_remote_code=args.reranker_trust_remote_code)
        print(f"Reranker initialized in {time.time()-t0:.2f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine max retrieval limit for k evaluation
    k_eval_list = [10, 20] if args.limit >= 20 else [10]
    eval_limit = max(args.limit, max(k_eval_list))

    results = {}
    vector_cache = VectorMatrixCache()
    for mode in args.modes:
        print(f"\n=======================================================")
        print(f"Running evaluation for mode: {mode}")
        print(f"=======================================================")
        t_mode_start = time.time()
        run_records = []
        latencies = []
        timings_breakdowns = []

        is_graph_mode = mode.endswith("_graph")
        current_graph_db = graph_db if is_graph_mode else None

        for idx, j in enumerate(judgments):
            t_q_start = time.perf_counter()
            qid = j["query_id"]
            lang = j.get("language")
            query = j["query"]

            res = search_unified(
                db=db,
                query=query,
                mode=mode,
                encoder=encoder,
                graph_db=current_graph_db,
                graph_seeds=args.graph_seeds,
                graph_limit=args.graph_limit,
                graph_weight=getattr(args, "graph_weight", 0.1),
                min_dense_sim=getattr(args, "min_dense_sim", None),
                limit=eval_limit,
                candidates=args.candidates,
                rrf=args.rrf,
                language=lang,
                reranker=reranker if "rerank" in mode else None,
                rerank_candidates=args.rerank_candidates,
                task=args.task,
                instruction=args.instruction,
                pooling=args.pooling,
                dimension=args.dimension,
                trust_remote_code=args.trust_remote_code,
                vector_cache=vector_cache
            )
            lat_ms = (time.perf_counter() - t_q_start) * 1000.0
            latencies.append(lat_ms)
            timings_breakdowns.append(res.get("timings_ms", {}))

            retrieved_passage_ids = []
            seen_pids = set()
            for r_item in res["results"]:
                for p in r_item.get("passages", []):
                    pid = p.get("id") or p.get("record", {}).get("passage_id")
                    if pid and pid not in seen_pids:
                        seen_pids.add(pid)
                        retrieved_passage_ids.append(pid)
                        if len(retrieved_passage_ids) >= eval_limit:
                            break
                if len(retrieved_passage_ids) >= eval_limit:
                    break

            run_records.append({
                "query_id": qid,
                "passage_ids": retrieved_passage_ids
            })

        t_mode_elapsed = time.time() - t_mode_start
        run_file = out_dir / f"run_{mode}.jsonl"
        with open(run_file, "w", encoding="utf-8") as f:
            for r in run_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Evaluate run across k_eval_list (e.g. k=10 and k=20)
        eval_metrics = evaluate(judgments, run_records, k=args.k, k_list=k_eval_list)
        eval_metrics["runtime"] = {
            "total_seconds": round(t_mode_elapsed, 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        }
        results[mode] = eval_metrics

        rec10 = eval_metrics["overall"].get("recall@10")
        rec20 = eval_metrics["overall"].get("recall@20")
        mrr10 = eval_metrics["overall"].get("mrr@10")
        ndcg10 = eval_metrics["overall"].get("ndcg@10")
        unans10 = eval_metrics["overall"].get("unanswerable_nonempty_rate@10")

        print(f"Completed {mode} in {t_mode_elapsed:.2f}s (mean: {eval_metrics['runtime']['mean_latency_ms']} ms/query)")
        print(f"Overall Recall@10: {rec10:.4f}" + (f" | Recall@20: {rec20:.4f}" if rec20 is not None else "") + f" | MRR@10: {mrr10:.4f}" + (f" | nDCG@10: {ndcg10:.4f}" if ndcg10 is not None else ""))
        print(f"Unanswerable Non-empty Rate@10: {unans10:.4f}")

    # Save summary report
    report_file = out_dir / "evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull evaluation report saved to {report_file}")

    vector_cache.clear()
    db.close()
    graph_db.close()
    encoder.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", default="rag_pilot/work/benchmark_v2.jsonl")
    parser.add_argument("--db", default="rag_pilot/work/system.sqlite")
    parser.add_argument("--graph", default="rag_pilot/work/graph.sqlite")
    parser.add_argument("--model", default="rag_pilot/work/model")
    parser.add_argument("--revision", default="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
    parser.add_argument("--cache", default="rag_pilot/work/embeddings.sqlite")
    parser.add_argument("--output-dir", default="rag_pilot/work")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["lexical", "lexical_graph", "dense", "dense_graph", "hybrid", "hybrid_graph", "hybrid_rerank", "hybrid_rerank_graph"],
        help="List of retrieval modes to evaluate"
    )
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--rrf", type=int, default=60)
    parser.add_argument("--graph-seeds", type=int, default=3)
    parser.add_argument("--graph-limit", type=int, default=40)
    parser.add_argument("--graph-weight", type=float, default=0.1)
    parser.add_argument("--min-dense-sim", type=float, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pooling", default=None, choices=["mean", "last_token", "cls", None])
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow custom Python code from the selected encoder model")
    parser.add_argument("--task", default="modern_query_to_primary_evidence")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--reranker", default=None, help="Path to neural cross-encoder model")
    parser.add_argument("--reranker-revision", default=None)
    parser.add_argument("--reranker-trust-remote-code", action="store_true",
                        help="Allow custom Python code from the selected reranker model")
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--languages", nargs="*", default=None, help="Optional language filter list")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
