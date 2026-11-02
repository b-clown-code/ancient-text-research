#!/usr/bin/env python3
"""Evaluate ranked passage IDs against explicit binary judgments."""
import argparse
import json
from collections import defaultdict


def read_rows(path):
    with open(path, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc


import math

def compute_dcg(hits_with_rel, k):
    dcg = 0.0
    for pos, rel in hits_with_rel:
        if pos <= k:
            dcg += (2.0 ** rel - 1.0) / math.log2(pos + 1)
    return dcg

def evaluate(judgments, runs, k=10, k_list=None):
    if k_list is None:
        k_list = [k] if isinstance(k, int) else list(k)
    for kv in k_list:
        if kv < 1:
            raise ValueError('k must be positive')
    max_k = max(k_list)

    queries = {}
    for row in judgments:
        q = row['query_id']
        if q in queries:
            raise ValueError(f'duplicate judgment query: {q}')
        relevant = row['relevant_ids']
        if not isinstance(relevant, list) or len(set(relevant)) != len(relevant):
            raise ValueError(f'{q}: relevant_ids must be a unique list')
        if not isinstance(row['answerable'], bool) or row['answerable'] != bool(relevant):
            raise ValueError(f'{q}: answerable must match whether relevant_ids is nonempty')
        queries[q] = row
    ranked = {}
    for row in runs:
        q = row['query_id']
        if q not in queries or q in ranked:
            raise ValueError(f'unknown or duplicate run query: {q}')
        ids = row['passage_ids']
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise ValueError(f'{q}: passage_ids must be a unique list')
        ranked[q] = ids
    missing = queries.keys() - ranked.keys()
    if missing:
        raise ValueError(f'missing run queries: {sorted(missing)}')
    
    groups = defaultdict(lambda: defaultdict(list))
    for q, row in queries.items():
        relevant = set(row['relevant_ids'])
        rel_grades = {}
        for gt in row.get('ground_truth_details', []):
            if isinstance(gt, dict) and 'passage_id' in gt:
                rel_grades[gt['passage_id']] = gt.get('grade', 1)
        for pid in relevant:
            if pid not in rel_grades:
                rel_grades[pid] = 1

        ids = ranked[q][:max_k]
        lang_group = 'language:' + row.get('language', 'unknown')

        for kv in k_list:
            sub_ids = ids[:kv]
            if relevant:
                hits = [i for i, pid in enumerate(sub_ids, 1) if pid in relevant]
                recall = len(hits) / len(relevant)
                mrr = 1.0 / hits[0] if hits else 0.0

                # Compute nDCG@k
                hits_with_rel = [(i, rel_grades.get(pid, 1)) for i, pid in enumerate(sub_ids, 1) if pid in relevant]
                dcg = compute_dcg(hits_with_rel, kv)
                ideal_rels = sorted(rel_grades.values(), reverse=True)[:kv]
                idcg = sum((2.0 ** r - 1.0) / math.log2(idx + 2) for idx, r in enumerate(ideal_rels))
                ndcg = (dcg / idcg) if idcg > 0 else 0.0

                item = (recall, mrr, ndcg, None)
            else:
                item = (None, None, None, bool(sub_ids))

            groups['overall'][kv].append(item)
            groups[lang_group][kv].append(item)

    result = {}
    all_group_names = sorted(groups.keys())
    for name in all_group_names:
        result[name] = {}
        for kv in k_list:
            items = groups[name][kv]
            answerable = [i for i in items if i[0] is not None]
            unanswerable = [i for i in items if i[3] is not None]
            if f'answerable_queries' not in result[name]:
                result[name]['answerable_queries'] = len(answerable)
                result[name]['unanswerable_queries'] = len(unanswerable)
            result[name][f'recall@{kv}'] = sum(i[0] for i in answerable) / len(answerable) if answerable else None
            result[name][f'mrr@{kv}'] = sum(i[1] for i in answerable) / len(answerable) if answerable else None
            result[name][f'ndcg@{kv}'] = sum(i[2] for i in answerable) / len(answerable) if answerable else None
            result[name][f'unanswerable_nonempty_rate@{kv}'] = sum(i[3] for i in unanswerable) / len(unanswerable) if unanswerable else None

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('judgments')
    parser.add_argument('run')
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(read_rows(args.judgments), read_rows(args.run), args.k), indent=2, ensure_ascii=False))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
