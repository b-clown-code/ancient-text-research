"""Train and evaluate the three Gate 1 byte-level BPE candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tokenizers

from .data import EXCLUDED, SOURCES, iter_records, language_code

SEED = 20261101
LANGUAGE_WORD_CAP = 500_000
VOCABS = (16_384, 32_768, 49_152)
EOS = "<|eos|>"
SAMPLE_FILE_LIMITS = {"aed": 5000, "bilara": 1500, "greek": 800,
                      "gretil": 500, "latin-170m": 800}


def script_group(text: str, representation: str = "") -> str:
    counts = Counter()
    for c in text:
        if c.isspace() or not c.isalpha():
            continue
        name = unicodedata.name(c, "UNKNOWN")
        for script in ("GREEK", "SYRIAC", "COPTIC", "ARABIC", "DEVANAGARI", "TIBETAN", "LATIN", "EGYPTIAN"):
            if script in name:
                counts[script] += 1
                break
        else:
            counts["OTHER"] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return representation or "UNKNOWN"


def collect_sample(cap: int = LANGUAGE_WORD_CAP):
    fit, heldout, words = [], [], Counter()
    strata = Counter()
    # Independent per-language budgets prevent Latin's larger reservoir from
    # setting the tokenizer vocabulary. Inputs remain bounded by the cap.
    rng = random.Random(SEED)
    for source in SOURCES:
        for record in iter_records("train", sources=[source],
                                   file_limit=SAMPLE_FILE_LIMITS.get(source, 500)):
            lang = record["_language_code"]
            text = record["text"]
            n_words = max(1, len(text.split()))
            if words[lang] + n_words > cap:
                continue
            words[lang] += n_words
            rep = str(record.get("representation", ""))
            script = script_group(text, rep)
            bucket = (lang, script)
            strata[bucket] += 1
            key = str(record.get("record_id") or record.get("passage_id") or text[:120])
            hold = int(hashlib.sha256((str(SEED) + key).encode()).hexdigest()[:8], 16) % 10 == 0
            (heldout if hold else fit).append((lang, script, text))
    return fit, heldout, dict(words), {f"{l}:{s}": n for (l, s), n in strata.items()}


def build_tokenizer(texts, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS], show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.add_special_tokens([EOS])
    return tokenizer


def evaluate(tokenizer: Tokenizer, rows):
    agg = defaultdict(lambda: {"tokens": 0, "chars": 0, "bytes": 0, "words": 0})
    usage = defaultdict(Counter)
    token_groups = defaultdict(set)
    examples = {}
    for lang, script, text in rows:
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        if decoded != text:
            raise ValueError(f"Unicode round-trip failed for {lang}/{script}")
        k = f"{lang}:{script}"
        a = agg[k]
        a["tokens"] += len(enc.ids)
        a["chars"] += len(text)
        a["bytes"] += len(text.encode("utf-8"))
        a["words"] += len(text.split())
        usage[k].update(enc.ids)
        for token_id in enc.ids:
            token_groups[token_id].add(k)
        if k not in examples:
            example_ids = tokenizer.encode(text[:180]).ids
            examples[k] = {"text": text[:180],
                           "tokens": [tokenizer.decode([token_id]) for token_id in example_ids[:80]]}
    metrics = {}
    by_lang = defaultdict(lambda: Counter())
    for k, a in agg.items():
        metrics[k] = {**a, "tokens_per_char": a["tokens"] / max(1, a["chars"]),
                      "tokens_per_byte": a["tokens"] / max(1, a["bytes"]),
                      "tokens_per_word": a["tokens"] / max(1, a["words"])}
        lang = k.split(":", 1)[0]
        by_lang[lang].update(a)
    lang_metrics = {}
    for lang, a in by_lang.items():
        lang_metrics[lang] = {"tokens": a["tokens"], "chars": a["chars"], "bytes": a["bytes"],
                              "words": a["words"],
                              "tokens_per_char": a["tokens"] / max(1, a["chars"]),
                              "tokens_per_byte": a["tokens"] / max(1, a["bytes"]),
                              "tokens_per_word": a["tokens"] / max(1, a["words"])}
    balanced = sum(x["tokens_per_char"] for x in lang_metrics.values()) / max(1, len(lang_metrics))
    vocab_allocation = {}
    for k, freq in usage.items():
        vocab_allocation[k] = {
            "distinct_tokens_used": len(freq),
            "tokens_used_exclusively_here": sum(1 for token_id in freq if len(token_groups[token_id]) == 1),
            "tokens_shared_with_other_groups": sum(1 for token_id in freq if len(token_groups[token_id]) > 1),
            "token_occurrences": sum(freq.values()),
        }
    return {"by_language_script": metrics, "by_language": lang_metrics,
            "vocabulary_use_by_language_script": vocab_allocation,
            "language_balanced_tokens_per_char": balanced, "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cap-words", type=int, default=LANGUAGE_WORD_CAP)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fit, heldout, counts, strata = collect_sample(args.cap_words)
    if not fit or not heldout:
        raise SystemExit("No fit/evaluation sample; verify permitted train shards")
    candidates = {}
    summary = {"seed": SEED, "source_allowlist": list(SOURCES), "excluded": EXCLUDED,
               "sample_words_by_language": counts, "fit_records": len(fit),
               "tokenizer_eval_records": len(heldout), "script_strata_records": strata,
               "candidates": {}}
    best_tpc = None
    for vocab in VOCABS:
        tok = build_tokenizer((r[2] for r in fit), vocab)
        path = args.output / f"ancient-byte-bpe-{vocab}.json"
        tok.save(str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = evaluate(tok, heldout)
        size = tok.get_vocab_size(with_added_tokens=True)
        summary["candidates"][str(vocab)] = {
            "artifact": path.name, "sha256": digest, "vocab_size": size,
            "embedding_parameters_dim_576": size * 576,
            "embedding_bf16_bytes": size * 576 * 2,
            "metrics": metrics,
        }
        best_tpc = min(best_tpc or metrics["language_balanced_tokens_per_char"],
                       metrics["language_balanced_tokens_per_char"])
        candidates[vocab] = (path, tok, metrics)
    eligible = []
    group_best = {}
    language_best = {}
    for _, _, metrics in candidates.values():
        for group, m in metrics["by_language_script"].items():
            group_best[group] = min(group_best.get(group, m["tokens_per_char"]), m["tokens_per_char"])
        for lang, m in metrics["by_language"].items():
            language_best[lang] = min(language_best.get(lang, m["tokens_per_char"]), m["tokens_per_char"])
    for vocab, (_, _, metrics) in candidates.items():
        if metrics["language_balanced_tokens_per_char"] > best_tpc * 1.05:
            continue
        language_regression = any(
            m["words"] >= 5_000 and m["tokens_per_char"] > language_best[lang] * 1.10
            for lang, m in metrics["by_language"].items()
        )
        script_regression = any(
            m["words"] >= 5_000 and m["tokens_per_char"] > group_best[group] * 1.10
            for group, m in metrics["by_language_script"].items()
        )
        if language_regression or script_regression:
            continue
        eligible.append(vocab)
    selected = min(eligible) if eligible else min(candidates, key=lambda v: candidates[v][2]["language_balanced_tokens_per_char"])
    selected_path = candidates[selected][0]
    default_path = args.output / "ancient-byte-bpe-default.json"
    shutil.copyfile(selected_path, default_path)
    summary["selected_vocab"] = selected
    summary["selection_rule"] = "smallest candidate within 5% of best language-balanced tokens/char and within 10% of best for language/script groups with at least 5,000 tokenizer-eval words"
    summary["selected_sha256"] = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    manifest_hashes = {}
    for source, directory in SOURCES.items():
        manifest = directory / "EXPORT-MANIFEST.json"
        if manifest.exists():
            manifest_hashes[source] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = {
        "library": "tokenizers", "library_version": tokenizers.__version__,
        "algorithm": "byte-level BPE", "vocab_candidates": list(VOCABS),
        "normalizer": None, "byte_alphabet": "all 256 ByteLevel alphabet symbols",
        "pre_tokenizer": "ByteLevel(add_prefix_space=false)", "decoder": "ByteLevel",
        "special_tokens": [EOS], "min_frequency": 2,
        "seed": SEED, "language_word_cap": args.cap_words,
        "sample_file_limits": SAMPLE_FILE_LIMITS,
        "input_manifest_sha256": manifest_hashes,
        "source_allowlist": list(SOURCES), "excluded_scopes": EXCLUDED,
        "selected_vocab": selected, "selected_artifact": default_path.name,
        "selected_sha256": hashlib.sha256(default_path.read_bytes()).hexdigest(),
        "selection_rule": summary["selection_rule"],
    }
    (args.output / "tokenizer-config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary["configuration"] = config
    summary["sampling_estimates"] = estimate_sampling_tokens(summary["candidates"][str(selected)]["metrics"]["by_language"])
    (args.output / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected_vocab": selected, "fit_records": len(fit),
                      "evaluation_records": len(heldout), "languages": sorted(counts)}, indent=2))


def estimate_sampling_tokens(metrics_by_lang):
    """Project tokens from existing approved word counts without reading all shards."""
    totals = Counter()
    for source, directory in SOURCES.items():
        if source == "seed-syriac":
            continue
        p = directory / "SCORECARD.json"
        if not p.exists():
            continue
        score = json.loads(p.read_text())
        for outcome in score.get("source_outcomes", []):
            lang = language_code(outcome.get("decision", {}).get("language"))
            if lang not in metrics_by_lang:
                continue
            for row in outcome.get("counts", []):
                if row.get("metric") == "words_ws" and row.get("split") == "train" and row.get("eligibility_status") == "approved":
                    totals[lang] += int(row["value"])
    # The formal Syriac seed is separate from the incremental source counts.
    totals["syr"] += 51023
    estimates = {}
    for lang, words in totals.items():
        tpw = metrics_by_lang[lang]["tokens_per_word"]
        estimates[lang] = {"approved_train_words_by_manifest": words,
                           "projected_tokens_corpus_proportional": round(words * tpw),
                           "tokens_per_million_words_equal_language": round(tpw * 1_000_000)}
    estimates["method_note"] = "Counts use approved SCORECARD train words grouped by the recorded source language, plus the formal seed once. Source counts are representations, not deduplicated unique-corpus counts."
    estimates["corpus_proportional_total_tokens"] = sum(v["projected_tokens_corpus_proportional"] for v in estimates.values() if isinstance(v, dict))
    estimates["equal_language_tokens_per_million_word_per_language"] = {
        k: v["tokens_per_million_words_equal_language"] for k, v in estimates.items() if isinstance(v, dict)
    }
    return estimates


if __name__ == "__main__":
    main()
