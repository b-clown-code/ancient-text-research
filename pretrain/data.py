"""Streaming reader and fixed-length packer for approved Gate 1 exports."""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

EXPORT_ROOT = Path("catalog/curation/parallel-v1/bulk-corpus-v2/research-turnaround-v2/bulk-corpus-v2")
SOURCES = {
    "seed-syriac": Path("catalog/releases/first-corpus-v1"),
    "greek": EXPORT_ROOT / "greek-export-v1",
    "gretil": EXPORT_ROOT / "gretil-export-v1",
    "aed": EXPORT_ROOT / "aed-export-v1",
    "latin-170m": EXPORT_ROOT / "latin-export-v1",
    "openmgh": EXPORT_ROOT / "openmgh-export-v1",
    "perseus-latin": EXPORT_ROOT / "perseus-latin-export-v1",
    "bilara": EXPORT_ROOT / "bilara-export-v1",
    "coptic": EXPORT_ROOT / "coptic-export-v1",
    "syriac": EXPORT_ROOT / "syriac-export-v1",
    "tanzil": EXPORT_ROOT / "tanzil-export-v1",
}
SOURCE_LANGUAGES = {
    "seed-syriac": {"syr"}, "greek": {"grc"}, "gretil": {"san", "bod", "tam"},
    "aed": {"egy"}, "latin-170m": {"lat"}, "openmgh": {"lat"},
    "perseus-latin": {"lat"}, "bilara": {"pli"}, "coptic": {"cop"},
    "syriac": {"syr"}, "tanzil": {"ara"},
}
EXCLUDED = {
    "ddbdp": "current register: group review pending; training_ready is false",
}


def sources_for_language(language: str) -> list[str]:
    return [source for source, languages in SOURCE_LANGUAGES.items() if language in languages]


def language_code(value: object) -> str:
    value = str(value or "und").strip().lower()
    aliases = {
        "sahidic coptic": "cop", "bohairic coptic": "cop",
        "sahidic coptic, with lycopolitan features": "cop",
        "syc": "syr", "syriac": "syr", "classical syriac": "syr",
        "greek": "grc", "latin": "lat", "sanskrit": "san", "pali": "pli",
        "arabic": "ara",
        "egyp": "egy", "egyh": "egy", "egyh-egyhol": "egy", "egy": "egy",
    }
    if value in aliases:
        return aliases[value]
    if value.startswith(("egyp", "egyh")):
        return "egy"
    return value.split("-", 1)[0]


def _files(source: str, split: str, file_limit: int | None = None) -> list[Path]:
    if source not in SOURCES:
        raise ValueError(f"Source {source!r} is not in the Gate 1 allowlist")
    root = SOURCES[source]
    if source == "seed-syriac":
        path = root / ("validation.jsonl.gz" if split == "validation" else f"{split}.jsonl.gz")
        return [path] if path.exists() and split in {"train", "validation"} else []
    files = sorted(root.rglob(f"*-{split}.jsonl.gz"))
    if file_limit is not None and len(files) > file_limit:
        import hashlib
        selected = set(sorted(files, key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())[:file_limit])
        # Keep deterministic sampling while reading in path order to reduce
        # random seeks on the SATA-backed export tree.
        files = [p for p in files if p in selected]
    return files


def iter_records(split: str = "train", sources: list[str] | None = None,
                 languages: set[str] | None = None,
                 file_limit: int | None = None) -> Iterator[dict]:
    """Stream eligible records; test/quarantine files are never selected."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    sources = sources or list(SOURCES)
    for source in sources:
        for path in _files(source, split, file_limit):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("split", split) != split:
                        continue
                    # The formal release predates the per-row training_ready field.
                    if source != "seed-syriac" and record.get("training_ready") is not True:
                        continue
                    text = record.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    lang = language_code(record.get("language"))
                    if languages and lang not in languages:
                        continue
                    yield {**record, "_source": source, "_language_code": lang}


def _records_for_spec(split: str, spec: str,
                      source_override: list[str] | None = None) -> Iterator[dict]:
    parts = spec.split("/")
    if len(parts) == 2 and parts[0] == "language":
        lang = parts[1]
        selected_sources = source_override or sources_for_language(lang)
        if not selected_sources:
            raise ValueError(f"No allowlisted sources for language {lang!r}")
        return iter_records(split, selected_sources, {lang})
    if len(parts) == 2 and parts[0] == "source" and parts[1] in SOURCES:
        return iter_records(split, [parts[1]])
    if (len(parts) == 4 and parts[0] == "source" and parts[1] in SOURCES
            and parts[2] == "language"):
        return iter_records(split, [parts[1]], {parts[3]})
    raise ValueError(f"Invalid weight key {spec!r}")


class WeightedRecordPool:
    """Stateful deterministic interleaver whose weights may change by stage."""

    def __init__(self, split: str, specs, seed: int = 20261101,
                 source_override: list[str] | None = None):
        self.streams = {spec: _records_for_spec(split, spec, source_override) for spec in specs}
        self.rng = random.Random(seed)
        self.exhausted = set()
        self.records_by_source_language = defaultdict(int)

    def next_record(self, weights: dict[str, float]) -> dict:
        if not weights or any(value < 0 for value in weights.values()) or not any(weights.values()):
            raise ValueError("weights must contain at least one positive value")
        if not set(weights).issubset(self.streams):
            raise ValueError("weight key was not included when this record pool was created")
        while True:
            choices = [key for key, value in weights.items()
                       if value > 0 and key not in self.exhausted]
            if not choices:
                raise StopIteration
            key = self.rng.choices(choices, weights=[weights[x] for x in choices], k=1)[0]
            try:
                record = next(self.streams[key])
            except StopIteration:
                self.exhausted.add(key)
                continue
            self.records_by_source_language[(record["_source"], record["_language_code"])] += 1
            return record


def scheduled_records(pool: WeightedRecordPool, weights_getter) -> Iterator[dict]:
    while True:
        yield pool.next_record(weights_getter())


def weighted_records(split: str, weights: dict[str, float], seed: int = 20261101,
                     sources: list[str] | None = None) -> Iterator[dict]:
    """Interleave configured language/source streams.

    Weight keys are ``language/<code>``, ``source/<name>``, or
    ``source/<name>/language/<code>``.
    """
    pool = WeightedRecordPool(split, [key for key, value in weights.items() if value > 0],
                              seed, source_override=sources)
    yield from scheduled_records(pool, lambda: weights)


def pack_records(records: Iterator[dict], tokenizer, sequence_length: int = 1024,
                 batch_size: int = 1):
    """Yield CPU torch.long input/target batches; EOS closes each document."""
    if sequence_length not in {512, 1024}:
        raise ValueError("sequence_length must be 512 or 1024")
    import torch
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        raise ValueError("tokenizer must define <|eos|>")
    buf: list[int] = []
    batch = []
    for record in records:
        buf.extend(tokenizer.encode(record["text"]).ids)
        buf.append(eos)
        while len(buf) >= sequence_length + 1:
            chunk, buf = buf[:sequence_length + 1], buf[sequence_length + 1:]
            batch.append(chunk)
            if len(batch) == batch_size:
                tensor = torch.tensor(batch, dtype=torch.long)
                yield tensor[:, :-1], tensor[:, 1:]
                batch = []


def source_language_counts(split: str = "train") -> dict:
    counts = defaultdict(int)
    for r in iter_records(split):
        counts[(r["_source"], r["_language_code"])] += 1
    return {f"{s}:{l}": n for (s, l), n in sorted(counts.items())}
