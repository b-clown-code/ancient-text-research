#!/usr/bin/env python3
"""Freeze and verify the immutable raw layer against catalog/corpus.sqlite.

The catalog database remains authoritative.  This script creates a compressed,
human-auditable manifest and re-hashes every catalogued payload file so that a
later run can detect changes without moving or rewriting the raw sources.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "corpus.sqlite"
ACQUISITION = ROOT / "ACQUISITION-2026-10-21.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(base: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(base), *args], capture_output=True, text=True
    )
    if result.returncode:
        return None
    return result.stdout.strip() or None


def mount_info(path: Path) -> dict[str, str | None]:
    result = subprocess.run(
        ["findmnt", "-T", str(path), "-rn", "-o", "SOURCE,FSTYPE,OPTIONS"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return {"source": None, "fstype": None, "options": None}
    fields = result.stdout.strip().split(" ", 2)
    fields += [None] * (3 - len(fields))
    return {"source": fields[0], "fstype": fields[1], "options": fields[2]}


def acquisition_date() -> tuple[str | None, str]:
    if not ACQUISITION.exists():
        return None, "not_recorded"
    text = ACQUISITION.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?:acquisition|round)\D+(20\d{2}-\d{2}-\d{2})", text, re.I)
    if match:
        return match.group(1), "ACQUISITION-2026-10-21.md"
    return None, "not_recorded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="verify without replacing manifests")
    args = parser.parse_args()
    if not CATALOG.exists():
        raise SystemExit(f"Missing catalog database: {CATALOG}")

    db = sqlite3.connect(CATALOG)
    db.row_factory = sqlite3.Row
    sources = db.execute("SELECT * FROM sources ORDER BY source").fetchall()
    rows = db.execute(
        "SELECT id,source,path,bytes,mtime_ns,format,role,sha256,representation_family,admission "
        "FROM files ORDER BY id"
    )
    recorded_date, date_basis = acquisition_date()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    errors: list[dict[str, object]] = []
    changed = 0
    checked = 0
    source_meta: list[dict[str, object]] = []
    archive_count = 0

    manifest_path = ROOT / "catalog" / "raw_manifest.jsonl.gz"
    tmp_manifest = manifest_path.with_suffix(".tmp.gz")
    stream = None if args.verify_only else gzip.open(tmp_manifest, "wt", encoding="utf-8", newline="\n")
    try:
        for source_row in sources:
            source = source_row["source"]
            base = ROOT / source
            mount = mount_info(base) if base.exists() else {"source": None, "fstype": None, "options": None}
            commit = git_value(base, "rev-parse", "HEAD") if base.is_dir() else None
            commit_date = git_value(base, "show", "-s", "--format=%cI", "HEAD") if commit else None
            source_record = {
                "source": source,
                "root_path": str(base),
                "root_exists": base.is_dir(),
                "storage_source": mount["source"],
                "filesystem": mount["fstype"],
                "mount_options": mount["options"],
                "origin": source_row["origin"],
                "catalog_commit_sha": source_row["commit_sha"],
                "current_commit_sha": commit,
                "commit_date": commit_date,
                "license_claim": source_row["license_claim"],
                "provenance_family": source_row["provenance_family"],
                "rights_review": source_row["rights_review"],
                "admission": source_row["admission"],
                "retrieval_date": recorded_date,
                "retrieval_date_basis": date_basis,
            }
            source_meta.append(source_record)
            if not base.is_dir():
                errors.append({"stage": "source", "source": source, "error": "source_directory_missing"})
            if source_row["commit_sha"] and commit != source_row["commit_sha"]:
                errors.append({"stage": "git", "source": source, "error": "commit_mismatch", "expected": source_row["commit_sha"], "actual": commit})

        for row in rows:
            checked += 1
            path = ROOT / row["path"]
            status = "ok"
            actual_size = None
            actual_mtime_ns = None
            actual_sha256 = None
            try:
                stat = path.stat()
                actual_size = stat.st_size
                actual_mtime_ns = stat.st_mtime_ns
                actual_sha256 = sha256_file(path)
                if actual_size != row["bytes"] or actual_sha256 != row["sha256"]:
                    status = "changed"
                    changed += 1
                    errors.append({"stage": "file", "path": row["path"], "error": "content_mismatch", "expected_sha256": row["sha256"], "actual_sha256": actual_sha256, "expected_bytes": row["bytes"], "actual_bytes": actual_size})
            except OSError as exc:
                status = "missing_or_unreadable"
                errors.append({"stage": "file", "path": row["path"], "error": str(exc)})
            if stream is not None:
                stream.write(json.dumps({
                    "record_type": "raw_file",
                    "catalog_file_id": row["id"],
                    "source": row["source"],
                    "path": row["path"],
                    "storage_path": str(path),
                    "bytes": row["bytes"],
                    "mtime_ns": row["mtime_ns"],
                    "format": row["format"],
                    "role": row["role"],
                    "sha256": row["sha256"],
                    "representation_family": row["representation_family"],
                    "admission": row["admission"],
                    "verification": status,
                    "observed_bytes": actual_size,
                    "observed_mtime_ns": actual_mtime_ns,
                    "observed_sha256": actual_sha256,
                }, ensure_ascii=False) + "\n")
            if row["format"] == ".zip":
                archive_count += 1
        if stream is not None:
            stream.close()
            os.replace(tmp_manifest, manifest_path)
    finally:
        if stream is not None and not stream.closed:
            stream.close()
        if tmp_manifest.exists() and (args.verify_only or stream is None):
            tmp_manifest.unlink(missing_ok=True)

    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = {
        "record_type": "raw_freeze",
        "catalog": str(CATALOG),
        "started_utc": started,
        "completed_utc": finished,
        "root_path": str(ROOT),
        "catalog_files_checked": checked,
        "changed_or_missing": changed + sum(1 for e in errors if e["stage"] == "file" and e["error"] != "content_mismatch"),
        "source_count": len(sources),
        "archive_file_count": archive_count,
        "error_count": len(errors),
        "manifest": str(manifest_path) if not args.verify_only else None,
        "sources": source_meta,
        "errors": errors,
    }
    (ROOT / "catalog" / "raw_freeze.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Raw layer freeze",
        "",
        f"Catalogued files checked: {checked:,}",
        f"Sources: {len(sources):,}",
        f"Archive files: {archive_count:,}",
        f"Verification errors: {len(errors):,}",
        "",
        "The catalog database remains authoritative. `raw_manifest.jsonl.gz` is a compressed verification snapshot; it contains paths, hashes, sizes, source provenance, and verification status, but does not copy raw payloads.",
        "",
        "Source roots and storage mounts:",
        "",
        "| Source | Root | Storage | Commit | Retrieval date | Rights claim |",
        "|---|---|---|---|---|---|",
    ]
    for item in source_meta:
        report.append(f"| {item['source']} | `{item['root_path']}` | `{item['storage_source'] or 'unknown'}` | `{item['current_commit_sha'] or 'archive/non-git'}` | {item['retrieval_date'] or 'not recorded'} | {item['license_claim']} |")
    if errors:
        report += ["", "## Verification errors", "", "See `raw_freeze.json` for complete details."]
        report += [f"- `{e.get('stage')}` `{e.get('source', e.get('path', ''))}`: {e.get('error')}" for e in errors[:50]]
    else:
        report += ["", "All catalogued files matched their recorded size and SHA-256 hash; Git source HEADs matched the catalog commits."]
    (ROOT / "catalog" / "RAW_FREEZE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    db.close()
    print(json.dumps({"checked": checked, "errors": len(errors), "manifest": str(manifest_path) if not args.verify_only else None}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
