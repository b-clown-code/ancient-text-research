"""Run lexical retrieval and citation formatting on generated fixture passages."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag_pilot"))

from generate import build_rag_prompt  # noqa: E402
from system import SCHEMA, search_unified  # noqa: E402


def main() -> None:
    rows = [
        ("synthetic-1", "The river carries boats toward the sea.", "work-river"),
        ("synthetic-2", "Farmers gather grain after the harvest.", "work-harvest"),
    ]
    with tempfile.TemporaryDirectory(prefix="ancient-text-retrieval-") as temp:
        db = sqlite3.connect(Path(temp) / "synthetic.sqlite")
        db.row_factory = sqlite3.Row
        db.executescript(SCHEMA)
        with db:
            for passage_id, text, work_id in rows:
                record = {
                    "text": text,
                    "language": "en-synthetic",
                    "source": "fixture",
                    "work_id": work_id,
                    "document_id": f"doc-{work_id}",
                    "locator": "fixture:1",
                }
                db.execute(
                    "INSERT INTO passage VALUES(?,?,?,?,?)",
                    (passage_id, json.dumps(record), "synthetic-fixture", "fixture", 1),
                )
                cursor = db.execute(
                    "INSERT INTO window(text,source,language,work_id,document_id,vector) "
                    "VALUES(?,?,?,?,?,?)",
                    (text, "fixture", "en-synthetic", work_id, f"doc-{work_id}",
                     bytes(384 * 4)),
                )
                window_id = cursor.lastrowid
                db.execute("INSERT INTO lexical(rowid,text) VALUES(?,?)", (window_id, text))
                db.execute(
                    "INSERT INTO mapping VALUES(?,?,?,?,?)",
                    (window_id, passage_id, 0, len(text), 0),
                )

        found = search_unified(db, "boats river", mode="lexical", limit=2)
        prompt = build_rag_prompt("What travels on the river?", found["results"])
        print(json.dumps({
            "retrieved_passages": prompt["available_passage_ids"],
            "citation_context": prompt["full_prompt"].split("CONTEXT PASSAGES:", 1)[1]
                                      .split("USER QUERY:", 1)[0].strip(),
        }, ensure_ascii=False, indent=2))
        assert prompt["available_passage_ids"] == ["synthetic-1"]
        db.close()


if __name__ == "__main__":
    main()
