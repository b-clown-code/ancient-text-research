from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag_pilot"))
sys.path.insert(0, str(ROOT))

import dense  # noqa: E402
import rerank  # noqa: E402
import system  # noqa: E402
from pretrain.model import AncientDecoder, ModelConfig, parameter_count  # noqa: E402
from pretrain.progressive import model_parameter_count  # noqa: E402


def make_index(identity="identity-a", vector_id=1, vector=(1.0, 0.0)):
    db = __import__("sqlite3").connect(":memory:")
    db.row_factory = __import__("sqlite3").Row
    db.executescript(system.SCHEMA)
    record = {"text": "synthetic passage", "work_id": "work-a", "language": "en"}
    db.execute("INSERT INTO metadata VALUES (?,?)", ("model", "model-a"))
    db.execute("INSERT INTO metadata VALUES (?,?)", ("revision", "revision-a"))
    db.execute("INSERT INTO metadata VALUES (?,?)", ("encoder_identity", identity))
    db.execute("INSERT INTO metadata VALUES (?,?)", ("embedding_dimension", "2"))
    db.execute("INSERT INTO passage VALUES(?,?,?,?,?)",
               ("passage-a", json.dumps(record), "fixture", "fixture-hash", 1))
    db.execute("INSERT INTO window(id,text,source,language,work_id,document_id,vector) "
               "VALUES(?,?,?,?,?,?,?)",
               (vector_id, "synthetic passage", "fixture", "en", "work-a", "doc-a",
                np.asarray(vector, dtype="<f4").tobytes()))
    db.execute("INSERT INTO lexical(rowid,text) VALUES(?,?)", (vector_id, "synthetic passage"))
    db.execute("INSERT INTO mapping VALUES(?,?,?,?,?)",
               (vector_id, "passage-a", 0, len("synthetic passage"), 0))
    db.commit()
    return db


class FakeEncoder:
    dimension = 2

    def __init__(self, identity):
        self.identity = identity
        self.encoded = False

    def get_index_identity(self):
        return self.identity

    def encode_queries(self, queries, **kwargs):
        self.encoded = True
        return np.asarray([[1.0, 0.0] for _ in queries], dtype=np.float32)

    def close(self):
        pass


class DenseIdentityTests(unittest.TestCase):
    def test_transformer_loaders_receive_requested_revision_and_opt_in(self):
        tokenizer_loader = Mock()
        tokenizer_loader.from_pretrained.return_value = Mock()
        model = Mock()
        model.config = SimpleNamespace(max_position_embeddings=128, hidden_size=4,
                                       is_decoder=False)
        model.to.return_value = model
        model_loader = Mock()
        model_loader.from_pretrained.return_value = model
        transformers = SimpleNamespace(AutoTokenizer=tokenizer_loader, AutoModel=model_loader)
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules,
                                                                 {"transformers": transformers}):
            encoder = dense.Encoder("sample/encoder", "abc123", Path(temp) / "cache.sqlite",
                                    device="cpu")
            encoder.close()
            trusted = dense.Encoder("sample/encoder", "abc123", Path(temp) / "trusted.sqlite",
                                    device="cpu", trust_remote_code=True)
            trusted.close()
        self.assertEqual(tokenizer_loader.from_pretrained.call_args.kwargs["revision"], "abc123")
        self.assertEqual(model_loader.from_pretrained.call_args.kwargs["revision"], "abc123")
        self.assertFalse(tokenizer_loader.from_pretrained.call_args_list[0].kwargs["trust_remote_code"])
        self.assertFalse(model_loader.from_pretrained.call_args_list[0].kwargs["trust_remote_code"])
        self.assertTrue(tokenizer_loader.from_pretrained.call_args_list[1].kwargs["trust_remote_code"])
        self.assertTrue(model_loader.from_pretrained.call_args_list[1].kwargs["trust_remote_code"])

    def test_sentence_transformer_loader_receives_revision(self):
        model = Mock()
        model.tokenizer = Mock()
        model.max_seq_length = 64
        model.get_sentence_embedding_dimension.return_value = 4
        loader = Mock(return_value=model)
        transformers_package = SimpleNamespace(SentenceTransformer=loader)
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "snapshot"
            model_path.mkdir()
            (model_path / "modules.json").write_text("[]")
            with tempfile.TemporaryDirectory() as cache, patch.dict(
                    sys.modules, {"sentence_transformers": transformers_package, "transformers": None}):
                encoder = dense.Encoder(model_path, "fedcba", Path(cache) / "cache.sqlite",
                                        device="cpu")
                encoder.close()
                trusted = dense.Encoder(model_path, "fedcba", Path(cache) / "trusted.sqlite",
                                        device="cpu", trust_remote_code=True)
                trusted.close()
        self.assertEqual(loader.call_args.kwargs["revision"], "fedcba")
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertTrue(loader.call_args.kwargs["trust_remote_code"])
        self.assertFalse(loader.call_args_list[0].kwargs["trust_remote_code"])

    def test_reranker_loaders_receive_revision_and_remote_code_is_opt_in(self):
        tokenizer_loader = Mock()
        model = Mock()
        model.to.return_value = model
        model_loader = Mock()
        model_loader.from_pretrained.return_value = model
        transformers = SimpleNamespace(AutoTokenizer=tokenizer_loader,
                                       AutoModelForSequenceClassification=model_loader)
        with patch.dict(sys.modules, {"transformers": transformers}):
            rerank.Reranker("sample/reranker", revision="def456", device="cpu")
            rerank.Reranker("sample/reranker", revision="def456", device="cpu",
                            trust_remote_code=True)
        self.assertEqual(tokenizer_loader.from_pretrained.call_args.kwargs["revision"], "def456")
        self.assertEqual(model_loader.from_pretrained.call_args.kwargs["revision"], "def456")
        self.assertFalse(tokenizer_loader.from_pretrained.call_args_list[0].kwargs["trust_remote_code"])
        self.assertFalse(model_loader.from_pretrained.call_args_list[0].kwargs["trust_remote_code"])
        self.assertTrue(tokenizer_loader.from_pretrained.call_args_list[1].kwargs["trust_remote_code"])
        self.assertTrue(model_loader.from_pretrained.call_args_list[1].kwargs["trust_remote_code"])
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            rerank.Reranker("sample/reranker")

    def test_supplied_and_internal_encoders_must_match_index_identity(self):
        db = make_index()
        matching = FakeEncoder("identity-a")
        result = system.search_unified(db, "synthetic", mode="dense", encoder=matching)
        self.assertEqual(result["results"][0]["passages"][0]["id"], "passage-a")

        mismatching = FakeEncoder("identity-b")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            system.search_unified(db, "synthetic", mode="dense", encoder=mismatching)
        self.assertFalse(mismatching.encoded)

        with patch.object(system, "Encoder", return_value=mismatching):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                system.search_unified(db, "synthetic", mode="dense", model="model-a",
                                      revision="revision-a", cache="unused.sqlite")
        with patch.object(system, "Encoder", return_value=matching):
            internal_result = system.search_unified(
                db, "synthetic", mode="dense", model="model-a",
                revision="revision-a", cache="unused.sqlite")
        self.assertEqual(internal_result["results"][0]["passages"][0]["id"], "passage-a")
        db.close()


class VectorCacheTests(unittest.TestCase):
    def test_cache_is_connection_scoped_refreshes_on_write_and_clears(self):
        import sqlite3

        cache = system.VectorMatrixCache()
        first = sqlite3.connect(":memory:")
        second = sqlite3.connect(":memory:")
        for db in (first, second):
            db.execute("CREATE TABLE window(id INTEGER, vector BLOB)")
        first.execute("INSERT INTO window VALUES(1,?)", (np.asarray([1, 0], dtype="<f4").tobytes(),))
        second.execute("INSERT INTO window VALUES(2,?)", (np.asarray([0, 1], dtype="<f4").tobytes(),))
        first.commit()
        second.commit()

        ids_a, vec_a = system.get_vector_matrix(first, cache=cache)
        ids_b, vec_b = system.get_vector_matrix(second, cache=cache)
        self.assertEqual(ids_a.tolist(), [1])
        self.assertEqual(ids_b.tolist(), [2])
        np.testing.assert_array_equal(vec_b[0], [0, 1])

        second.execute("UPDATE window SET vector=? WHERE id=2",
                       (np.asarray([1, 0], dtype="<f4").tobytes(),))
        second.commit()
        _, refreshed = system.get_vector_matrix(second, cache=cache)
        np.testing.assert_array_equal(refreshed[0], [1, 0])
        # Reading uncommitted updates must not poison the cache after rollback.
        second.execute("UPDATE window SET vector=? WHERE id=2",
                       (np.asarray([0, 1], dtype="<f4").tobytes(),))
        _, uncommitted = system.get_vector_matrix(second, cache=cache)
        np.testing.assert_array_equal(uncommitted[0], [0, 1])
        second.rollback()
        _, restored = system.get_vector_matrix(second, cache=cache)
        np.testing.assert_array_equal(restored[0], [1, 0])
        cache.clear()
        self.assertIsNone(cache.connection)
        self.assertEqual(cache.matrices, {})
        first.close()
        second.close()


class PublicInterfaceTests(unittest.TestCase):
    def test_tied_and_untied_parameter_counts_match_model(self):
        for tie_embeddings in (True, False):
            config = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2,
                                 n_kv_heads=1, ffn_dim=32, max_seq_len=8,
                                 tie_embeddings=tie_embeddings,
                                 activation_checkpointing=False)
            actual = sum(parameter.numel() for parameter in AncientDecoder(config).parameters())
            self.assertEqual(parameter_count(config), actual)
            self.assertEqual(model_parameter_count(config), actual)
        self.assertEqual(parameter_count(ModelConfig(vocab_size=32, d_model=16,
                                                     n_layers=1, n_heads=2,
                                                     n_kv_heads=1, ffn_dim=32,
                                                     max_seq_len=8,
                                                     tie_embeddings=False)), 3376)

    def test_curriculum_requires_an_explicit_existing_config(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "must-not-be-created"
            result = subprocess.run(
                [sys.executable, "-m", "pretrain.train", "--tokenizer", "missing.json",
                 "--output", str(output), "--sampling-mode", "curriculum"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--curriculum-config is required", result.stderr)
            self.assertFalse(output.exists())

    def test_catalog_help_works_without_workspace_helpers(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for script in ("catalog_corpus.py", "catalog_documents.py"):
            result = subprocess.run([sys.executable, str(ROOT / "tools" / script), "--help"],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout.lower())

    def test_catalog_build_reports_omitted_runtime_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "catalog"
            for script, expected in (("catalog_corpus.py", "workspace curation runtime modules"),
                                     ("catalog_documents.py", "workspace curation_schema module")):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / script),
                     "--root", temp, "--out", str(output)],
                    cwd=ROOT, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
