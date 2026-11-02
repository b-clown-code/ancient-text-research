"""Offline neural embedding adapter. Supports SentenceTransformer and AutoModel architectures.

Supports asymmetric query/passage encoding, task instructions, causal (last_token) and
encoder (mean, cls) pooling, Matryoshka dimension truncation, and composite cryptographic
cache identity verification.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

DEFAULT_TASK_INSTRUCTIONS = {
    "modern_query_to_primary_evidence": "Given a modern research question, retrieve relevant ancient primary literature passages that answer or substantiate the question",
    "ancient_passage_to_passage_parallel": "Given an ancient literature passage, retrieve relevant textual parallels, citations, allusions, or direct translations across ancient traditions",
}


def compute_weights_sha256(model_path):
    mpath = Path(model_path).resolve()
    for fname in ('model.safetensors', 'pytorch_model.bin', 'rust_model.ot'):
        wpath = mpath / fname
        if wpath.is_file():
            h = hashlib.sha256()
            with open(wpath, 'rb') as f:
                while chunk := f.read(1048576):
                    h.update(chunk)
            return fname, h.hexdigest()
    # Check for sharded weights
    shards = sorted(mpath.glob("model-*.safetensors"))
    if shards:
        h = hashlib.sha256()
        for s in shards:
            with open(s, 'rb') as f:
                while chunk := f.read(1048576):
                    h.update(chunk)
        return "sharded.safetensors", h.hexdigest()
    return None, None


class Encoder:
    def __init__(
        self,
        model_path,
        revision,
        cache_path,
        batch_size=16,
        device=None,
        pooling=None,              # "mean", "last_token", "cls", or None (auto)
        dimension=None,            # MRL truncation target dimension or None
        max_length=None,           # Max sequence length or None (model default)
        default_query_instruction=None,
        trust_remote_code=False
    ):
        if not revision:
            raise ValueError('An immutable model revision is required')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')

        self.model_path = Path(model_path).resolve() if Path(model_path).exists() else model_path
        self.revision = revision
        self.batch_size = batch_size
        self.default_query_instruction = default_query_instruction
        self.trust_remote_code = bool(trust_remote_code)

        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                device = 'cpu'
        self.device = device

        # Cryptographically verify model weight file if local
        self.weight_file, self.weight_sha256 = (None, None)
        if isinstance(self.model_path, Path) and self.model_path.is_dir():
            self.weight_file, self.weight_sha256 = compute_weights_sha256(self.model_path)

        # Detect model architecture and initialize backend
        self._init_backend(pooling, max_length, dimension)

        # Connect to SQLite embedding cache
        self.cache_path = Path(cache_path).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.cache_path))
        self.db.execute('PRAGMA synchronous = NORMAL')
        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.execute(
            'CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dimensions INTEGER NOT NULL, vector BLOB NOT NULL)'
        )
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        if self.weight_sha256:
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('weight_sha256', self.weight_sha256))
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('weight_file', self.weight_file or ''))
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('model_path', str(self.model_path)))
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('revision', revision))
        self.db.commit()

    def _init_backend(self, pooling, max_length, dimension):
        import torch

        model_str = str(self.model_path)
        # Try loading with SentenceTransformer first if it has modules.json
        has_st_modules = isinstance(self.model_path, Path) and (self.model_path / "modules.json").is_file()
        if has_st_modules:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(
                model_str, revision=self.revision, device=self.device,
                local_files_only=isinstance(self.model_path, Path),
                trust_remote_code=self.trust_remote_code,
            )
            self.tokenizer = self.st_model.tokenizer
            self.max_length = max_length or self.st_model.max_seq_length
            native_dim = self.st_model.get_sentence_embedding_dimension()
            self.pooling_method = pooling or "mean"
            self.dimension = dimension or native_dim
            self.use_st = True
            self.model = self.st_model
        else:
            # Use AutoModel / AutoTokenizer directly
            from transformers import AutoTokenizer, AutoModel
            local_files_only = isinstance(self.model_path, Path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_str, revision=self.revision,
                local_files_only=local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            self.hf_model = AutoModel.from_pretrained(
                model_str,
                revision=self.revision,
                local_files_only=local_files_only,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.hf_model.eval()
            self.model = self.hf_model

            # Determine max sequence length
            self.max_length = max_length or getattr(self.hf_model.config, 'max_position_embeddings', 512)
            if self.max_length is None or self.max_length > 8192:
                self.max_length = 512

            # Determine native dimension
            native_dim = getattr(self.hf_model.config, 'hidden_size', None)
            if native_dim is None:
                native_dim = getattr(self.hf_model.config, 'd_model', 384)
            self.dimension = dimension or native_dim

            # Determine pooling: causal models (Qwen, Gemma, Llama) default to last_token; encoder models default to mean
            if pooling:
                self.pooling_method = pooling
            else:
                is_causal = getattr(self.hf_model.config, 'is_decoder', False) or 'qwen' in model_str.lower() or 'gemma' in model_str.lower()
                self.pooling_method = "last_token" if is_causal else "mean"
            self.use_st = False

    def tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def get_cache_identity(self, encoding_type="passage", task=None, instruction=None):
        config = {
            "weight_sha256": self.weight_sha256 or "none",
            "revision": self.revision,
            "model_path": str(self.model_path),
            "encoding_type": encoding_type,
            "task": task if encoding_type == "query" else None,
            "instruction": instruction if encoding_type == "query" else None,
            "pooling": self.pooling_method,
            "max_length": self.max_length,
            "dimension": self.dimension,
            "trust_remote_code": self.trust_remote_code,
            "normalize": True
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def get_index_identity(self):
        """Stable identity for passage vectors stored in a retrieval index."""
        return self.get_cache_identity(encoding_type="passage")

    def format_query(self, query, task="modern_query_to_primary_evidence", instruction=None):
        """Format an asymmetric search query with instruction header if applicable."""
        instr = instruction if instruction is not None else (self.default_query_instruction or DEFAULT_TASK_INSTRUCTIONS.get(task, ""))
        model_str = str(self.model_path).lower()
        if "mitra" in model_str:
            mitra_instr = instr if instr else "Please find the semantically most similar text."
            return f"<instruct>{mitra_instr}\n<query>{query}"
        elif instr and "qwen" in model_str:
            # Qwen3-Embedding instruction format
            return f"Instruct: {instr}\nQuery: {query}"
        elif instr:
            return f"Instruct: {instr}\nQuery: {query}"
        return query

    def encode_queries(self, queries, task="modern_query_to_primary_evidence", instruction=None):
        """Encode asymmetric queries with task instructions."""
        formatted = [self.format_query(q, task=task, instruction=instruction) for q in queries]
        return self._encode_batch(formatted, encoding_type="query", task=task, instruction=instruction, raw_texts=queries)

    def encode_passages(self, passages):
        """Encode unadorned candidate context passages."""
        return self._encode_batch(passages, encoding_type="passage", task=None, instruction=None, raw_texts=passages)

    def encode(self, texts, is_query=False, task="modern_query_to_primary_evidence", instruction=None):
        """Unified encoding method supporting both passages and queries for backward compatibility."""
        if is_query:
            return self.encode_queries(texts, task=task, instruction=instruction)
        return self.encode_passages(texts)

    def _encode_batch(self, texts, encoding_type="passage", task=None, instruction=None, raw_texts=None):
        texts = list(texts)
        raw_texts = list(raw_texts) if raw_texts is not None else texts
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cache_id = self.get_cache_identity(encoding_type=encoding_type, task=task, instruction=instruction)

        # For backward compatibility with existing MiniLM baseline cache:
        legacy_weight_id = hashlib.sha256(((self.weight_sha256 or '') + '\0' + self.revision).encode()).hexdigest() if self.weight_sha256 else None
        legacy_path_id = hashlib.sha256((str(self.model_path) + '\0' + self.revision).encode()).hexdigest()

        keys = []
        legacy_keys = []
        for text, raw in zip(texts, raw_texts):
            if self.tokens(text) > self.max_length:
                raise ValueError(f'Input ({self.tokens(text)} tokens) exceeds model token limit ({self.max_length}); split before embedding')
            # Cache key incorporates composite identity + input text
            keys.append(hashlib.sha256((cache_id + '\0' + text).encode('utf-8')).hexdigest())
            if legacy_weight_id:
                legacy_keys.append(hashlib.sha256((legacy_weight_id + '\0' + raw).encode('utf-8')).hexdigest())
            else:
                legacy_keys.append(hashlib.sha256((legacy_path_id + '\0' + raw).encode('utf-8')).hexdigest())

        found, missing = {}, {}
        for key, leg_key, text in zip(keys, legacy_keys, texts):
            # Try composite key first, then legacy key for passages
            row = self.db.execute('SELECT dimensions,vector FROM embeddings WHERE key=?', (key,)).fetchone()
            if not row and encoding_type == "passage" and leg_key:
                row = self.db.execute('SELECT dimensions,vector FROM embeddings WHERE key=?', (leg_key,)).fetchone()
                if row and row[0] == self.dimension:
                    # Upgrade legacy entry to new composite key
                    self.db.execute('INSERT OR IGNORE INTO embeddings VALUES (?,?,?)', (key, row[0], row[1]))

            if row:
                vec = np.frombuffer(row[1], dtype='<f4').copy()
                if row[0] != self.dimension or len(vec) != self.dimension or not np.isfinite(vec).all():
                    raise ValueError('Invalid cached vector')
                found[key] = vec
            else:
                missing[key] = text

        items = list(missing.items())
        if items:
            missing_texts = [x[1] for x in items]
            missing_keys = [x[0] for x in items]
            computed_vectors = self._compute_vectors(missing_texts)
            with self.db:
                for key, vec in zip(missing_keys, computed_vectors):
                    if vec.shape != (self.dimension,) or not np.isfinite(vec).all():
                        raise ValueError('Invalid model output')
                    self.db.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?,?)', (key, self.dimension, vec.tobytes()))
                    found[key] = vec

        return np.stack([found[key] for key in keys])

    def _compute_vectors(self, texts):
        import torch
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            if self.use_st:
                vecs = self.st_model.encode(
                    batch_texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                if self.dimension < vecs.shape[1]:
                    vecs = vecs[:, :self.dimension]
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    vecs = vecs / np.maximum(norms, 1e-9)
                results.append(vecs.astype(np.float32))
            else:
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.hf_model(**inputs)
                    last_hidden = outputs.last_hidden_state
                    mask = inputs.attention_mask

                    if self.pooling_method == "last_token":
                        # Causal EOS pooling: locate last non-padded token
                        left_padded = (mask[:, -1].sum().item() == mask.shape[0])
                        if left_padded:
                            pooled = last_hidden[:, -1]
                        else:
                            seq_lengths = mask.sum(dim=1) - 1
                            batch_idx = torch.arange(last_hidden.shape[0], device=self.device)
                            pooled = last_hidden[batch_idx, seq_lengths]
                    elif self.pooling_method == "cls":
                        pooled = last_hidden[:, 0]
                    else:  # "mean"
                        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden.size()).float()
                        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        pooled = sum_embeddings / sum_mask

                    if self.dimension < pooled.shape[1]:
                        pooled = pooled[:, :self.dimension]

                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    vecs = pooled.to(torch.float32).cpu().numpy()
                    results.append(vecs)

        return np.concatenate(results, axis=0) if results else np.empty((0, self.dimension), dtype=np.float32)

    def close(self):
        self.db.close()


def exact_search(query, vectors, limit=10):
    """Return (row index, cosine similarity), ties ordered by row index."""
    vectors = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if limit < 1 or vectors.ndim != 2 or query.shape != (vectors.shape[1],):
        raise ValueError('Invalid search dimensions or limit')
    scores = vectors @ query
    return [(int(i), float(scores[i])) for i in np.lexsort((np.arange(len(scores)), -scores))[:limit]]
