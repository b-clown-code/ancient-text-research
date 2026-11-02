"""Neural Cross-Encoder Reranker adapter and fallback rescorer.

Reranks candidate retrieval windows using pair cross-attention scoring (query, passage).
"""
import hashlib
from pathlib import Path
import numpy as np


class Reranker:
    def __init__(self, model_path=None, revision=None, device=None, batch_size=16,
                 trust_remote_code=False):
        self.model_path = Path(model_path).resolve() if model_path and Path(model_path).exists() else model_path
        self.revision = revision
        self.batch_size = batch_size
        self.trust_remote_code = bool(trust_remote_code)
        if model_path and not isinstance(self.model_path, Path) and not revision:
            raise ValueError("an immutable revision is required for remote reranker models")

        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                device = 'cpu'
        self.device = device

        self.model = None
        self.tokenizer = None
        if self.model_path:
            self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_str = str(self.model_path)
        local_files_only = isinstance(self.model_path, Path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_str, revision=self.revision, local_files_only=local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_str,
            revision=self.revision,
            local_files_only=local_files_only,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def score_pairs(self, query, texts):
        """Compute relevance scores for a list of (query, text) pairs."""
        if not texts:
            return np.empty(0, dtype=np.float32)

        if self.model is None:
            # Fallback lexical overlap / character Jaccard rescorer
            q_tokens = set(query.lower().split())
            scores = []
            for t in texts:
                t_tokens = set(t.lower().split())
                overlap = len(q_tokens & t_tokens)
                jaccard = overlap / max(len(q_tokens | t_tokens), 1)
                scores.append(jaccard)
            return np.array(scores, dtype=np.float32)

        import torch
        scores = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            pairs = [[query, t] for t in batch_texts]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1).float().cpu().numpy()
                elif logits.shape[-1] == 2:
                    # Binary classification (0: irrelevant, 1: relevant) -> sigmoid / softmax
                    batch_scores = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
                else:
                    batch_scores = logits[:, 0].float().cpu().numpy()
                scores.append(batch_scores)

        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float32)

    def rerank(self, query, candidates, top_k=10):
        """Rerank a list of candidate items returned by search_unified()."""
        if not candidates:
            return []

        # Extract text from candidate window or passages
        texts = []
        for c in candidates:
            w_text = c.get('window', {}).get('text', '')
            if not w_text and c.get('passages'):
                w_text = ' '.join(p.get('record', {}).get('text', '') for p in c['passages'])
            texts.append(w_text)

        scores = self.score_pairs(query, texts)

        rescored = []
        for idx, (c, score) in enumerate(zip(candidates, scores)):
            item = dict(c)
            retrieval_meta = dict(item.get('retrieval', {}))
            retrieval_meta['rerank_score'] = float(score)
            retrieval_meta['original_rank'] = idx + 1
            item['retrieval'] = retrieval_meta
            rescored.append((score, item))

        # Sort descending by reranker score
        rescored.sort(key=lambda x: -x[0])
        return [item for _, item in rescored[:top_k]]
