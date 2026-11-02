from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.cpu_transformer_step import make_tokenizer  # noqa: E402
from pretrain.model import AncientDecoder, ModelConfig  # noqa: E402


class PublicFixtureTests(unittest.TestCase):
    def test_synthetic_retrieval_and_citation_fixture(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "retrieval_synthetic.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["retrieved_passages"], ["synthetic-1"])
        self.assertIn("[passage:synthetic-1]", report["citation_context"])

    def test_tokenizer_round_trip_on_generated_fixture(self):
        text = "Synthetic Greek: λόγος. <|eos|>"
        tokenizer = make_tokenizer([text, "Synthetic Latin: verbum. <|eos|>"])
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(text).ids, skip_special_tokens=False),
            text,
        )

    def test_small_model_forward_loss_backward_and_optimizer_step(self):
        torch.manual_seed(11)
        config = ModelConfig(
            vocab_size=32,
            d_model=16,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            ffn_dim=32,
            max_seq_len=8,
            activation_checkpointing=False,
        )
        model = AncientDecoder(config).cpu().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        before = model.embed_tokens.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, labels=targets)
        self.assertEqual(tuple(output["logits"].shape), (1, 4, 32))
        self.assertTrue(math.isfinite(float(output["loss"].detach())))
        output["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.embed_tokens.weight.detach()))


if __name__ == "__main__":
    unittest.main()
