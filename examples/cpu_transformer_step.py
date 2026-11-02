"""Train the included AncientDecoder for one CPU step on generated text."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pretrain.model import AncientDecoder, ModelConfig, parameter_count  # noqa: E402


def make_tokenizer(texts: list[str]) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=300,
            min_frequency=1,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|eos|>"],
            show_progress=False,
        ),
    )
    tokenizer.add_special_tokens(["<|eos|>"])
    return tokenizer


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    synthetic_text = [
        "The river carries boats toward the sea. <|eos|>",
        "Farmers gather grain after the harvest. <|eos|>",
    ]
    tokenizer = make_tokenizer(synthetic_text)
    token_ids = tokenizer.encode(" ".join(synthetic_text)).ids
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(with_added_tokens=True),
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=64,
        max_seq_len=32,
        activation_checkpointing=False,
    )
    model = AncientDecoder(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.tensor([token_ids[:-1]], dtype=torch.long)
    y = torch.tensor([token_ids[1:]], dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    loss = model(x, labels=y)["loss"]
    loss.backward()
    optimizer.step()

    print(json.dumps({
        "device": "cpu",
        "parameters": parameter_count(config),
        "sequence_tokens": int(x.shape[1]),
        "optimizer_steps": 1,
        "loss": round(float(loss.detach()), 6),
    }, indent=2))


if __name__ == "__main__":
    main()
