# Setup and synthetic examples

## Requirements

- Python 3.10 or newer.
- SQLite built with FTS5 (the retrieval example checks this at database setup).
- `numpy` for retrieval internals.
- `torch` and `tokenizers` for the CPU Transformer example.

From the repository root, install the focused demo dependencies into a virtual
environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch tokenizers
```

The included catalog fixtures exercise the standalone classification and
metadata parsers using Python's standard library. Full catalog assembly and raw
freeze commands also expect the workspace-only curation runtime and inventory
artifacts, which are deliberately omitted; those end-to-end commands are not a
reproduction path in this portfolio snapshot. Dense neural
retrieval additionally needs `transformers` and either `sentence-transformers`
or a compatible local Transformers model; model files are intentionally not
included. When loading a remote model, provide its immutable Hub commit with
`--revision`; that revision is passed to the tokenizer and model loaders and is
part of the index compatibility identity. Custom model Python is disabled by
default; `--trust-remote-code` explicitly opts in. Dense queries must use the
same model revision, weights, pooling, and embedding dimension as the index.
Older indexes without an encoder identity need to be rebuilt before dense
retrieval. GPU training requires a separately installed PyTorch build compatible
with the chosen CUDA or ROCm system. The CPU examples below do not need a model
download or GPU.

## Synthetic lexical retrieval

Run:

```bash
python examples/retrieval_synthetic.py
```

The example creates a temporary SQLite FTS5 index from two generated passages,
calls the repository's `rag_pilot.system.search_unified()` lexical path, then
formats the retrieved evidence with `rag_pilot.generate.build_rag_prompt()`.
It asserts that the river passage ranks for a synthetic query and prints its
passage citation context. This exercises the included retrieval and citation
formatting implementation; the zero-filled vectors are unused in lexical mode.
It is not a dense-retrieval or corpus-quality evaluation.

## One CPU Transformer step

Run:

```bash
python examples/cpu_transformer_step.py
```

The example fits a tiny byte-level BPE tokenizer from generated English fixture
sentences, constructs the included `pretrain.model.AncientDecoder` with a small
configuration, shifts the token sequence into inputs and targets, and performs
one forward/backward/AdamW update on CPU. It prints the parameter count and loss.
This uses the same model forward/loss implementation as the trainer, with a
small architecture to keep the demo quick.

The full `pretrain.train` entry point streams approved corpus shards and validates
per language; it cannot run on synthetic text without adding a data-source
interface to that trainer. No such refactor is included. The direct model-step
demo exercises the genuine model implementation, but does not claim to validate
the full streaming trainer, checkpoint/resume behavior, or progressive-growth
schedule. Curriculum sampling is available only when an external JSON is
provided explicitly with `--sampling-mode curriculum --curriculum-config PATH`;
no curriculum schedule is included in this snapshot.

## Continuous integration

The `CPU fixtures` workflow runs catalog parser/classification fixtures and
synthetic retrieval, tokenizer, and small-model checks on CPU. It does not run
corpus-scale jobs, download pretrained weights, or run GPU benchmarks; hardware
benchmark results remain documented in [RESULTS.md](RESULTS.md).

Dense vector caching is optional and caller-owned. Reads inside an active
transaction bypass and clear the cache so rolled-back vectors cannot persist.
Call `VectorMatrixCache.clear()` when its owner is finished.
