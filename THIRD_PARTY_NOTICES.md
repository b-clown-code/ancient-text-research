# Third-party notices

This curated repository does not vendor third-party package source or model
weights. The following projects are used through documented interfaces or are
optional runtime dependencies; install and redistribute them under their own
license terms. Version-specific package wheels can include additional
third-party components and notices, so retain the notices shipped with any
binary package you redistribute.

| Component | Use in this repository | License / attribution |
| --- | --- | --- |
| NumPy | Dense retrieval and numerical helpers | BSD 3-Clause. [Project license](https://github.com/numpy/numpy/blob/main/LICENSE.txt). |
| PyTorch | Transformer model, training and tensor operations | Main project license is BSD 3-Clause. Binary distributions may contain additional separately licensed components; see their bundled license files. [Project license information](https://github.com/pytorch/pytorch/blob/main/pyproject.toml). |
| Hugging Face Tokenizers | BPE tokenizer implementation and interface | Apache License 2.0. [License](https://github.com/huggingface/tokenizers/blob/main/LICENSE). |
| Hugging Face Transformers | Optional local encoder/reranker model loading | Apache License 2.0. [License](https://github.com/huggingface/transformers/blob/main/LICENSE). |
| Sentence Transformers | Optional sentence-embedding model loading | Apache License 2.0. [Package metadata](https://github.com/huggingface/sentence-transformers/blob/main/pyproject.toml). |
| SQLite | SQLite database engine and FTS5 used through Python's `sqlite3` interface | SQLite states that its deliverable code is public domain. No SQLite implementation source is copied here. [SQLite copyright](https://www.sqlite.org/copyright.html). |
| GitHub Actions `checkout` and `setup-python` | CI workflow actions, referenced by version tag | MIT License. These actions are hosted dependencies, not vendored source. [checkout license](https://github.com/actions/checkout/blob/main/LICENSE), [setup-python license](https://github.com/actions/setup-python/blob/main/LICENSE). |

Python's standard-library modules are used but not copied into this repository.

## Research attribution

The progressive-depth experiments are conceptually related to progressive
layer stacking described by Karanpartap Singh, Neil Band, and Ehsan Adeli,
“Curriculum-Guided Layer Scaling for Language Model Pretraining,” arXiv
2506.11389 (2025), [paper](https://arxiv.org/abs/2506.11389). The progressive
growth implementation in this repository is original; no CGLS source code is
included.

The SCALE optimizer adaptation previously present in the working repository
is excluded from this curated release. The referenced
[OptimAI-Lab implementation](https://github.com/OptimAI-Lab/Minimalist_LLM_Pretraining/tree/main/mem_eff_pt/pt_scale)
does not publish an identifiable license in the repository, so this release
does not redistribute that implementation or its adaptation. The included
trainer supports AdamW.

## Artifact rights

These notices cover code dependencies only. They do not grant rights to any
corpus, source text, dataset, fitted tokenizer, model weights, embeddings,
benchmark material, or other third-party artifact. None of those artifacts is
included. Any future model or dataset release, including a Hugging Face
release, requires separate artifact-specific rights and licensing review.
