# Reviewed aggregate results

These are historical pilot measurements, not general claims. The reports and
status register qualify their evidence. This repository excludes source text,
benchmark queries and labels, fitted tokenizers, databases, embeddings, and
checkpoints; the full experiments cannot be rerun from this tree alone.

## Retrieval pilot

The reported evaluation used a 35,000-passage pilot index and 150 queries: 120
answerable queries spanning thematic paraphrase, metadata constraints,
cross-lingual concepts, and technical topics, plus 30 hard-negative controls.
The stored relevance labels were constructed using retrieval rank, system
agreement, and top-result fallback (3,246 positive judgments); they were not
independently adjudicated. Treat these values as diagnostics against that
constructed label set, not as independent evidence of semantic relevance.
Recall@10 and MRR@10 cover the 120 answerable queries; non-empty rate covers the
30 unanswerable controls. Latencies are the historical report's end-to-end
measurements, including query tokenization/inference, lexical search, fusion,
and graph expansion; they are not a fresh benchmark of this curated snapshot.

| Retrieval mode | Recall@10 | MRR@10 | Non-empty rate on unanswerable queries | Mean latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical (BM25 / FTS5) | 0.3065 | 0.7500 | 0.4667 | 7.07 ms | 32.26 ms |
| Dense (MiniLM-L12) | 0.3330 | 0.8296 | 1.0000 | 12.18 ms | 72.67 ms |
| Hybrid (reciprocal-rank fusion, k=60) | 0.4567 | 0.9750 | 1.0000 | 12.28 ms | 49.39 ms |
| Hybrid plus graph proximity (weight 0.1) | 0.4441 | 0.8681 | 1.0000 | 22.22 ms | 70.05 ms |

The hybrid row had the highest reported Recall@10 and MRR@10 on this constructed
benchmark. Adding graph proximity raised the reported metrics from 0.2962 / 0.7003
for the corresponding hybrid-graph baseline to 0.4441 / 0.8681, but remained
below hybrid without graph boosting. The graph represents passage/work/document
membership, not validated semantic relations. A non-empty result for an
unanswerable query is not itself a false claim, and these rates do not measure
refusal quality, citation correctness, or entailment. Current project review
also flags held-out exposure and label-construction limitations, so the
historical benchmark does not establish independent application quality.

## Tokenizer and training

Gate 1 compared byte-level BPE vocabularies using the same deterministic,
train-only sample: 182,403 fitting records, 20,026 tokenizer-only evaluation
records (which remain in the original training split), and up to 500,000 sampled
words per language. All three candidates round-tripped the multilingual
evaluation text without normalization.

| Vocabulary | Language-balanced tokens/character | Tied embedding parameters at width 576 | BF16 embedding table |
| ---: | ---: | ---: | ---: |
| 16,384 | 0.4218 | 9.44M | 18 MiB |
| 32,768 (selected) | 0.3922 | 18.87M | 36 MiB |
| 49,152 | 0.3783 | 28.31M | 54 MiB |

The 32K candidate's balanced token-per-character score was 3.7% worse than the
49K candidate, within the 5% selection tolerance, while saving 9.44M tied
embedding parameters and 18 MiB of BF16 embedding weights. The 16K candidate
scored 11.5% worse than the best candidate. This comparison does not imply
equal quality for every language/script: language-level scores vary, some
groups have smaller evaluation samples, and the tokenizer-only sample remains
inside the original training split. The full corpus was not tokenized for this
gate, and Gate 1 itself did not train a model.

Gate 5 trained a 135.03M-parameter model with a frozen 32K tokenizer, BF16,
1,024-token sequences, AdamW, microbatch 1, gradient accumulation 8, and a
balanced random sampler on a Radeon AI PRO R9700 using ROCm. The six-language
training validation set used 32 fixed batches per language in Gate 5. Progressive
12→20→30-layer growth used identity-initialized inserted blocks.

| Comparison | Tokens | Macro validation CE | Estimated FLOPs | Trainer time |
| --- | ---: | ---: | ---: | ---: |
| Fixed depth, seed 20261103 (Gate 4) | 20.005M | 6.1578 | 18.332 PF | 2,767.0 s |
| Progressive, seed 20261103 (Gate 4) | 20.005M | 6.1100 | 14.796 PF | 2,190.9 s |
| Fixed depth, seed 20261104 | 20.005M | 6.1365 | 18.332 PF | 2,771.7 s |
| Progressive, seed 20261104 | 20.005M | 5.8921 | 14.796 PF | 2,220.5 s |
| Fixed depth, seed 20261104 | 100.008M | 6.0560 | 91.643 PF | 13,597.3 s |
| Progressive, seed 20261104 | 100.008M | 5.3510 | 88.107 PF | 13,133.5 s |
| Progressive, seed 20261104, equal-FLOP continuation | 103.866M | 5.3294 | 91.643 PF | 13,658.1 s |

Progressive growth used 19.3% fewer estimated FLOPs than fixed depth at 20M
tokens in both seeds; at 100M it used 3.86% fewer estimated FLOPs and 3.41%
less trainer time. At equal estimated compute it processed 3.86% more tokens.
FLOPs are model-based estimates, not hardware-counter measurements. The 100M
comparison has only one seed; its validation uses a small fixed sample. Fixed
depth's Egyptian validation CE rose from 3.9277 at 20M to 7.7613 at 100M, an
unexplained regression. Progressive growth scored lower than fixed depth across
the six languages at the 100M equal-token and equal-FLOP endpoints in this run,
but this does not establish a general advantage. Keep fixed-depth AdamW as the
matched control in the next bounded pilot.

## CUDA and ROCm microbatch comparison

The matched benchmark used the same 135M model, 1,024-token sequences, BF16,
AdamW, initialization, tokenizer, and ordered batches. Each configuration had
five warmup updates followed by 100 measured optimizer updates (819,200 measured
tokens); CUDA and ROCm used different PyTorch builds.

| GPU / backend | Best tested microbatch / accumulation | Throughput | Peak allocated / reserved VRAM |
| --- | ---: | ---: | ---: |
| RTX 4060 Ti / CUDA | 4 / 2 | 15,237 tokens/s | 2.70 / 2.94 GiB |
| Radeon AI PRO R9700 / ROCm | 8 / 1 | 24,548 tokens/s | 4.23 / 4.53 GiB |

At the matched 4 / 2 setting, the R9700 reached 22,993 tokens/s, 50.9% above
the 4060 Ti. The best tested microbatch was backend-specific. This is a result
for these devices, software stacks, and benchmark settings; vendor GPU-busy
sampling differs and should be treated as directional context.
