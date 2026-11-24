# Model card — Hams.AI bilingual RAG stack

Sizes marked "measured" were measured on the development machine; others are file sizes.
VRAM figures are for inference in fp16 (encoders) or Q4_K_M (LLM) and include activations for the
batch sizes used by the service.


**Fine-tuned models on Hugging Face**
* Embedding model: [AbdelrahmanHassan/hams-e5-base-ar-en-retrieval](https://huggingface.co/AbdelrahmanHassan/hams-e5-base-ar-en-retrieval), fine-tuned from [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
* Reranker: [AbdelrahmanHassan/hams-gte-reranker-ar-en](https://huggingface.co/AbdelrahmanHassan/hams-gte-reranker-ar-en), fine-tuned from [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
* Answer model (not fine-tuned): [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Q4_K_M GGUF from [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)

## Components

| Role | Model | Hugging Face | License | Commercial use | Params | Disk | Runs on CPU | Runs on GPU | Quantization |
|---|---|---|---|---|---|---|---|---|---|
| Embedder (fine-tuned) | AbdelrahmanHassan/hams-e5-base-ar-en-retrieval | https://huggingface.co/AbdelrahmanHassan/hams-e5-base-ar-en-retrieval | MIT (inherits base) | Yes | 278M | ≈1.1 GB (fp32) | Yes | Yes | fp16 at inference on GPU |
| Embedder base | intfloat/multilingual-e5-base | https://huggingface.co/intfloat/multilingual-e5-base | MIT | Yes | 278M | 1.1 GB | Yes | Yes | — |
| Reranker (fine-tuned) | AbdelrahmanHassan/hams-gte-reranker-ar-en (selected run `lr2e6_dn`) | https://huggingface.co/AbdelrahmanHassan/hams-gte-reranker-ar-en | Apache-2.0 (inherits base) | Yes | 306M | ≈0.6 GB (fp16) | Yes (slow) | Yes | fp16 at inference on GPU |
| Reranker base | Alibaba-NLP/gte-multilingual-reranker-base | https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base | Apache-2.0 | Yes | 306M | 0.6 GB (fp16) | Yes | Yes | — |
| Generator | Qwen3.5-2B (dev) / Qwen3.5-4B (L4 profile) | https://huggingface.co/Qwen/Qwen3.5-2B · GGUF: https://huggingface.co/unsloth/Qwen3.5-2B-GGUF | Apache-2.0 | Yes | ≈2B / ≈4B | ≈1.3 GB / 2.8 GB (Q4_K_M) | Yes (llama.cpp) | Yes | GGUF Q4_K_M weights, q8_0 KV cache |
| Keyword engine | SQLite FTS5 | — | Public domain | Yes | — | <5 MB index | Yes | — | — |
| Vector index | FAISS | https://github.com/facebookresearch/faiss | MIT | Yes | — | 0.3 MB (98×768 fp32) | Yes | — | none (exact) / HNSW above 50k vectors |
| Serving | llama.cpp `llama-server` | https://github.com/ggml-org/llama.cpp | MIT | Yes | — | — | Yes | Yes | — |
| Backup reranker (compared, not deployed) | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Apache-2.0 | Yes | 118M | | | | fine-tuned once as a control run (validation MRR@10 0.856 → 0.878); the GTE reranker was better |

Known limitations of the base models: Qwen3.5 small models are multilingual generalists — Arabic
fluency is good but not native-level, and the 2B model occasionally drifts into English for mixed
queries (mitigated by an explicit answer-language instruction, measured as language correctness).
The gte models require `trust_remote_code` (custom architecture code from `Alibaba-NLP/new-impl`).

## Fine-tuning: embedding model

| | |
|---|---|
| Base | intfloat/multilingual-e5-base (XLM-RoBERTa base architecture, mean pooling, `query: ` / `passage: ` prefixes) |
| Method | Full fine-tuning of all 12 transformer layers, layer norms and position/type embeddings; the 250,002 × 768 word-embedding matrix is **frozen** (stored in bf16 during training, saved as fp32). Not LoRA: at 86M trainable parameters full fine-tuning fits 4 GB and avoids adapter overhead at inference. |
| Why freeze word embeddings | They hold 192M of 278M parameters (69%). Freezing cuts optimizer + gradient memory from ≈4.5 GB to ≈1.4 GB and protects the multilingual vocabulary (incl. Arabic sub-words that are rare in our data) from drifting. |
| Objective | `CachedMultipleNegativesRankingLoss` (InfoNCE with GradCache): each query is scored against its positive, its mined hard negative, and every other positive/negative in a batch of 64; scale 20 (temperature 0.05). GradCache keeps memory at a micro-batch of 16 while preserving 64-way in-batch negatives — plain gradient accumulation would not add negatives. |
| Hard negatives | Top-30 base-model candidates for each training query, **training documents only**; candidates scoring > 0.95 × the positive are discarded as probable unlabelled positives; the highest remaining one is used. |
| Data | `data/train.jsonl`: authored queries on 31 training documents + reference answers reused as queries in the other language (cross-lingual augmentation) [+ filtered LLM-generated queries if `data/synthetic_train.jsonl` exists]. |
| Split | Document-level 31 / 5 / 10 (train / validation / test). |
| Hyper-parameters | 4 epochs, batch 64 (micro-batch 16), lr 2e-5 linear decay, warmup 10%, weight decay 0.01, max length 256, bf16 autocast, `NO_DUPLICATES` batch sampler, seed 42 |
| Result | Validation nDCG@10 0.660 → **0.890**, Recall@5 0.868 → 0.974; 9.7 min on the RTX 3050 Laptop GPU, peak 3.47 GiB |
| Model selection | Best epoch by validation nDCG@10 (`InformationRetrievalEvaluator` over the full corpus). |
| Reproducibility | `configs/training_config.yaml`; `models/e5-base-saraya-ft/training_report.json` stores config, data hash, before/after validation metrics, peak GPU memory and training time. |

## Fine-tuning: reranker

| | |
|---|---|
| Base | Alibaba-NLP/gte-multilingual-reranker-base (cross-encoder, 306M; native `gte` architecture in transformers 5) |
| Method | Full fine-tuning (fp32) with frozen word embeddings — 114M of 306M parameters trainable |
| Objective | Binary cross-entropy on (query, chunk) pairs, positive class weighted by the negative:positive ratio |
| Negatives | Candidates of the deployed first stage (BM25 + fine-tuned e5 → RRF) on **training documents only**; the top-ranked non-gold candidate is skipped and any candidate the *base* reranker scores within 1 logit of the positive is dropped as a probable false negative (RocketQA-style denoising), then up to 4 negatives per query |
| Hyper-parameters | 2 epochs, lr **2e-6**, warmup 10%, weight decay 0.01, batch 8 × grad-accumulation 4, max length 384, gradient clipping 1.0 |
| Selection | Validation MRR@10 evaluated every 20 steps on the real top-20 candidates; best checkpoint kept; best run across the sweep chosen by `scripts/select_reranker.py` |
| Result | Validation MRR@10 0.878 → **0.939**; selected checkpoint `models/gte-reranker-saraya-ft_lr2e6_dn` |

### What went wrong first, and how it was fixed (documented for transparency)
The first run (lr 2e-5, bf16, 7 un-filtered hard negatives) **collapsed**: validation MRR@10 fell from
0.878 to 0.15 and the training loss plateaued at the value of a constant predictor. Investigation
(`tools/diag_reranker.py`, `results/diag_reranker.json`):
1. The forward pass is correct — identical logits alone vs. in padded batches, in train vs. eval mode,
   sdpa vs. eager attention; bf16 deviates by < 0.05 logits.
2. Gradients are correct — tiny models of the same architecture overfit a toy batch normally.
3. The same data fine-tunes the XLM-R based mMiniLM cross-encoder stably at lr 2e-5
   (0.856 → 0.878), so the data and loss are usable.
4. A learning-rate sweep on GTE showed extreme sensitivity: 1e-5 collapses, 5e-6 degrades
   (0.878 → 0.855), 2e-6 improves (0.878 → 0.923; with denoised negatives 0.939).

Conclusion: this already-converged reranker must be fine-tuned with a very small learning rate and
step-wise validation; cleaner negatives help further. Validation MRR moves in steps of ≈0.026 per
query (38 queries), so checkpoint selection is noisy — the held-out test set is the arbiter
(see README results).

| Run | lr | precision | negatives | val MRR@10 (base → best) |
|---|---|---|---|---|
| first attempt | 2e-5 | bf16 | 7, unfiltered | 0.878 → 0.149 (collapsed) |
| sweep A | 5e-6 | fp32 | 7, unfiltered | 0.878 → 0.855 |
| sweep B | 5e-6 | bf16 | 7, unfiltered | 0.878 → 0.868 |
| sweep C | 2e-6 | fp32 | 7, unfiltered | 0.878 → 0.923 |
| denoised | 1e-5 | bf16 | 4, denoised | 0.878 → 0.164 (collapsed) |
| **selected** | **2e-6** | **fp32** | **4, denoised** | **0.878 → 0.939** |
| mMiniLM (backup arch.) | 2e-5 | bf16 | 4, denoised | 0.856 → 0.878 |

## Held-out test results (10 unseen documents, 138 questions, RTX 3050 Laptop)

Same index, fusion settings (RRF k=5, dense weight 5), LLM and prompts in both columns; only the two
fine-tuned models differ. Full tables: README §7 and `results/`.

| Metric (hybrid + rerank) | Base models | Fine-tuned |
|---|---|---|
| Recall@1 / Recall@5 (all) | 0.717 / 0.848 | **0.783 / 0.920** |
| MRR / nDCG@10 (all) | 0.786 / 0.810 | **0.849 / 0.873** |
| Recall@5 cross-lingual (53 q) | 0.679 | **0.906** |
| Recall@5 Arabic (29 q) | **0.862** | 0.828 |
| Recall@5 English (24 q) / mixed (32 q) | 1.000 / 1.000 | 1.000 / 0.969 |
| False "not found" on answerable | 0.246 | **0.159** |
| Correct "not found" on unanswerable (26 q) | 0.885 | **0.923** |
| Hallucination rate (answerable) | 0.115 | **0.069** |
| Numeric faithfulness | 0.947 | **0.966** |

The abstention gate threshold is calibrated separately for each reranker on validation
(`scripts/calibrate_gate.py`: base 0.368, fine-tuned 0.097 on the sigmoid score), so each column uses its
own best gate.

## Intended use and limitations

Intended for retrieval over Arabic (MSA) and English company documents of the kind in this knowledge
bank: policies, product catalogues, manuals, contracts. Trained on a small synthetic corpus (31 training
documents, 475 training rows). Expect smaller gains on very different domains, and re-check Arabic
in-language retrieval, which did not improve here. Dialect coverage is minimal. The models rank text;
they do not verify facts.

Not done in this submission: a zero-shot comparison against multilingual-e5-small, gte-multilingual-base,
bge-m3 and bge-reranker-v2-m3 (`tools/run_A2_bakeoff.bat` exists but was not run). The model choices are
justified by licence, size and 4 GB trainability (README §1), not by that comparison.

## Text normalisation used in training and inference
Arabic: NFKC, tashkeel/tatweel removal, Eastern → Western digits (neural models); additionally alef / ya /
ta-marbuta folding, script-boundary splitting and light stemming for BM25. English: lower-casing and
Snowball stemming for BM25 only.

## Latency optimisations after fine-tuning
fp16 encoder/reranker inference; single GPU lock to avoid contention on 4 GB; llama.cpp with all
layers on GPU, flash attention, q8_0 KV cache, prompt-prefix cache (static system prompt), 2 parallel
slots; retrieval-score abstention gate that skips the LLM for unanswerable questions; follow-up
suggestions generated in the same LLM call; query rewriting only when the follow-up needs it, with a
rewrite cache. Measured latency is in `results/latency_after.json` and `results/concurrency_report.json`.
