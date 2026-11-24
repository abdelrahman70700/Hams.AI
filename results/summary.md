### Before vs after fine-tuning (held-out test set, hybrid + rerank)

Before = base e5-base + base gte reranker; after = fine-tuned e5-base + fine-tuned reranker. Same LLM, prompts, index and data in both columns.

| Metric | Before | After |
|---|---|---|
| Recall@5 (AR) | 0.862 | 0.828 |
| Recall@5 (EN) | 1.000 | 1.000 |
| Recall@5 (mixed) | 1.000 | 0.969 |
| Recall@5 (cross-lingual) | 0.679 | 0.906 |
| MRR (AR) | 0.849 | 0.833 |
| MRR (EN) | 0.958 | 0.958 |
| nDCG@10 (AR) | 0.884 | 0.861 |
| nDCG@10 (EN) | 0.969 | 0.969 |
| nDCG@10 (all) | 0.810 | 0.873 |
| nDCG@10 (cross-lingual) | 0.628 | 0.793 |
| Faithfulness (numeric) | 0.958 | 0.977 |
| Faithfulness (lexical support) | 0.785 | 0.802 |
| Answer relevance (sim. to reference) | 0.930 | 0.929 |
| Hallucination rate (answerable) | 0.090 | 0.061 |
| Hallucination rate (unanswerable) | 0.077 | 0.115 |
| Correct "not found" on unanswerable | 0.923 | 0.885 |
| False "not found" on answerable | 0.275 | 0.167 |
| Citation precision | 0.900 | 0.900 |
| Local-LLM judge: faithful | 0.720 | 0.739 |
| Answer language correct | 1.000 | 1.000 |
| Human meaning score (1–5) | — | — |
| Human fluency AR (1–5) | — | — |
| Human fluency EN (1–5) | — | — |
| Time to first token (Smart AI, avg) | 456 ms | 396 ms |
| Total latency (Smart AI, avg) | 959 ms | 869 ms |
| p95 latency (Smart AI) | 1,442 ms | 1,234 ms |
| Quick Search p95 | 1 ms | 1 ms |
| GPU memory (all processes, MiB) | 3869 | 3907 |
| System RAM in use when measured (GiB of 16) | 13.160 | 12.720 |
| CPU memory (API process RSS, GiB) | 0.290 | 0.140 |

### Retrieval — after fine-tuning (`e5-base-ft` + `reranker-ft`)

| Mode | Group | n | R@1 | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | all | 138 | 0.435 | 0.573 | 0.609 | 0.630 | 0.573 | 0.498 | 0.523 |
| bm25 | ar | 29 | 0.690 | 0.931 | 1.000 | 1.000 | 0.931 | 0.805 | 0.853 |
| bm25 | en | 24 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 0.954 |
| bm25 | mixed | 32 | 0.594 | 0.812 | 0.844 | 0.938 | 0.812 | 0.691 | 0.723 |
| bm25 | cross | 53 | 0.000 | 0.038 | 0.075 | 0.075 | 0.038 | 0.014 | 0.028 |
| bm25 | cross ar->en | 26 | 0.000 | 0.077 | 0.154 | 0.154 | 0.077 | 0.028 | 0.057 |
| bm25 | cross en->ar | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bm25 | dialect | 7 | 0.286 | 0.429 | 0.714 | 0.714 | 0.429 | 0.401 | 0.474 |
| dense | all | 138 | 0.638 | 0.927 | 0.971 | 0.993 | 0.927 | 0.769 | 0.818 |
| dense | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.728 | 0.793 |
| dense | en | 24 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.918 |
| dense | mixed | 32 | 0.750 | 0.938 | 0.969 | 1.000 | 0.938 | 0.841 | 0.871 |
| dense | cross | 53 | 0.528 | 0.906 | 0.943 | 0.981 | 0.906 | 0.694 | 0.753 |
| dense | cross ar->en | 26 | 0.462 | 0.885 | 0.885 | 0.962 | 0.885 | 0.635 | 0.692 |
| dense | cross en->ar | 27 | 0.593 | 0.926 | 1.000 | 1.000 | 0.926 | 0.750 | 0.812 |
| dense | dialect | 7 | 0.286 | 0.857 | 1.000 | 1.000 | 0.857 | 0.587 | 0.689 |
| hybrid_rrf | all | 138 | 0.638 | 0.927 | 0.971 | 0.993 | 0.927 | 0.768 | 0.817 |
| hybrid_rrf | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.736 | 0.801 |
| hybrid_rrf | en | 24 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.918 |
| hybrid_rrf | mixed | 32 | 0.812 | 0.969 | 0.969 | 1.000 | 0.969 | 0.878 | 0.899 |
| hybrid_rrf | cross | 53 | 0.491 | 0.887 | 0.943 | 0.981 | 0.887 | 0.665 | 0.731 |
| hybrid_rrf | cross ar->en | 26 | 0.423 | 0.846 | 0.885 | 0.962 | 0.846 | 0.595 | 0.661 |
| hybrid_rrf | cross en->ar | 27 | 0.556 | 0.926 | 1.000 | 1.000 | 0.926 | 0.732 | 0.798 |
| hybrid_rrf | dialect | 7 | 0.286 | 0.857 | 1.000 | 1.000 | 0.857 | 0.589 | 0.691 |
| hybrid_union | all | 138 | 0.638 | 0.927 | 0.971 | 0.986 | 0.927 | 0.768 | 0.817 |
| hybrid_union | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.736 | 0.801 |
| hybrid_union | en | 24 | 0.792 | 1.000 | 1.000 | 1.000 | 1.000 | 0.889 | 0.918 |
| hybrid_union | mixed | 32 | 0.812 | 0.969 | 0.969 | 0.969 | 0.969 | 0.877 | 0.899 |
| hybrid_union | cross | 53 | 0.491 | 0.887 | 0.943 | 0.981 | 0.887 | 0.664 | 0.731 |
| hybrid_union | cross ar->en | 26 | 0.423 | 0.846 | 0.885 | 0.962 | 0.846 | 0.595 | 0.661 |
| hybrid_union | cross en->ar | 27 | 0.556 | 0.926 | 1.000 | 1.000 | 0.926 | 0.732 | 0.798 |
| hybrid_union | dialect | 7 | 0.286 | 0.857 | 1.000 | 1.000 | 0.857 | 0.589 | 0.691 |
| hybrid+rerank | all | 138 | 0.783 | 0.920 | 0.957 | 0.993 | 0.920 | 0.849 | 0.873 |
| hybrid+rerank | ar | 29 | 0.793 | 0.828 | 0.966 | 1.000 | 0.828 | 0.833 | 0.861 |
| hybrid+rerank | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid+rerank | mixed | 32 | 0.875 | 0.969 | 1.000 | 1.000 | 0.969 | 0.927 | 0.945 |
| hybrid+rerank | cross | 53 | 0.660 | 0.906 | 0.906 | 0.981 | 0.906 | 0.761 | 0.793 |
| hybrid+rerank | cross ar->en | 26 | 0.577 | 0.846 | 0.846 | 0.962 | 0.846 | 0.691 | 0.724 |
| hybrid+rerank | cross en->ar | 27 | 0.741 | 0.963 | 0.963 | 1.000 | 0.963 | 0.828 | 0.860 |
| hybrid+rerank | dialect | 7 | 0.571 | 0.714 | 1.000 | 1.000 | 0.714 | 0.659 | 0.737 |
| interactive(gold-rewrite) | all | 44 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 | 0.862 | 0.897 |

### Retrieval — before fine-tuning (`e5-base` + `gte-rerank`)

| Mode | Group | n | R@1 | R@5 | R@10 | R@20 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | all | 138 | 0.435 | 0.573 | 0.609 | 0.630 | 0.573 | 0.498 | 0.523 |
| bm25 | ar | 29 | 0.690 | 0.931 | 1.000 | 1.000 | 0.931 | 0.805 | 0.853 |
| bm25 | en | 24 | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 0.954 |
| bm25 | mixed | 32 | 0.594 | 0.812 | 0.844 | 0.938 | 0.812 | 0.691 | 0.723 |
| bm25 | cross | 53 | 0.000 | 0.038 | 0.075 | 0.075 | 0.038 | 0.014 | 0.028 |
| bm25 | cross ar->en | 26 | 0.000 | 0.077 | 0.154 | 0.154 | 0.077 | 0.028 | 0.057 |
| bm25 | cross en->ar | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| bm25 | dialect | 7 | 0.286 | 0.429 | 0.714 | 0.714 | 0.429 | 0.401 | 0.474 |
| dense | all | 138 | 0.485 | 0.725 | 0.833 | 0.891 | 0.725 | 0.600 | 0.651 |
| dense | ar | 29 | 0.586 | 0.862 | 1.000 | 1.000 | 0.862 | 0.732 | 0.797 |
| dense | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| dense | mixed | 32 | 0.656 | 0.875 | 0.875 | 0.938 | 0.875 | 0.745 | 0.772 |
| dense | cross | 53 | 0.132 | 0.434 | 0.641 | 0.755 | 0.434 | 0.278 | 0.353 |
| dense | cross ar->en | 26 | 0.077 | 0.269 | 0.500 | 0.577 | 0.269 | 0.186 | 0.246 |
| dense | cross en->ar | 27 | 0.185 | 0.593 | 0.778 | 0.926 | 0.593 | 0.366 | 0.455 |
| dense | dialect | 7 | 0.286 | 0.571 | 0.857 | 0.857 | 0.571 | 0.426 | 0.524 |
| hybrid_rrf | all | 138 | 0.471 | 0.717 | 0.826 | 0.891 | 0.717 | 0.587 | 0.638 |
| hybrid_rrf | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.735 | 0.800 |
| hybrid_rrf | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid_rrf | mixed | 32 | 0.625 | 0.875 | 0.938 | 1.000 | 0.875 | 0.734 | 0.780 |
| hybrid_rrf | cross | 53 | 0.113 | 0.396 | 0.585 | 0.717 | 0.396 | 0.249 | 0.314 |
| hybrid_rrf | cross ar->en | 26 | 0.077 | 0.231 | 0.500 | 0.577 | 0.231 | 0.173 | 0.237 |
| hybrid_rrf | cross en->ar | 27 | 0.148 | 0.556 | 0.667 | 0.852 | 0.556 | 0.321 | 0.389 |
| hybrid_rrf | dialect | 7 | 0.286 | 0.714 | 0.857 | 0.857 | 0.714 | 0.431 | 0.529 |
| hybrid_union | all | 138 | 0.471 | 0.717 | 0.826 | 0.891 | 0.717 | 0.587 | 0.638 |
| hybrid_union | ar | 29 | 0.586 | 0.897 | 1.000 | 1.000 | 0.897 | 0.735 | 0.800 |
| hybrid_union | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid_union | mixed | 32 | 0.625 | 0.875 | 0.938 | 1.000 | 0.875 | 0.734 | 0.780 |
| hybrid_union | cross | 53 | 0.113 | 0.396 | 0.585 | 0.717 | 0.396 | 0.249 | 0.315 |
| hybrid_union | cross ar->en | 26 | 0.077 | 0.231 | 0.500 | 0.577 | 0.231 | 0.174 | 0.237 |
| hybrid_union | cross en->ar | 27 | 0.148 | 0.556 | 0.667 | 0.852 | 0.556 | 0.322 | 0.390 |
| hybrid_union | dialect | 7 | 0.286 | 0.714 | 0.857 | 0.857 | 0.714 | 0.431 | 0.529 |
| hybrid+rerank | all | 138 | 0.717 | 0.848 | 0.891 | 0.891 | 0.848 | 0.786 | 0.810 |
| hybrid+rerank | ar | 29 | 0.793 | 0.862 | 1.000 | 1.000 | 0.862 | 0.849 | 0.884 |
| hybrid+rerank | en | 24 | 0.917 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.969 |
| hybrid+rerank | mixed | 32 | 0.812 | 1.000 | 1.000 | 1.000 | 1.000 | 0.896 | 0.923 |
| hybrid+rerank | cross | 53 | 0.528 | 0.679 | 0.717 | 0.717 | 0.679 | 0.606 | 0.628 |
| hybrid+rerank | cross ar->en | 26 | 0.423 | 0.577 | 0.577 | 0.577 | 0.577 | 0.509 | 0.520 |
| hybrid+rerank | cross en->ar | 27 | 0.630 | 0.778 | 0.852 | 0.852 | 0.778 | 0.700 | 0.732 |
| hybrid+rerank | dialect | 7 | 0.571 | 0.571 | 0.857 | 0.857 | 0.571 | 0.623 | 0.673 |
| interactive(gold-rewrite) | all | 44 | 0.727 | 0.955 | 0.977 | 0.977 | 0.955 | 0.838 | 0.873 |

### Generation — after fine-tuning

| Set | n | language_correctness | abstain_rate | hallucination_rate | context_recall | context_precision | citation_precision | citation_recall | numeric_faithfulness | lexical_support | answer_relevance | judge_faithful | bleu | chrf | rouge_l |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 138 | 1.000 | 0.167 | 0.061 | 0.884 | 0.221 | 0.900 | 0.913 | 0.977 | 0.802 | 0.929 | 0.739 | 17.750 | 42.860 | 0.369 |
| test:ar | 29 | 1.000 | 0.138 | 0.000 | 0.828 | 0.207 | 0.880 | 0.880 | 1.000 | 0.801 | 0.944 | 0.720 | 23.490 | 48.870 | 0.442 |
| test:cross | 53 | 1.000 | 0.264 | 0.103 | 0.830 | 0.207 | 0.808 | 0.821 | 0.963 | 0.594 | 0.912 | 0.590 | 7.310 | 33.390 | 0.269 |
| test:mixed | 32 | 1.000 | 0.125 | 0.036 | 0.938 | 0.234 | 0.982 | 1.000 | 0.982 | 0.830 | 0.932 | 0.857 | 20.010 | 42.730 | 0.368 |
| test:en | 24 | 1.000 | 0.042 | 0.087 | 1.000 | 0.250 | 0.978 | 1.000 | 0.972 | 0.830 | 0.935 | 0.870 | 27.090 | 51.780 | 0.461 |
| unanswerable | 26 | 1.000 | 0.885 | 0.115 | — | — | — | — | — | — | — | — | — | — | — |
| conversation | 45 | 1.000 | 0.178 | 0.158 | 0.795 | 0.199 | 0.649 | 0.811 | 0.974 | 0.674 | — | 0.486 | — | — | — |

### Generation — before fine-tuning

| Set | n | language_correctness | abstain_rate | hallucination_rate | context_recall | context_precision | citation_precision | citation_recall | numeric_faithfulness | lexical_support | answer_relevance | judge_faithful | bleu | chrf | rouge_l |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 138 | 1.000 | 0.275 | 0.090 | 0.812 | 0.203 | 0.900 | 0.910 | 0.958 | 0.785 | 0.930 | 0.720 | 19.200 | 44.150 | 0.388 |
| test:ar | 29 | 1.000 | 0.379 | 0.056 | 0.828 | 0.207 | 0.944 | 0.944 | 1.000 | 0.768 | 0.947 | 0.833 | 30.180 | 53.810 | 0.490 |
| test:cross | 53 | 1.000 | 0.415 | 0.097 | 0.641 | 0.160 | 0.758 | 0.774 | 0.957 | 0.636 | 0.920 | 0.581 | 9.750 | 36.510 | 0.314 |
| test:mixed | 32 | 1.000 | 0.094 | 0.103 | 0.938 | 0.234 | 0.948 | 0.966 | 0.921 | 0.810 | 0.926 | 0.690 | 16.350 | 39.990 | 0.339 |
| test:en | 24 | 1.000 | 0.083 | 0.091 | 1.000 | 0.250 | 1.000 | 1.000 | 0.972 | 0.822 | 0.936 | 0.864 | 28.970 | 52.080 | 0.471 |
| unanswerable | 26 | 1.000 | 0.923 | 0.077 | — | — | — | — | — | — | — | — | — | — | — |
| conversation | 45 | 1.000 | 0.222 | 0.111 | 0.750 | 0.188 | 0.664 | 0.800 | 0.976 | 0.732 | — | 0.457 | — | — | — |

### Latency (sequential, NVIDIA GeForce RTX 3050 Laptop GPU, 16 CPU threads)

| Mode | Stage | n | avg | p50 | p95 | max |
|---|---|---|---|---|---|---|
| quick_search | total | 60 | 1 ms | 1 ms | 1 ms | 1 ms |
| quick_search | search | 60 | 0 ms | 0 ms | 0 ms | 1 ms |
| quick_search | preprocessing | 60 | 0 ms | 0 ms | 0 ms | 0 ms |
| retrieval_only(hybrid+rerank) | retrieval_stage | 60 | 14 ms | 14 ms | 17 ms | 18 ms |
| retrieval_only(hybrid+rerank) | rerank | 60 | 143 ms | 149 ms | 158 ms | 162 ms |
| retrieval_only(hybrid+rerank) | first_result | 60 | 14 ms | 14 ms | 17 ms | 18 ms |
| smart_ai_search | retrieval | 60 | 17 ms | 16 ms | 23 ms | 26 ms |
| smart_ai_search | rerank | 60 | 148 ms | 151 ms | 168 ms | 182 ms |
| smart_ai_search | time_to_first_token | 60 | 396 ms | 378 ms | 569 ms | 628 ms |
| smart_ai_search | generation_total | 60 | 704 ms | 691 ms | 1,065 ms | 1,473 ms |
| smart_ai_search | total | 60 | 869 ms | 840 ms | 1,234 ms | 1,644 ms |
| interactive | query_rewrite | 45 | 150 ms | 196 ms | 372 ms | 386 ms |
| interactive | time_to_first_token | 45 | 539 ms | 515 ms | 825 ms | 930 ms |
| interactive | total | 45 | 1,383 ms | 1,331 ms | 2,197 ms | 4,325 ms |
| interactive | query_rewrite_when_triggered | 29 | 233 ms | 207 ms | 380 ms | 386 ms |

**Latency vs corpus size** (p95 per query)

| Chunks | BM25 | FAISS flat | FAISS HNSW |
|---|---|---|---|
| 1000 | 1 ms | 0 ms | 0 ms |
| 10000 | 9 ms | 3 ms | 0 ms |
| 100000 | 75 ms | 22 ms | 1 ms |

**Rerank top-k** (latency and quality)

| rerank_k | avg | p95 | R@5 | MRR |
|---|---|---|---|---|
| 5 | 43 ms | 50 ms | 0.950 | 0.876 |
| 10 | 77 ms | 88 ms | 0.950 | 0.870 |
| 20 | 142 ms | 160 ms | 0.933 | 0.863 |
| 30 | 194 ms | 210 ms | 0.917 | 0.852 |
| 50 | 326 ms | 353 ms | 0.917 | 0.850 |

**Context size vs time to first token**

| context chunks | prompt tokens | TTFT avg | TTFT p95 |
|---|---|---|---|
| 2 | 665.6 | 342 ms | 413 ms |
| 4 | 981.7 | 395 ms | 484 ms |
| 6 | 1241.4 | 436 ms | 475 ms |

**Longest input tested:** 164 words (≈1032 prompt tokens incl. context), total 1,486 ms, answered: True.

### Concurrency (through the HTTP API)

| Mode | Users | Requests | avg | p95 | QPS | Failures | GPU MiB (max) | RAM GiB (max) |
|---|---|---|---|---|---|---|---|---|
| quick_search | 1 | 64 | 4 ms | 5 ms | 250.29 | 0.000 | 3820.000 | 15.240 |
| quick_search | 2 | 64 | 4 ms | 5 ms | 492.27 | 0.000 | 3820.000 | 15.250 |
| quick_search | 4 | 64 | 9 ms | 13 ms | 404.33 | 0.000 | 3820.000 | 15.250 |
| quick_search | 8 | 64 | 28 ms | 60 ms | 264.26 | 0.000 | 3820.000 | 15.260 |
| quick_search | 16 | 128 | 64 ms | 144 ms | 231.29 | 0.000 | 3820.000 | 15.260 |
| smart_ai_search | 1 | 16 | 2,227 ms | 2,568 ms | 0.45 | 0.000 | 3843.000 | 15.620 |
| smart_ai_search | 2 | 16 | 2,506 ms | 3,966 ms | 0.76 | 0.000 | 3851.000 | 15.570 |
| smart_ai_search | 4 | 16 | 3,576 ms | 4,478 ms | 1.06 | 0.000 | 3876.000 | 15.540 |
| smart_ai_search | 8 | 16 | 5,528 ms | 8,344 ms | 1.33 | 0.000 | 3912.000 | 15.570 |
| smart_ai_search | 16 | 32 | 9,364 ms | 12,827 ms | 1.43 | 0.000 | 3967.000 | 15.720 |
| interactive | 1 | 16 | 2,502 ms | 2,901 ms | 0.4 | 0.000 | 3967.000 | 15.720 |
| interactive | 2 | 16 | 3,967 ms | 6,033 ms | 0.5 | 0.000 | 3969.000 | 15.710 |
| interactive | 4 | 16 | 5,582 ms | 8,574 ms | 0.67 | 0.000 | 3969.000 | 15.720 |
| interactive | 8 | 16 | 8,672 ms | 12,258 ms | 0.74 | 0.000 | 3969.000 | 14.810 |
| interactive | 16 | 32 | 17,808 ms | 31,170 ms | 0.68 | 0.000 | 3968.000 | 15.130 |
