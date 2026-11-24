# Dataset card — Saraya Living bilingual knowledge bank

## Summary
| Item | Value |
|---|---|
| Documents | 46 (20 Arabic, 17 English, 9 mixed Arabic/English) |
| Formats | 11 PDF, 6 DOCX, 9 HTML, 20 Markdown |
| Chunks | 98 (≈120 words max, heading-aware) |
| Labelled facts | 180 (each: evidence phrase, Arabic + English reference answer) |
| Authored queries | 441 (Arabic MSA, English, code-switched Arabic/English, Saudi dialect) |
| Unanswerable questions | 40 |
| Conversations | 15 × 3 turns |
| License | CC BY 4.0 (synthetic content created for this project) |

## Sources and generation
* **Content is synthetic.** "Saraya Living" and "Saraya Business" are fictional; any resemblance to
  real companies is unintended. Documents were drafted with the help of an AI assistant (Anthropic's
  Claude, used as an authoring tool only — it is not part of the runtime system) and reviewed for
  internal consistency (prices, dates and policies agree across documents; each fact lives in one
  document unless explicitly labelled with alternative evidence).
* Documents are authored in Markdown (`data/corpus/src`) and rendered to their delivery format
  (`data/corpus/kb`) with `hamsrag/corpus_build.py` (python-docx; PDFs via LibreOffice from DOCX, which
  shapes Arabic correctly). Ingestion reads the rendered files, not the sources.
* Queries and reference answers (`data/qa_src`) were written per fact, in four styles. Long,
  multi-clause questions and number/date/price questions are included on purpose.
* Optional: `scripts/generate_synthetic_queries.py` adds LLM-generated queries for **training
  documents only**, produced by the local Qwen model and filtered (language check, copy check,
  de-duplication, round-trip retrieval consistency). Rejections are logged. Not used for the reported models.
* Public datasets (MIRACL, TyDi, mMARCO) were considered; the available Hugging Face versions are
  script-based (no longer loadable with `datasets>=4`) or lack a license tag, so they were not used.

## Labels
Each fact has an evidence phrase; after ingestion the gold chunk is the chunk whose section + text
contains it (`hamsrag/qa_data.py`). Labels therefore survive re-chunking and verify that PDF/DOCX/HTML
extraction preserved the text. Two facts are stated in two documents and carry both as gold.

## Splits
Document-level, stratified by document language (seed 13): 31 train, 5 validation, 10 test documents.
Leakage guards in `prepare_data.py`: no shared documents and no identical (normalised) query text
across splits; hard negatives are mined from training documents only.

Evaluation groups per query: `ar` (Arabic query, Arabic doc), `en`, `mixed` (code-switched query or
mixed-language doc), `cross` (Arabic query → English doc or English → Arabic), plus `dialect` as an
extra slice.

## Known limitations
* Small: 46 documents / 98 chunks; metrics on 138 test queries have wide confidence intervals.
* Synthetic text is cleaner and more uniform than real enterprise documents.
* Conversations reference facts in training documents too (they evaluate the interactive pipeline,
  not unseen-document retrieval).
* Dialect coverage: Saudi only, and only in queries.
