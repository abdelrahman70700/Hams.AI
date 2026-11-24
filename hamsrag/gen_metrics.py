"""Deterministic generation metrics (no closed-source judge needed).

* numeric faithfulness: every number in the answer appears in the cited (or, failing that, the
  retrieved) chunks - catches the most damaging hallucinations (prices, dates, durations).
* lexical support: share of the answer's content tokens (analysed: normalised + stemmed) that
  occur in the cited chunks - a cheap proxy for "is the answer grounded".
* language correctness: detected answer language == expected answer language.
* BLEU / chrF (sacrebleu) and ROUGE-L with an Arabic-aware tokenizer (rouge-score's default
  tokenizer drops all non-Latin characters, which would score every Arabic answer as 0).
"""
from __future__ import annotations

import re

from .text import analyze, clean_text, detect_language, tokenize

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    t = clean_text(text)
    t = re.sub(r"\[\d+\]", " ", t)  # drop citation markers
    return {n.replace(",", "") for n in _NUM.findall(t)}


def numeric_faithfulness(answer: str, support_texts: list[str]) -> float | None:
    nums = numbers(answer)
    if not nums:
        return None
    pool = set()
    for s in support_texts:
        pool |= numbers(s)
    return sum(n in pool for n in nums) / len(nums)


def lexical_support(answer: str, support_texts: list[str]) -> float | None:
    """Only defined when the answer and its sources share a language (word overlap cannot measure
    grounding of an Arabic answer in an English source); returns None for cross-lingual answers."""
    a = set(analyze(re.sub(r"\[\d+\]", " ", answer)))
    if not a:
        return None
    a_lang = detect_language(answer)["lang"]
    support_texts = [t for t in support_texts if detect_language(t)["lang"] == a_lang]
    if not support_texts:
        return None
    pool = set()
    for s in support_texts:
        pool |= set(analyze(s))
    return len(a & pool) / len(a)


def language_ok(answer: str, expected: str) -> bool:
    return detect_language(answer)["lang"] == expected


def rouge_l(pred: str, ref: str) -> float:
    a, b = tokenize(pred), tokenize(ref)
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def corpus_bleu_chrf(preds: list[str], refs: list[str]) -> dict:
    import sacrebleu
    pp = [" ".join(tokenize(re.sub(r"\[\d+\]", " ", x))) for x in preds]
    rr = [" ".join(tokenize(x)) for x in refs]
    return {"bleu": round(sacrebleu.corpus_bleu(pp, [rr], tokenize="none").score, 2),
            "chrf": round(sacrebleu.corpus_chrf([clean_text(x) for x in preds], [[clean_text(x) for x in refs]]).score, 2)}
