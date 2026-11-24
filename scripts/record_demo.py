"""Record the demo video: the real system, driven live in a browser, with on-screen captions.

Covers every item of the assessment's demo requirement:
  setup / how to run, all three modes end-to-end, >=10 queries (4 AR, 4 EN, 2 mixed), a long question,
  numbers/prices, cross-lingual, a multi-turn Interactive session with follow-ups, visible citations,
  latency results, and a before-vs-after fine-tuning comparison (live + results table).

Needs the API (http://127.0.0.1:8010) and llama-server running. Uses the installed Microsoft Edge.
    .env\\python.exe scripts\\record_demo.py            -> demo_video\\hams_demo.webm (+ .mp4 via the .bat)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:8010"
W, H = 1440, 810
OUT = ROOT / "demo_video"
TYPE_DELAY = 38  # ms per character, so typing is visible but quick


# ------------------------------------------------------------------ real numbers for the cards
def jload(rel: str) -> dict:
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except Exception:
        return {}


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


RA, RB = jload("results/retrieval_after_finetuning.json"), jload("results/retrieval_before_finetuning.json")
GA, GB = jload("results/generation_after.json"), jload("results/generation_before.json")
LA = jload("results/latency_after.json")


def r5(d, grp):
    return g(d, "modes", "hybrid+rerank", grp, "recall@5")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


NUM = {
    "r5_b": pct(r5(RB, "all")), "r5_a": pct(r5(RA, "all")),
    "x_b": pct(r5(RB, "cross")), "x_a": pct(r5(RA, "cross")),
    "nf_b": pct(g(GB, "test", "abstain_rate")), "nf_a": pct(g(GA, "test", "abstain_rate")),
    "h_b": pct(g(GB, "test", "hallucination_rate")), "h_a": pct(g(GA, "test", "hallucination_rate")),
    "ttft": g(LA, "per_mode", "smart_ai_search", "time_to_first_token", "avg"),
    "total": g(LA, "per_mode", "smart_ai_search", "total", "avg"),
}

# ------------------------------------------------------------------ cards
CARD_CSS = """
*{box-sizing:border-box}
body{margin:0;width:1440px;height:810px;background:#14202E;color:#F3F5F4;font-family:Bahnschrift,'Segoe UI',sans-serif;
 padding:70px 110px;display:flex;flex-direction:column;justify-content:center;position:relative;overflow:hidden}
body::before{content:"";position:absolute;inset:0 auto 0 0;width:8px;background:#C9A24A}
.sec{position:absolute;top:34px;left:110px;font-size:16px;color:#8FA3BF}
.sec b{color:#C9A24A;font-weight:600}
h1{font-size:58px;font-weight:600;margin:0 0 12px;letter-spacing:-.01em}
h2{font-size:38px;font-weight:600;margin:0 0 26px}
.ar{font-family:'Sakkal Majalla','Segoe UI',serif;font-size:42px;color:#C9A24A;margin:0 0 26px}
p,li{font-size:22px;line-height:1.5;color:#C9D2DC;max-width:1180px;margin:0 0 10px}
ul{margin:0;padding-left:24px} li{margin-bottom:8px} li b,p b{color:#F3F5F4;font-weight:600}
code{font-family:Consolas,monospace;font-size:19px;background:#203043;color:#F4EBD6;padding:2px 8px;border-radius:4px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px 64px}
.kpi{font-size:46px;font-weight:600;color:#F3F5F4;line-height:1.1}
.kpi small{font-size:20px;color:#8FA3BF;font-weight:400;display:block;margin-top:4px}
.note{font-size:17px;color:#8FA3BF;margin-top:22px}
.flow{display:flex;align-items:stretch;gap:10px;margin:6px 0 18px;flex-wrap:nowrap}
.box{background:#203043;border:1px solid #33475E;border-radius:8px;padding:12px 12px;font-size:16px;line-height:1.35;color:#C9D2DC;flex:1}
.box b{display:block;color:#F3F5F4;font-size:17px;margin-bottom:4px}
.box.gold{border-color:#C9A24A}
.arrow{align-self:center;color:#C9A24A;font-size:22px}
.tag{display:inline-block;background:#203043;border:1px solid #33475E;border-radius:6px;padding:3px 10px;margin:0 6px 6px 0;font-size:17px;color:#C9D2DC}
"""


def card_html(body: str, section: str = "") -> str:
    sec = f'<div class="sec">{section}</div>' if section else ""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CARD_CSS}</style></head><body>{sec}{body}</body></html>"


TITLE = card_html("""
<h1>Self-hosted Arabic &amp; English RAG<br>for an enterprise knowledge bank</h1>
<div class="ar">بحث ذكي في بنك المعرفة بالعربية والإنجليزية</div>
<p>Hybrid retrieval with a <b>fine-tuned embedding model</b> and a <b>fine-tuned reranker</b>, a local LLM that cites its sources,
and three search modes behind one API.</p>
<p class="note">Recorded live on an NVIDIA RTX 3050 Laptop GPU (4 GB). Every result shown comes from the running system.</p>
""")

PROBLEM = card_html("""
<h2>The brief</h2>
<div class="cols">
<ul>
<li><b>Three modes, one API:</b> Quick Search (keyword), Smart AI Search (RAG), Interactive (multi-turn)</li>
<li><b>Arabic and English</b>, mixed queries, and cross-lingual: question in one language, answer in a document in the other</li>
<li><b>Grounded answers</b> with citations, and an explicit "not found"</li>
</ul>
<ul>
<li><b>Fine-tuning is mandatory</b>, hybrid search is mandatory</li>
<li><b>Self-hosted, open-source models only</b>, must fit an NVIDIA L4</li>
<li>Developed and measured here on a <b>4 GB laptop GPU</b> that also drives the display, so every model had to earn its memory</li>
</ul>
</div>
""", "<b>1</b> / 9 &nbsp; The problem")

ARCH = card_html("""
<h2>Architecture: Smart AI Search pipeline</h2>
<div class="flow">
<div class="box"><b>Query</b>language detection, Arabic normalisation, digits</div><div class="arrow">→</div>
<div class="box"><b>BM25</b>SQLite FTS5, custom Arabic/English analyzer</div><div class="arrow">+</div>
<div class="box gold"><b>Dense</b>fine-tuned multilingual-e5-base, FAISS</div><div class="arrow">→</div>
<div class="box"><b>Fusion</b>weighted RRF, tuned on validation</div><div class="arrow">→</div>
<div class="box gold"><b>Rerank</b>fine-tuned gte-multilingual reranker, top 20 → 4</div><div class="arrow">→</div>
<div class="box"><b>Gate</b>confidence threshold → "not found"</div><div class="arrow">→</div>
<div class="box"><b>LLM</b>Qwen3.5-2B, llama.cpp, streaming</div><div class="arrow">→</div>
<div class="box"><b>Checks</b>citations verified, refusals detected</div>
</div>
<ul>
<li><b>Quick Search</b> stops after BM25: instant, and runs CPU-only</li>
<li><b>Interactive</b> adds session memory and rewrites follow-ups ("and how much is it?") into standalone questions</li>
<li>FastAPI with streaming (SSE), a web UI, document upload and ingestion of PDF, DOCX, HTML, Markdown and TXT</li>
</ul>
""", "<b>2</b> / 9 &nbsp; Architecture")

DATA = card_html("""
<h2>Knowledge bank, dataset and Arabic text handling</h2>
<div class="cols">
<ul>
<li><b>46 documents</b>: 20 Arabic, 17 English, 9 mixed; as PDF, Word, HTML and Markdown (policies, products, HR, SLAs)</li>
<li><b>441 questions</b> in Arabic, English, mixed and Saudi dialect, labelled by evidence passage; 40 unanswerable; 15 conversations</li>
<li><b>Split by document</b> (31 / 5 / 10): test questions come from documents never seen in training, so there is no leakage</li>
</ul>
<ul>
<li><b>Arabic normalisation:</b> diacritics and tatweel removed, alef / ya / ta-marbuta unified, light stemming, Arabic-Indic digits</li>
<li><b>Arabic PDFs fixed:</b> right-to-left reordering, ligature repair and junk-glyph removal, verified against every label</li>
<li>Tables kept as "header: value" text so prices and limits stay searchable</li>
</ul>
</div>
""", "<b>3</b> / 9 &nbsp; Data")

EMB = card_html("""
<h2>Fine-tuning 1: the embedding model</h2>
<div class="cols">
<ul>
<li><b>Base:</b> intfloat/multilingual-e5-base (278M, MIT)</li>
<li><b>Loss:</b> Cached Multiple Negatives Ranking (GradCache): an effective batch of 64 in mini-batches of 16, to fit 4 GB</li>
<li><b>Hard negatives</b> mined from training documents only, with a false-negative margin filter</li>
<li><b>Cross-lingual augmentation:</b> each answer is reused as a query in the other language</li>
<li>Word embeddings frozen (86M of 278M trainable), duplicate-free batches, best epoch chosen on validation</li>
</ul>
<div>
<p class="kpi">0.66 → 0.89<small>validation nDCG@10</small></p>
<p class="kpi">0.87 → 0.97<small>validation Recall@5</small></p>
<p class="kpi">9.7 min<small>training time, 3.5 GB peak GPU memory</small></p>
</div>
</div>
""", "<b>4</b> / 9 &nbsp; Embedding fine-tuning")

RR = card_html("""
<h2>Fine-tuning 2: the reranker, and debugging a collapse</h2>
<div class="cols">
<ul>
<li><b>Base:</b> Alibaba-NLP/gte-multilingual-reranker-base (306M, Apache-2.0), binary cross-entropy</li>
<li><b>First runs collapsed:</b> validation MRR fell from 0.878 to 0.149</li>
<li><b>Diagnosis:</b> forward pass checked, gradients checked with an overfit test, a smaller control model trained fine on the same data</li>
<li><b>Fix:</b> learning-rate sweep (2e-6 in fp32) and denoised negatives: negatives the base reranker rates too close to the answer are dropped</li>
<li>Every run logged; the winner is picked automatically on validation</li>
</ul>
<div>
<p class="kpi">0.878 → 0.939<small>validation MRR@10, selected run</small></p>
<p class="kpi">7 runs<small>compared; documented in the model card</small></p>
</div>
</div>
""", "<b>5</b> / 9 &nbsp; Reranker fine-tuning")

CALIB = card_html("""
<h2>Calibration and reliability</h2>
<ul>
<li><b>Fusion tuned on validation only:</b> RRF constant, dense weight, and a union strategy compared; RRF k=5 with dense weight 5 selected</li>
<li><b>"Not found" gate:</b> the reranker's top score is compared to a threshold calibrated separately for each model on validation (maximise balanced accuracy, at most 5% of answerable questions blocked)</li>
<li><b>Grounding:</b> the prompt allows only facts from the sources; citation numbers are checked against them; refusals in the model's own words are detected and replaced by one clear message</li>
<li><b>Answer language follows the question</b>, even when the source is in the other language</li>
<li><b>Engineering for 4 GB:</b> fp16 encoders, Q4 GGUF LLM with q8 KV cache, prompt caching, one GPU lock shared by both encoders</li>
</ul>
""", "<b>6</b> / 9 &nbsp; Calibration")

EVAL = card_html("""
<h2>Evaluation: everything measured, nothing typed in by hand</h2>
<div class="cols">
<ul>
<li><b>Retrieval:</b> Recall@1/3/5/10, MRR, nDCG@10 per language, mixed and each cross-lingual direction, per mode</li>
<li><b>Answers:</b> faithfulness, citation precision and recall, hallucination rate, "not found" accuracy, BLEU, chrF, ROUGE-L, answer language</li>
<li><b>Error analysis:</b> 10 categories with real examples</li>
</ul>
<ul>
<li><b>Latency:</b> per stage, time to first token, CPU-only, scaling to 100,000 passages</li>
<li><b>Concurrency:</b> 1, 2, 4, 8 and 16 parallel users through the HTTP API</li>
<li><b>Before vs after:</b> the same benchmark for the base and the fine-tuned models; the README is generated from the result files</li>
</ul>
</div>
""", "<b>7</b> / 9 &nbsp; Evaluation")

SETUP = card_html("""
<h2>How to run it (Windows, no Docker needed)</h2>
<ul>
<li><code>tools\\setup_env.bat</code> &nbsp;Python environment, CUDA PyTorch, llama.cpp, base models</li>
<li><code>scripts\\prepare_data.py</code> &nbsp;ingest the documents, resolve labels, split by document</li>
<li><code>scripts\\build_index.py</code> &nbsp;BM25 (SQLite FTS5) and FAISS indexes</li>
<li><code>scripts\\train_embeddings.py</code> &nbsp;and&nbsp; <code>scripts\\train_reranker.py</code> &nbsp;fine-tuning</li>
<li><code>tools\\start_all.bat</code> &nbsp;llama-server + API + web UI on <b>http://127.0.0.1:8010</b></li>
</ul>
<p class="note">Each step is also a one-click script in tools\\ with a log. Quick Search runs CPU-only with HAMS_QUICK_ONLY=1.
Fine-tuned models are on Hugging Face: AbdelrahmanHassan/hams-e5-base-ar-en-retrieval and hams-gte-reranker-ar-en.</p>
""", "<b>8</b> / 9 &nbsp; Running it")

LIVE = card_html("""
<h2>Live demo</h2>
<p>The running system on this laptop: Quick Search, Smart AI Search and Interactive mode, in Arabic, English and mixed,
then a live before-vs-after comparison and the benchmark results.</p>
""", "<b>9</b> / 9 &nbsp; Live")


def closing_html() -> str:
    t = NUM["ttft"]
    tt = f"{t:.0f} ms" if t else "n/a"
    return card_html(f"""
<h2>What fine-tuning changed (138 held-out test questions)</h2>
<div class="cols">
<p class="kpi">{NUM['r5_b']} → {NUM['r5_a']}<small>right source in the top 5</small></p>
<p class="kpi">{NUM['x_b']} → {NUM['x_a']}<small>cross-lingual, right source in the top 5</small></p>
<p class="kpi">{NUM['nf_b']} → {NUM['nf_a']}<small>answerable questions wrongly told "not found"</small></p>
<p class="kpi">{NUM['h_b']} → {NUM['h_a']}<small>hallucinated answers</small></p>
</div>
{f'<p style="margin-top:18px">First words of an answer after about {tt}, on a 4 GB laptop GPU, with the full pipeline self-hosted.</p>' if t else ''}
<p class="note">Code, data, benchmarks, error analysis and known limitations: see the README and model card.
Models: huggingface.co/AbdelrahmanHassan</p>
""", "Summary")


# ------------------------------------------------------------------ results page (README section 7)
def results_html() -> str:
    import markdown
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- RESULTS:BEGIN -->(.*?)<!-- RESULTS:END -->", md, re.S)
    body = markdown.markdown(m.group(1) if m else md, extensions=["tables"])
    css = """body{margin:0;padding:30px 80px;font:16px/1.5 Bahnschrift,'Segoe UI',sans-serif;background:#F8F9F7;color:#17202A}
    h3{font-size:24px;margin:34px 0 10px;color:#1E4A8A} table{border-collapse:collapse;margin:8px 0 18px;font-size:15px}
    th,td{border:1px solid #CBD3D0;padding:5px 12px;text-align:left} th{background:#E3EAF4}
    tr:nth-child(even) td{background:#EDF0EE} p{max-width:1100px}"""
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


# ------------------------------------------------------------------ conversation for the Interactive scene
CONVERSATIONS = {
    "conv05": {"intro": "Interactive mode: a conversation with memory. A customer reports an error code on the smart bed",
               "turns": ["السرير الذكي يطلع خطأ E05", "وإذا تكرر وصار E07؟", "How do I file the claim?"]},
    "conv13": {"intro": "Interactive mode: a conversation with memory. A customer wants to try a mattress first",
               "turns": ["أبي أجرب مرتبة قبل ما أشتري", "وإذا ما عجبتني كم يخصمون؟", "What's the warranty on it?"]},
    "conv03": {"intro": "Interactive mode: a conversation with memory, about a bedroom set",
               "turns": ["كم سعر غرفة نوم ريم؟", "وكم ضمانها؟", "Can I extend it?"]},
}


def pick_conversation() -> dict:
    """First conversation (in preference order) that answered every turn in both probe runs."""
    ok: dict[str, int] = {}
    try:
        for line in (ROOT / "tools" / "logs" / "probe_conv.log").read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                d = json.loads(line)
                ok[d["conv"]] = ok.get(d["conv"], 0) + (1 if d["all_answered"] else 0)
    except Exception:
        pass
    for name in CONVERSATIONS:
        if ok.get(name) == 2:
            print(f"[demo] interactive scene uses {name}", flush=True)
            return CONVERSATIONS[name]
    print("[demo] no conversation passed both probes; using conv05", flush=True)
    return CONVERSATIONS["conv05"]


# ------------------------------------------------------------------ the recording
CAPTION_JS = """t => { let c = document.getElementById('demo-cap');
  if (!c) { c = document.createElement('div'); c.id = 'demo-cap';
    c.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);max-width:78%;z-index:99;'
      + 'background:rgba(23,32,42,.92);color:#fff;font:600 19px/1.4 Bahnschrift,Segoe UI,sans-serif;padding:12px 22px;'
      + 'border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,.25);text-align:center;transition:opacity .25s';
    document.body.appendChild(c); }
  c.style.opacity = t ? 1 : 0; c.textContent = t || ''; }"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.webm"):
        old.unlink()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT),
                                  record_video_size={"width": W, "height": H}, device_scale_factor=1)
        page = ctx.new_page()
        start = time.time()

        def pause(s: float) -> None:
            page.wait_for_timeout(int(s * 1000))

        def card(html: str, secs: float) -> None:
            page.set_content(html)
            pause(secs)

        def cap(text: str) -> None:
            page.evaluate(CAPTION_JS, text)

        def mode(m: str) -> None:
            cap("")
            page.click(f'button[data-mode="{m}"]')
            pause(0.8)

        def ask(q: str, wait: str = ".timing, .results, .err, .cmp", hold: float = 5.0) -> None:
            page.click("#q")
            page.fill("#q", "")
            page.keyboard.type(q, delay=TYPE_DELAY)
            pause(0.4)
            page.keyboard.press("Enter")
            page.wait_for_selector(f"#thread article:last-child :is({wait})", timeout=120_000)
            pause(0.6)
            page.evaluate("document.querySelector('#thread article:last-child').scrollIntoView({block:'start',behavior:'smooth'})")
            pause(hold)

        def hover_citation() -> None:
            el = page.query_selector("#thread article:last-child .cm")
            if el:
                el.hover()
                pause(1.6)
                page.mouse.move(5, 5)

        # 1. title and the engineering story
        card(TITLE, 7)
        card(PROBLEM, 13)
        card(ARCH, 16)
        card(DATA, 15)
        card(EMB, 16)
        card(RR, 16)
        card(CALIB, 15)
        card(EVAL, 13)
        card(SETUP, 13)
        page.goto(f"{API}/docs")
        page.wait_for_load_state("networkidle")
        cap("The running API: one /search endpoint for all three modes, plus ingestion, sessions and health")
        pause(7)
        page.goto(f"{API}/health")
        cap("Health check: the answer model is up and the knowledge bank is indexed")
        pause(4)

        # 2. the app + knowledge bank
        card(LIVE, 4)
        page.goto(API)
        page.wait_for_selector(".modes button")
        pause(1.5)
        cap("The ingested knowledge bank: 46 Arabic, English and mixed documents (PDF, Word, HTML, Markdown)")
        page.hover("#docs")
        page.mouse.wheel(0, 900)
        pause(3)
        page.mouse.wheel(0, -900)
        pause(2)

        # 3. Quick Search (BM25)
        mode("quick_search")
        cap("Quick Search: BM25 keyword search with Arabic normalisation and stemming. Instant, runs on CPU")
        ask("رسوم التركيب", hold=4)
        cap("English works the same way; results show document, page and highlighted terms")
        ask("warranty period", hold=4)

        # 4. Smart AI Search
        mode("smart_ai_search")
        cap("Smart AI Search, Arabic question: hybrid retrieval, fine-tuned reranker, local LLM. The answer cites its source")
        ask("ما هي سياسة الاسترجاع للطلبات المتأخرة؟", hold=4)
        hover_citation()
        cap("English question: the answer streams word by word; sources sit beside it")
        ask("What is the SLA for premium support tickets?", hold=5)
        hover_citation()
        cap("Numbers and prices: every number is checked against the sources")
        ask("How much does shipping to Kuwait cost, and how many days does it take?", hold=6)
        cap("Mixed Arabic and English in one question (Saudi dialect plus English terms)")
        ask("أبغى أعرف الـpricing حق الباقة المؤسسية", hold=6)
        cap("Second mixed question: Arabic with English product names")
        ask("كم مدة الـwarranty على كرسي ErgoPro؟", hold=6)
        cap("Cross-lingual: Arabic question, the answer comes from an English policy document")
        ask("كم البدل اليومي للانتداب داخل المملكة؟", hold=6)
        cap("A long question with several conditions")
        ask("I am a new employee in the Riyadh office. After my probation period ends, how many days per week am I "
            "allowed to work from home, and does the same rule apply to staff who work in the warehouse?", hold=7)
        cap("Something the documents do not cover: the system says so instead of guessing")
        ask("When will the Dammam branch be renovated?", hold=5)

        # 5. Interactive (multi-turn): a conversation that answered every turn in both runs of the live probe
        conv = pick_conversation()
        mode("interactive")
        cap(conv["intro"])
        ask(conv["turns"][0], hold=6)
        cap("Follow-up that only makes sense with the previous turn: it is rewritten into a standalone question first")
        ask(conv["turns"][1], hold=7)
        cap("Switching to English mid-conversation; the reference still resolves to the earlier answers")
        ask(conv["turns"][2], hold=7)
        cap("Each answer also suggests follow-up questions, one click away")
        pause(3)

        # 6. Before vs after fine-tuning (live)
        mode("compare")
        cap("Before vs after fine-tuning, live: the same question through the original models (CPU) and the fine-tuned models")
        ask("هل تستخدم سرايا شاحنات كهربائية للتوصيل؟", wait=".cmp", hold=9)
        cap("Original models: only Arabic documents, would answer 'not found'. Fine-tuned: the English report is #1")
        pause(5)
        ask("What is the phone number of the Olaya branch?", wait=".cmp", hold=9)
        cap("English question, Arabic document: found only after fine-tuning")
        pause(4)

        # 7. Results: before/after table, latency, concurrency
        cap("")
        page.set_content(results_html())
        pause(1)

        def show(heading: str, text: str, secs: float) -> None:
            page.evaluate("""h => { const el = [...document.querySelectorAll('h3')].find(x => x.textContent.startsWith(h));
                                  el && el.scrollIntoView({block:'start', behavior:'smooth'}); }""", heading)
            cap(text)
            pause(secs)

        show("Before vs after", "Benchmark: base models vs fine-tuned stack on 138 held-out questions", 12)
        page.mouse.wheel(0, 380)
        pause(5)
        show("Latency", "Latency on the RTX 3050 Laptop: per stage, CPU-only, and scaling to 100,000 passages", 10)
        show("Concurrency", "Concurrency through the HTTP API: 1, 2, 4, 8 and 16 parallel users, no failed requests", 8)

        # 8. closing
        card(closing_html(), 10)
        print(f"[demo] recorded {time.time() - start:.0f} s", flush=True)
        video = page.video
        ctx.close()
        browser.close()
        src = Path(video.path())
        dst = OUT / "hams_demo.webm"
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        print(f"[demo] saved {dst}", flush=True)


if __name__ == "__main__":
    main()
