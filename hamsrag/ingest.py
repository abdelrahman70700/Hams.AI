"""Document loading and structure-aware chunking.

Loaders turn PDF / DOCX / HTML / Markdown / TXT into a flat list of ``Block`` objects
(heading / paragraph / table-row, with page numbers for PDFs).  The chunker groups blocks
under their section heading and packs them into chunks of bounded size.  Every chunk carries
the metadata the API must return: doc_id, chunk_id, title, section, page, source path.

Table rows are serialised as ``header: value; header: value`` so that facts stored in tables
(prices, SLA targets, warranty periods) stay self-describing after chunking.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .text import clean_text, detect_language

# --------------------------------------------------------------------------- data model


@dataclass
class Block:
    kind: str  # "h1".."h6" | "p" | "row"
    text: str
    page: int | None = None


@dataclass
class Document:
    doc_id: str
    title: str
    lang: str
    source: str
    blocks: list[Block]
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    page: int | None
    lang: str
    source: str
    text: str  # clean body text (display / snippets / LLM context)
    ord: int

    def embed_text(self) -> str:
        """Text given to the encoder: contextual header + body."""
        head = self.title if not self.section or self.section == self.title else f"{self.title} — {self.section}"
        return f"{head}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- helpers
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _row_text(header: list[str], cells: list[str]) -> str:
    cells = [c.strip() for c in cells]
    if header and len(header) == len(cells):
        return "؛ ".join(f"{h}: {c}" for h, c in zip(header, cells) if c and c != "—") \
            if _is_ar(" ".join(header)) else \
            "; ".join(f"{h}: {c}" for h, c in zip(header, cells) if c and c != "—")
    return " | ".join(c for c in cells if c)


def _is_ar(s: str) -> bool:
    return detect_language(s)["lang"] == "ar"


def _doc_id_from_path(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", path.stem).strip("_").lower() or \
        "doc_" + hashlib.md5(str(path).encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- loaders
def _load_markdown_text(text: str) -> tuple[dict, list[Block]]:
    meta: dict = {}
    m = _FRONT.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        text = text[m.end():]
    blocks: list[Block] = []
    header: list[str] = []
    para: list[str] = []

    def flush():
        if para:
            blocks.append(Block("p", " ".join(para)))
            para.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush(); header = []; continue
        if line.startswith("#"):
            flush(); level = len(line) - len(line.lstrip("#"))
            blocks.append(Block(f"h{min(level, 6)}", line.lstrip("#").strip())); continue
        if line.startswith("|"):
            flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # separator row
            if not header:
                header = cells
            else:
                blocks.append(Block("row", _row_text(header, cells)))
            continue
        if line.startswith(">"):
            line = line.lstrip("> ").strip()
        if re.match(r"^([-*]|\d+\.)\s+", line):
            flush(); blocks.append(Block("p", re.sub(r"^([-*]|\d+\.)\s+", "", line).replace("**", ""))); continue
        para.append(line.replace("**", ""))
    flush()
    return meta, blocks


def load_markdown(path: Path) -> Document:
    meta, blocks = _load_markdown_text(path.read_text(encoding="utf-8"))
    return _finish(path, meta, blocks)


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [Block("p", p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _finish(path, {}, blocks)


def load_html(path: Path) -> Document:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    meta = {k: v for k, v in ((m.get("name"), m.get("content")) for m in soup.find_all("meta"))
            if k in {"doc_id", "lang", "category"} and v}
    if soup.title and soup.title.string:
        meta.setdefault("title", soup.title.string.strip())
    blocks: list[Block] = []
    body = soup.body or soup
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.name == "table":
            rows = el.find_all("tr")
            header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])] if rows else []
            for r in rows[1:]:
                blocks.append(Block("row", _row_text(header, [c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])])))
        elif el.find_parent("table") is None and el.find_parent("li") is None:
            txt = el.get_text(" ", strip=True)
            if txt:
                blocks.append(Block(el.name if el.name.startswith("h") else "p", txt))
    return _finish(path, meta, blocks)


def load_docx(path: Path) -> Document:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    meta = {}
    cp = d.core_properties
    if cp.title:
        meta["title"] = cp.title
    if cp.subject:  # we store doc_id in "subject" when generating the corpus
        meta["doc_id"] = cp.subject
    if cp.language:
        meta["lang"] = cp.language
    blocks: list[Block] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, d)
            txt = p.text.strip()
            if not txt:
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            m = re.match(r"(heading|title)\s*(\d*)", style)
            if m:
                level = int(m.group(2) or 1)
                blocks.append(Block(f"h{level}", txt))
            else:
                blocks.append(Block("p", txt))
        elif tag == "tbl":
            t = Table(child, d)
            rows = [[c.text for c in r.cells] for r in t.rows]
            if rows:
                for r in rows[1:]:
                    blocks.append(Block("row", _row_text(rows[0], r)))
    return _finish(path, meta, blocks)


_AR_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
# Latin capital letters glued to Arabic letters are mis-mapped diacritic glyphs (e.g. tanween
# rendered as "O" by some fonts' ToUnicode tables), not real text.
_GLYPH_JUNK = re.compile(r"(?<=[\u0621-\u064A])[A-Z](?![A-Za-z])|(?<![A-Za-z])[A-Z](?=[\u0621-\u064A])")


def fix_pdf_line(text: str) -> str:
    """Convert a PDF text line from visual to logical order.

    pdfplumber returns characters left-to-right as drawn on the page (visual order). For lines
    containing Arabic we run the Unicode bidi algorithm with an RTL base direction, which
    reverses the Arabic runs back to logical order while keeping numbers and Latin runs intact.
    Bidi runs BEFORE NFKC so that presentation-form ligatures (lam-alef) are reversed as single
    glyphs and only then decomposed into letters.
    """
    if not _AR_CHAR.search(text):
        return unicodedata.normalize("NFKC", text)
    from bidi import get_display

    logical = get_display(text, base_dir="R")
    logical = unicodedata.normalize("NFKC", logical)
    return _GLYPH_JUNK.sub("", logical)


def _prepare_pdf_page(page):
    """Glyph-level clean-up before line building.

    * Zero-advance glyphs are diacritics (tashkeel). Some fonts map them to ASCII symbols such
      as '"', so they are dropped (tashkeel is removed by normalization anyway).
    * A ligature glyph (e.g. lam-alef) carries several letters in logical order inside one
      visually ordered glyph; reversing its text lets the later bidi pass restore it correctly.
    """
    page = page.filter(lambda o: o.get("object_type") != "char" or (o["x1"] - o["x0"]) > 0.3)
    for c in page.chars:
        if len(c["text"]) > 1 and _AR_CHAR.search(c["text"]):
            c["text"] = c["text"][::-1]
    return page


def _fix_cell(text: str | None) -> str:
    return " ".join(fix_pdf_line(" ".join(line.split())) for line in (text or "").split("\n") if line.strip())


def load_pdf(path: Path) -> Document:
    """PDF loader: logical-order Arabic, heading detection by font size, ruled tables as rows."""
    import pdfplumber

    meta: dict = {}
    blocks: list[Block] = []
    with pdfplumber.open(str(path)) as pdf:
        info = pdf.metadata or {}
        if info.get("Title"):
            meta["title"] = str(info["Title"])
        if info.get("Subject"):
            meta["doc_id"] = str(info["Subject"])
        m = re.search(r"lang=(\w+)", str(info.get("Keywords", "")))
        if m:
            meta["lang"] = m.group(1)
        pages = [_prepare_pdf_page(pg) for pg in pdf.pages]
        sizes = Counter(round(c["size"], 1) for pg in pages for c in pg.chars)
        body = sizes.most_common(1)[0][0] if sizes else 11.0
        # Heading levels: distinct sizes clearly above body text, largest = h1
        head_sizes = sorted({s for s in sizes if s >= body * 1.15}, reverse=True)
        level_of = {s: min(i + 1, 3) for i, s in enumerate(head_sizes)}
        for pno, page in enumerate(pages, start=1):
            items: list[tuple[float, Block]] = []
            tables = page.find_tables()
            for t in tables:
                rows = [[_fix_cell(c) for c in r] for r in t.extract()]
                rows = [r for r in rows if any(r)]
                if not rows:
                    continue
                header = rows[0]
                for r in rows[1:]:
                    items.append((t.bbox[1], Block("row", _row_text(header, r), pno)))

            def in_table(line) -> bool:
                return any(t.bbox[0] - 2 <= line["x0"] and line["x1"] <= t.bbox[2] + 2 and
                           t.bbox[1] - 2 <= line["top"] and line["bottom"] <= t.bbox[3] + 2 for t in tables)

            para: list[str] = []
            para_top = 0.0
            for line in page.extract_text_lines(return_chars=True, strip=True):
                if in_table(line):
                    continue
                text = fix_pdf_line(line["text"])
                if not text.strip():
                    continue
                size = round(max(c["size"] for c in line["chars"]), 1)
                if size in level_of:
                    if para:
                        items.append((para_top, Block("p", " ".join(para), pno))); para = []
                    items.append((line["top"], Block(f"h{level_of[size]}", text, pno)))
                    continue
                if not para:
                    para_top = line["top"]
                if para and para[-1].endswith("-") and text[:1].islower():
                    para[-1] = para[-1] + text  # re-join a word hyphenated across lines
                else:
                    para.append(text)
                if re.search(r"[.:؟?!]$", text):
                    items.append((para_top, Block("p", " ".join(para), pno))); para = []
            if para:
                items.append((para_top, Block("p", " ".join(para), pno)))
            items.sort(key=lambda x: x[0])
            blocks.extend(b for _, b in items)
    return _finish(path, meta, blocks)


LOADERS = {".md": load_markdown, ".markdown": load_markdown, ".txt": load_txt,
           ".html": load_html, ".htm": load_html, ".docx": load_docx, ".pdf": load_pdf}


def load_document(path: str | Path) -> Document:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader(path)


def _finish(path: Path, meta: dict, blocks: list[Block]) -> Document:
    blocks = [Block(b.kind, clean_text(b.text), b.page) for b in blocks if clean_text(b.text)]
    title = meta.get("title") or next((b.text for b in blocks if b.kind == "h1"), path.stem)
    full = " ".join(b.text for b in blocks[:50])
    det = detect_language(full)
    lang = meta.get("lang") or ("mixed" if det["mixed"] and _mixed_ratio(full) else det["lang"])
    return Document(doc_id=str(meta.get("doc_id") or _doc_id_from_path(path)), title=clean_text(str(title)),
                    lang=str(lang), source=str(path).replace("\\", "/"), blocks=blocks, meta=meta)


def _mixed_ratio(text: str) -> bool:
    from .text import script_stats
    ar, en = script_stats(text)
    return min(ar, en) / max(1, ar + en) >= 0.15


# --------------------------------------------------------------------------- chunking
def chunk_document(doc: Document, *, max_words: int = 120, min_words: int = 40) -> list[Chunk]:
    """Pack blocks into chunks that never cross a level-1/2 section boundary.

    max_words ~120 keeps chunks around 200-250 XLM-R tokens for Arabic, inside the 256-token
    training length.  Small trailing pieces are merged into the previous chunk of the section.
    """
    sections: list[tuple[str, list[Block]]] = []
    current_title, current = doc.title, []
    for b in doc.blocks:
        if b.kind in {"h1", "h2"}:
            if current:
                sections.append((current_title, current))
            current_title, current = (b.text if b.kind == "h2" else doc.title), []
        elif b.kind.startswith("h"):
            current.append(Block("p", b.text + ":", b.page))  # h3+ become inline lead-ins
        else:
            current.append(b)
    if current:
        sections.append((current_title, current))

    # Merge short neighbouring sections (e.g. a one-line intro + the first real section) so
    # that no chunk is a fragment; the merged chunk keeps both section names.
    merged: list[tuple[str, list[Block]]] = []
    for section, blocks in sections:
        n = sum(len(b.text.split()) for b in blocks)
        if merged:
            prev_sec, prev_blocks = merged[-1]
            prev_n = sum(len(b.text.split()) for b in prev_blocks)
            if (prev_n < min_words or n < min_words) and prev_n + n <= max_words:
                lead = [] if section == doc.title else [Block("p", section + ":", blocks[0].page if blocks else None)]
                name = prev_sec if section == prev_sec else f"{prev_sec} / {section}"
                merged[-1] = (name, prev_blocks + lead + blocks)
                continue
        merged.append((section, list(blocks)))
    sections = merged

    chunks: list[Chunk] = []
    for section, blocks in sections:
        pieces: list[list[Block]] = [[]]
        words = 0
        for b in blocks:
            n = len(b.text.split())
            if pieces[-1] and words + n > max_words:
                pieces.append([]); words = 0
            pieces[-1].append(b); words += n
        if len(pieces) > 1 and sum(len(b.text.split()) for b in pieces[-1]) < min_words:
            pieces[-2].extend(pieces.pop())
        for piece in pieces:
            if not piece:
                continue
            text = "\n".join(b.text for b in piece)
            page = next((b.page for b in piece if b.page is not None), None)
            ordn = len(chunks)
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}#{ordn}", doc_id=doc.doc_id, title=doc.title,
                                section=section, page=page, lang=doc.lang, source=doc.source,
                                text=text, ord=ordn))
    return chunks
