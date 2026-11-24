"""Render the authored Markdown sources (data/corpus/src) into the formats the knowledge bank
is delivered in (PDF / DOCX / HTML / MD) under data/corpus/kb/<category>/.

PDFs are produced by LibreOffice from the generated DOCX, which shapes Arabic correctly.
The doc_id is stored in document metadata (DOCX subject, PDF Subject, HTML <meta>) so that
ingestion recovers stable ids regardless of format.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import markdown as md

from .ingest import _load_markdown_text
from .text import detect_language

SOFFICE_CANDIDATES = [
    os.environ.get("SOFFICE", ""),
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "soffice", "libreoffice",
]


def _soffice() -> str | None:
    for c in SOFFICE_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    return None


def _split(text: str) -> tuple[dict, str]:
    meta, _ = _load_markdown_text(text)
    body = text.split("\n---", 2)[-1] if text.startswith("---") else text
    body = body.split("\n", 1)[1] if body.startswith("\n") is False and body.startswith("---") else body
    return meta, body.lstrip("-\n")


def _is_rtl(s: str) -> bool:
    return detect_language(s)["lang"] == "ar"


def write_html(meta: dict, body: str, out: Path) -> None:
    rtl = meta.get("lang") != "en"
    html_body = md.markdown(body, extensions=["tables"])
    out.write_text(
        f"""<!DOCTYPE html>
<html lang="{'ar' if rtl else 'en'}" dir="{'rtl' if rtl else 'ltr'}">
<head><meta charset="utf-8"><title>{meta['title']}</title>
<meta name="doc_id" content="{meta['doc_id']}"><meta name="lang" content="{meta['lang']}">
<meta name="category" content="{meta.get('category', '')}"></head>
<body>
{html_body}
</body></html>
""", encoding="utf-8")


def write_docx(meta: dict, body: str, out: Path) -> None:
    import docx
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.shared import Pt

    d = docx.Document()
    d.core_properties.title = meta["title"]
    d.core_properties.subject = meta["doc_id"]
    d.core_properties.language = meta["lang"]
    d.core_properties.keywords = f"lang={meta['lang']}; category={meta.get('category', '')}"
    d.core_properties.author = "Saraya Living (synthetic corpus)"
    style = d.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    def rtlize(par, text):
        if _is_rtl(text):
            pPr = par._p.get_or_add_pPr()
            bidi = OxmlElement("w:bidi"); bidi.set(qn("w:val"), "1"); pPr.append(bidi)
            for r in par.runs:
                rPr = r._r.get_or_add_rPr()
                rtl = OxmlElement("w:rtl"); rtl.set(qn("w:val"), "1"); rPr.append(rtl)

    table_rows: list[list[str]] = []

    def flush_table():
        if not table_rows:
            return
        t = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        t.style = "Table Grid"
        for i, row in enumerate(table_rows):
            for j, cell in enumerate(row[: len(table_rows[0])]):
                t.cell(i, j).text = cell
                for p in t.cell(i, j).paragraphs:
                    rtlize(p, cell)
        table_rows.clear()

    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= set(":-") for c in cells if c):
                table_rows.append(cells)
            continue
        flush_table()
        if not line:
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            p = d.add_heading(text, level=min(level, 4))
        elif line.startswith(("- ", "* ")):
            text = line[2:].replace("**", "")
            p = d.add_paragraph(text, style="List Bullet")
        elif line[:3].rstrip(".").isdigit() and ". " in line[:4]:
            text = line.split(". ", 1)[1].replace("**", "")
            p = d.add_paragraph(text, style="List Number")
        else:
            text = line.lstrip("> ").replace("**", "")
            p = d.add_paragraph(text)
        rtlize(p, text)
    flush_table()
    d.save(str(out))


def build(src_dir: Path, kb_dir: Path, *, pdf: bool = True) -> list[Path]:
    kb_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pdf_jobs: list[tuple[Path, Path]] = []
    for src in sorted(src_dir.glob("*.md")):
        text = src.read_text(encoding="utf-8")
        meta, _ = _load_markdown_text(text)
        body = text.split("---", 2)[2] if text.startswith("---") else text
        fmt = meta.get("format", "md")
        out_dir = kb_dir / meta.get("category", "misc")
        out_dir.mkdir(parents=True, exist_ok=True)
        if fmt == "md":
            out = out_dir / f"{meta['doc_id']}.md"; shutil.copyfile(src, out)
        elif fmt == "html":
            out = out_dir / f"{meta['doc_id']}.html"; write_html(meta, body, out)
        elif fmt in ("docx", "pdf"):
            out = out_dir / f"{meta['doc_id']}.docx"; write_docx(meta, body, out)
            if fmt == "pdf":
                pdf_jobs.append((out, out_dir))
                continue
        else:
            raise ValueError(f"unknown format {fmt} in {src}")
        written.append(out)
    if pdf_jobs:
        soffice = _soffice() if pdf else None
        for docx_path, out_dir in pdf_jobs:
            if soffice:
                subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir),
                                str(docx_path)], check=True, capture_output=True, timeout=180)
                docx_path.unlink()
                written.append(docx_path.with_suffix(".pdf"))
            else:  # no LibreOffice: keep the DOCX so the corpus is still complete
                print(f"[warn] LibreOffice not found, keeping {docx_path.name} as DOCX")
                written.append(docx_path)
    return written
