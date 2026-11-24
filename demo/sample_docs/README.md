# Sample documents for the demo

Use these files to try document upload (left panel of the web UI, `POST /ingest/upload`) or
`scripts/ingest.py --path demo/sample_docs`.

* `eid_showroom_hours_2026_en.txt` is **new**: it is not part of the knowledge bank. Ask
  "متى تفتح المعارض في ثاني أيام العيد؟" (Arabic question, English document) before and after ingesting it
  to see a new document become searchable.
* The other files are copies of knowledge-bank documents in each supported format (Arabic PDF, Arabic DOCX,
  Arabic HTML, English PDF, Markdown). Re-ingesting a document replaces its chunks, so nothing is duplicated.
