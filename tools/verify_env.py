"""Environment check: versions + a real CUDA matmul. Read-only."""
import importlib, platform, sys
print("python", sys.version.replace("\n", " "), platform.platform())
for m in ["torch", "transformers", "sentence_transformers", "faiss", "Stemmer", "fastapi",
          "pydantic", "pypdf", "docx", "sacrebleu", "rouge_score", "huggingface_hub", "datasets"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:22s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:22s} MISSING ({e.__class__.__name__}: {e})")
import sqlite3
con = sqlite3.connect(":memory:")
try:
    con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    print("sqlite", sqlite3.sqlite_version, "FTS5 ok")
except Exception as e:  # noqa: BLE001
    print("sqlite FTS5 MISSING", e)
try:
    import torch
    print("cuda available:", torch.cuda.is_available(), "torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu:", p.name, f"{p.total_memory/2**30:.2f} GiB", "cc", f"{p.major}.{p.minor}",
              "bf16:", torch.cuda.is_bf16_supported())
        free, total = torch.cuda.mem_get_info()
        print(f"free now: {free/2**30:.2f} / {total/2**30:.2f} GiB")
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        print("matmul ok:", float((a @ a).float().abs().mean()) > 0)
except Exception as e:  # noqa: BLE001
    print("torch check failed:", e)
