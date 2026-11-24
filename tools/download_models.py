"""Download base models in priority order (slow connection: what is needed first comes first).
Weights only (no ONNX/OpenVINO duplicates). Re-running skips files already downloaded.

    python tools/download_models.py            # everything, in priority order
    python tools/download_models.py --core     # encoders + rerankers + dev LLM only
"""
import argparse
import os
import time
import traceback

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

MODELS = os.environ.get("MODELS_DIR", r"D:\Hams.AI\models")
ALLOW = ["*.json", "*.txt", "*.py", "*.model", "sentencepiece*", "tokenizer*", "1_Pooling/*",
         "modules.json", "*.safetensors"]

# (kind, repo, extra) in priority order
PLAN = [
    ("enc", "intfloat/multilingual-e5-base", None),
    ("enc", "Alibaba-NLP/new-impl", None),                 # custom code for both gte models
    ("enc", "Alibaba-NLP/gte-multilingual-reranker-base", None),
    ("enc", "intfloat/multilingual-e5-small", None),
    ("enc", "Alibaba-NLP/gte-multilingual-base", None),
    ("enc", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", None),
    ("gguf", "unsloth/Qwen3.5-2B-GGUF", "Q4_K_M.gguf"),    # dev default LLM
    ("gguf", "unsloth/Qwen3.5-4B-GGUF", "Q4_K_M.gguf"),    # L4 profile candidate
    ("enc", "BAAI/bge-reranker-v2-m3", None),              # zero-shot reference reranker
    ("gguf", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Q4_K_M.gguf"),
    ("enc", "BAAI/bge-m3", None),                          # zero-shot reference embedder
]
CORE = 7  # first N entries = what Day-1 work needs


def get_gguf(repo: str, suffix: str) -> None:
    files = [f for f in list_repo_files(repo) if f.endswith(suffix) and "mmproj" not in f and "/" not in f
             and "UD-" not in f]
    if not files:
        print("NO MATCH", repo, suffix, flush=True); return
    path = hf_hub_download(repo, files[0], local_dir=os.path.join(MODELS, "gguf"))
    print("OK", repo, path, f"{os.path.getsize(path)/2**30:.2f} GiB", flush=True)


def get_encoder(repo: str) -> None:
    files = list_repo_files(repo)
    allow = list(ALLOW)
    if not any(f.endswith(".safetensors") for f in files):
        allow.append("pytorch_model.bin")
    path = snapshot_download(repo, allow_patterns=allow, ignore_patterns=["onnx/*", "openvino/*", "*.onnx"])
    print("OK", repo, path, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--encoders-only", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on repo id, e.g. Qwen3.5-2B")
    args = ap.parse_args()
    plan = PLAN[:CORE] if args.core else PLAN
    if args.encoders_only:
        plan = [x for x in plan if x[0] == "enc"]
    if args.only:
        plan = [x for x in PLAN if args.only.lower() in x[1].lower()]
    for kind, repo, extra in plan:
        t = time.time()
        for attempt in range(3):
            try:
                get_gguf(repo, extra) if kind == "gguf" else get_encoder(repo)
                break
            except Exception:
                print(f"FAIL attempt {attempt + 1}", repo, flush=True); traceback.print_exc()
                time.sleep(10)
        print(f"  [{time.time() - t:.0f}s] {repo}", flush=True)
        with open(os.path.join(os.path.dirname(__file__), "logs", "models_done.txt"), "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {repo}\n")


if __name__ == "__main__":
    main()
