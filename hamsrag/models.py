"""Loading encoders / rerankers from configs/models.yaml with device + precision handling."""
from __future__ import annotations

import os
from functools import lru_cache

from .config import load_config, p


def device() -> str:
    forced = os.environ.get("HAMS_DEVICE")
    if forced:
        return forced
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(path: str) -> str:
    local = p(path)
    return str(local) if local.exists() else path


def embedder_spec(name: str) -> dict:
    spec = dict(load_config("models.yaml")["embedders"][name])
    spec["name"] = name
    return spec


def reranker_spec(name: str) -> dict:
    spec = dict(load_config("models.yaml")["rerankers"][name])
    spec["name"] = name
    return spec


@lru_cache(maxsize=4)
def load_embedder(name: str, dev: str | None = None):
    from sentence_transformers import SentenceTransformer

    spec = embedder_spec(name)
    dev = dev or device()
    model = SentenceTransformer(_resolve_path(spec["path"]), device=dev,
                                trust_remote_code=spec.get("trust_remote_code", False))
    model.max_seq_length = spec.get("max_seq_length", 256)
    if dev == "cuda":
        model.half()  # fp16 inference: half the VRAM, faster on Ampere
    else:
        model.float()  # weights may be stored in bf16; bf16 matmuls are very slow on most CPUs
    model.spec = spec
    return model


@lru_cache(maxsize=4)
def load_reranker(name: str, dev: str | None = None):
    from sentence_transformers import CrossEncoder

    spec = reranker_spec(name)
    dev = dev or device()
    model = CrossEncoder(_resolve_path(spec["path"]), device=dev, max_length=spec.get("max_length", 384),
                         trust_remote_code=spec.get("trust_remote_code", False))
    if dev == "cuda":
        model.model.half()
    else:
        model.model.float()
    model.spec = spec
    return model
