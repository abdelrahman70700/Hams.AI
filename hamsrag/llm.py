"""Minimal client for an OpenAI-compatible local server (llama.cpp `llama-server`, Ollama, vLLM).

Supports blocking and streaming chat completions and measures time-to-first-token on the client
side. Thinking/reasoning output is disabled for Qwen3-family templates via
`chat_template_kwargs.enable_thinking=False` (ignored by servers that don't know it).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator

import httpx

from .config import load_config


@dataclass
class Generation:
    text: str
    ttft_ms: float
    total_ms: float
    completion_tokens: int = 0
    prompt_tokens: int = 0
    timings: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config("serving_config.yaml")["llm"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg.get("model", "local")
        self.max_tokens = cfg.get("max_tokens", 320)
        self.temperature = cfg.get("temperature", 0.1)
        self.timeout = cfg.get("timeout_s", 60)
        self._client = httpx.Client(timeout=self.timeout)

    def _payload(self, messages, max_tokens, temperature, stream, stop=None) -> dict:
        body = {"model": self.model, "messages": messages, "stream": stream,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": 0.9, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        if stop:
            body["stop"] = stop
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def health(self) -> bool:
        try:
            r = self._client.get(self.base_url.replace("/v1", "") + "/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None,
             stop: list[str] | None = None) -> Generation:
        """Blocking call implemented on top of streaming, so TTFT is always measured."""
        parts, gen = [], None
        for piece in self.stream(messages, max_tokens=max_tokens, temperature=temperature, stop=stop):
            if isinstance(piece, Generation):
                gen = piece
            else:
                parts.append(piece)
        return gen

    def stream(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None,
               stop: list[str] | None = None) -> Iterator[str | Generation]:
        """Yields text deltas, then a final Generation object with timings."""
        t0 = time.perf_counter()
        ttft = None
        text, usage, timings = [], {}, {}
        with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                 json=self._payload(messages, max_tokens, temperature, True, stop)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("timings"):
                    timings = obj["timings"]
                for ch in obj.get("choices", []):
                    delta = (ch.get("delta") or {}).get("content") or ""
                    if delta:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        text.append(delta)
                        yield delta
        total = (time.perf_counter() - t0) * 1000
        yield Generation("".join(text), ttft if ttft is not None else total, total,
                         usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0), timings)
