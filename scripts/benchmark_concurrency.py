"""Concurrency benchmark against the running API (python demo/app.py + llama-server).

    python scripts/benchmark_concurrency.py --levels 1,2,4,8,16 --requests 24

For each mode and concurrency level: average and p95 latency, throughput (queries/s),
failure rate, GPU memory (all processes, via NVML) and RAM, sampled during the run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from hamsrag.config import p  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1) if xs else None


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.gpu, self.ram, self.stop = [], [], False

    def run(self):
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            h = None
        while not self.stop:
            if h:
                self.gpu.append(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20)
            self.ram.append(psutil.virtual_memory().used / 2**30)
            time.sleep(0.25)


async def run_level(client, url, mode, queries, conc, n_req):
    sem = asyncio.Semaphore(conc)
    lat, fails, ttft, errors = [], 0, [], []

    async def one(i):
        nonlocal fails
        q = queries[i % len(queries)]
        body = {"mode": mode, "query": q, "session_id": f"conc_{conc}_{i % conc}" if mode == "interactive" else None}
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.post(url, json=body, timeout=300)
                r.raise_for_status()
                d = r.json()
                lat.append((time.perf_counter() - t) * 1000)
                if "time_to_first_token" in d.get("latency_ms", {}):
                    ttft.append(d["latency_ms"]["time_to_first_token"])
            except Exception as e:  # keep the first errors so a failing run explains itself
                fails += 1
                if len(errors) < 3:
                    detail = getattr(getattr(e, "response", None), "text", "")[:300]
                    errors.append(f"{type(e).__name__}: {e} {detail}".strip())

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n_req)))
    wall = time.perf_counter() - t0
    return {"concurrency": conc, "requests": n_req, "avg_ms": round(statistics.mean(lat), 1) if lat else None,
            "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95), "server_ttft_p95_ms": pct(ttft, 0.95),
            "throughput_qps": round(len(lat) / wall, 2), "failure_rate": round(fails / n_req, 3),
            **({"errors": errors} if errors else {})}


async def main_async(args):
    rows = [json.loads(x) for x in p("data/test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [r["query"] for r in rows]
    url = args.api.rstrip("/") + "/search"
    out = {"api": args.api, "levels": {}}
    async with httpx.AsyncClient() as client:
        h = (await client.get(args.api.rstrip("/") + "/health", timeout=30)).json()
        out["server"] = h
        print("[health]", json.dumps(h, ensure_ascii=False), flush=True)
        # sanity check: one request per mode must succeed before measuring anything
        for mode in args.modes.split(","):
            r = await client.post(url, json={"mode": mode, "query": queries[0]}, timeout=300)
            if r.status_code != 200:
                raise SystemExit(f"[sanity] {mode} -> HTTP {r.status_code}: {r.text[:500]}")
            print(f"[sanity] {mode} ok", flush=True)
        for mode in args.modes.split(","):
            out["levels"][mode] = []
            for conc in [int(x) for x in args.levels.split(",")]:
                n_req = max(args.requests, conc * 2) if mode != "quick_search" else max(args.requests * 4, conc * 8)
                s = Sampler(); s.start()
                res = await run_level(client, url, mode, queries, conc, n_req)
                s.stop = True; s.join()
                res["gpu_mem_MiB_max"] = round(max(s.gpu), 0) if s.gpu else None
                res["ram_used_GiB_max"] = round(max(s.ram), 2) if s.ram else None
                out["levels"][mode].append(res)
                print(mode, json.dumps(res), flush=True)
    res_dir = p("results"); res_dir.mkdir(exist_ok=True)
    (res_dir / "concurrency_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8010")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--modes", default="quick_search,smart_ai_search,interactive")
    ap.add_argument("--requests", type=int, default=16)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
