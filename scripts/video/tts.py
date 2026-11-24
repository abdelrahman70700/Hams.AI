"""Generate the narration: one MP3 + subtitle file per segment (Microsoft neural voice via edge-tts).
Writes demo_video/audio/<id>.mp3, <id>.srt and durations.json."""
import json
import re
import subprocess
import sys
from pathlib import Path

import edge_tts
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "demo_video" / "audio"
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((Path(__file__).parent / "narration.json").read_text(encoding="utf-8"))
FF = imageio_ffmpeg.get_ffmpeg_exe()


def duration(path: Path) -> float:
    err = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


durs = {}
for seg in cfg["segments"]:
    mp3, srt = OUT / f"{seg['id']}.mp3", OUT / f"{seg['id']}.srt"
    for voice in (cfg["voice"], "en-US-GuyNeural"):
        try:
            com = edge_tts.Communicate(seg["text"], voice, rate="+2%", boundary="WordBoundary")
            sub = edge_tts.SubMaker()
            with open(mp3, "wb") as f:
                for chunk in com.stream_sync():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                        sub.feed(chunk)
            srt.write_text(sub.get_srt(), encoding="utf-8")
            break
        except Exception as e:  # fall back to a second voice if the first is unavailable
            print(f"[tts] {seg['id']} {voice} failed: {e}", flush=True)
    durs[seg["id"]] = round(duration(mp3), 3)
    print(f"[tts] {seg['id']}: {durs[seg['id']]} s", flush=True)
(OUT / "durations.json").write_text(json.dumps(durs, indent=1), encoding="utf-8")
print("[tts] total", round(sum(durs.values()), 1), "s")
