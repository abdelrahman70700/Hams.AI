"""Film the live system for each 'live' narration segment, at 1920x1080, as sharp JPEG frames
(Chrome DevTools screencast), and hold each scene at least as long as its narration.
Writes demo_video/footage/<id>.mp4."""
import asyncio
import base64
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8010"
OUT = ROOT / "demo_video" / "footage4k"
DUR = json.loads((ROOT / "demo_video" / "audio" / "durations.json").read_text(encoding="utf-8"))
FF = imageio_ffmpeg.get_ffmpeg_exe()
TYPE_DELAY = 32

CONV = ["كم يوم إجازة سنوية للموظف؟", "وبعد خمس سنوات؟", "Is there leave for Hajj?"]
probe = ROOT / "tools" / "logs" / "probe_conv.log"
if probe.exists():  # prefer the first conversation that answered every turn in both live probes
    ok = {}
    for line in probe.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            d = json.loads(line)
            ok[d["conv"]] = ok.get(d["conv"], 0) + d["all_answered"]
    pass


ONLY = set(sys.argv[1:])


def want(seg: str) -> bool:
    return not ONLY or seg in ONLY


class Film:
    def __init__(self, page, cdp):
        self.page, self.cdp = page, cdp
        self.frames, self.dir, self.t0 = [], None, 0.0
        cdp.on("Page.screencastFrame", lambda p: asyncio.ensure_future(self._frame(p)))

    async def _frame(self, p):
        if self.dir is not None:
            f = self.dir / f"{len(self.frames):05d}.jpg"
            f.write_bytes(base64.b64decode(p["data"]))
            self.frames.append((time.time() - self.t0, f))
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})
        except Exception:
            pass

    async def start(self, seg):
        self.seg = seg
        self.dir = OUT / f"_{seg}"
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.frames = []
        self.t0 = time.time()
        f0 = self.dir / "00000_first.jpg"
        await self.page.screenshot(path=str(f0), type="jpeg", quality=92, scale="device")
        self.frames.append((0.0, f0))

    async def stop(self, min_len):
        el = time.time() - self.t0
        if el < min_len:
            await self.page.wait_for_timeout(int((min_len - el) * 1000))
        end = time.time() - self.t0
        d, self.dir = self.dir, None
        frames = sorted(self.frames, key=lambda x: x[0])
        from PIL import Image  # screencast frames arrive at mixed sizes: make every one exactly 3840x2160
        for _, f in frames:
            with Image.open(f) as im:
                if im.size != (3840, 2160):
                    im.convert("RGB").resize((3840, 2160), Image.LANCZOS).save(f, quality=93)
        lines = []
        for i, (t, f) in enumerate(frames):
            nxt = frames[i + 1][0] if i + 1 < len(frames) else end
            lines.append(f"file '{f.as_posix()}'\nduration {max(nxt - t, 0.001):.3f}")
        lines.append(f"file '{frames[-1][1].as_posix()}'")
        (d / "list.txt").write_text("\n".join(lines), encoding="utf-8")
        (d / "end.txt").write_text(f"{end:.3f}")  # frames are kept and encoded afterwards (Windows ffmpeg crashes at 4K)
        print(f"[film] {self.seg}: {end:.1f} s, {len(frames)} frames", flush=True)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="msedge", headless=True)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=2.4)
        page = await ctx.new_page()
        cdp = await ctx.new_cdp_session(page)
        film = Film(page, cdp)
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 3840, "maxHeight": 2160, "everyNthFrame": 1})

        async def mode(m):
            await page.click(f'button[data-mode="{m}"]')
            await page.wait_for_timeout(600)

        async def ask(q, wait=".timing, .results, .err, .cmp", hold=1.5):
            await page.click("#q")
            await page.fill("#q", "")
            await page.keyboard.type(q, delay=TYPE_DELAY)
            await page.wait_for_timeout(300)
            await page.keyboard.press("Enter")
            await page.wait_for_selector(f"#thread article:last-child :is({wait})", timeout=120_000)
            await page.wait_for_timeout(400)
            await page.evaluate("document.querySelector('#thread article:last-child').scrollIntoView({block:'start',behavior:'smooth'})")
            await page.wait_for_timeout(int(hold * 1000))

        async def hover_cite():
            el = await page.query_selector("#thread article:last-child .cm")
            if el:
                await el.hover()
                await page.wait_for_timeout(1500)
                await page.mouse.move(5, 5)

        # warm-up (not filmed): load the page, the answer model and the CPU copy of the original models
        await page.goto(API)
        await page.wait_for_selector(".modes button")
        await page.evaluate("""async () => { await fetch('/compare', {method:'POST', headers:{'Content-Type':'application/json'},
                                 body: JSON.stringify({query:'warm up', k:3})}); }""")
        await page.evaluate("""async () => { await fetch('/search', {method:'POST', headers:{'Content-Type':'application/json'},
                                 body: JSON.stringify({mode:'smart_ai_search', query:'warm up'})}); }""")

        L = lambda k: DUR[k] + 0.7  # noqa: E731

        async def seg_l01():
            await page.goto(f"{API}/docs")
            await page.wait_for_load_state("networkidle")
            await film.start("l01_api")
            await page.wait_for_timeout(int(L("l01_api") * 450))
            await page.goto(f"{API}/health")
            await film.stop(L("l01_api"))

        async def seg_l02():
            await page.goto(API)
            await page.wait_for_selector(".modes button")
            await mode("quick_search")
            await film.start("l02_quick")
            await ask("رسوم التركيب", hold=2.5)
            await ask("warranty period", hold=1)
            await film.stop(L("l02_quick"))

        async def seg_l03():
            await mode("smart_ai_search")
            await film.start("l03_ar")
            await ask("ما هي سياسة الاسترجاع للطلبات المتأخرة؟", hold=1)
            await hover_cite()
            await film.stop(L("l03_ar"))

        async def seg_l04():
            await mode("smart_ai_search")
            await film.start("l04_num")
            await ask("How much does shipping to Kuwait cost, and how many days does it take?", hold=1)
            await hover_cite()
            await film.stop(L("l04_num"))

        async def seg_l05():
            await mode("smart_ai_search")
            await film.start("l05_mixed")
            await ask("أبغى أعرف الـpricing حق الباقة المؤسسية", hold=2.5)
            await ask("كم مدة الـwarranty على كرسي ErgoPro؟", hold=1)
            await film.stop(L("l05_mixed"))

        async def seg_l06():
            await mode("smart_ai_search")
            await film.start("l06_cross")
            await ask("كم البدل اليومي للانتداب داخل المملكة؟", hold=1)
            await hover_cite()
            await film.stop(L("l06_cross"))

        async def seg_l07():
            await mode("smart_ai_search")
            await film.start("l07_long")
            await ask("I am a new employee in the Riyadh office. After my probation period ends, how many days per week am I "
                      "allowed to work from home, and does the same rule apply to staff who work in the warehouse?", hold=2.5)
            await ask("When will the Dammam branch be renovated?", hold=1)
            await film.stop(L("l07_long"))

        async def seg_l08():
            await mode("interactive")
            await film.start("l08_conv")
            await ask(CONV[0], hold=1.5)
            await ask(CONV[1], hold=2.5)
            await ask(CONV[2], hold=1.5)
            await film.stop(L("l08_conv"))

        async def seg_l09():
            await mode("compare")
            await film.start("l09_compare")
            await ask("هل تستخدم سرايا شاحنات كهربائية للتوصيل؟", wait=".cmp", hold=1)
            await film.stop(L("l09_compare"))

        await page.goto(API)
        await page.wait_for_selector(".modes button")
        for name, fn in [("l01_api", seg_l01), ("l02_quick", seg_l02), ("l03_ar", seg_l03), ("l04_num", seg_l04),
                         ("l05_mixed", seg_l05), ("l06_cross", seg_l06), ("l07_long", seg_l07), ("l08_conv", seg_l08),
                         ("l09_compare", seg_l09)]:
            if want(name):
                await fn()

        await cdp.send("Page.stopScreencast")
        await browser.close()


asyncio.run(main())
