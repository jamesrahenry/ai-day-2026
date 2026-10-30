"""Render the poster .dc.html locally: swap /_blob/ ids for local figure files, screenshot at A1 px size,
and report the bounding boxes of every top-level section so overflow is measurable, not guessed."""
import re, sys, json
from pathlib import Path
from playwright.sync_api import sync_playwright
SCR = Path(__file__).resolve().parents[1]
SRC = SCR/"poster/project/Main.dc.html"
BLOBS = {  # asset id -> local file
    "dbb14ea3c46a32cc4245a21c805d15a1": SCR/"figgen/fig_ribbon_credibility_null.png",
    "76b38cd1f261a4f3b1f17e7da4be8987": SCR/"figgen/fig_sep_curves.png",
    "2662a7c9f553a0431bf6a49aef4cbb4d": SCR/"figgen/qr_ai_day_2026.png",
}
html = SRC.read_text()
for k,v in BLOBS.items(): html = html.replace(f"/_blob/{k}", v.resolve().as_uri())
missing = re.findall(r'/_blob/([0-9a-f]{32})', html)
html = html.replace('<script src="./support.js"></script>', '')
out = SCR/"render"; (out/"poster_render.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width":2246,"height":3178}, device_scale_factor=float(sys.argv[1]) if len(sys.argv)>1 else 0.5)
    pg.goto((out/"poster_render.html").resolve().as_uri()); pg.wait_for_timeout(2500)  # fonts
    boxes = pg.evaluate("""() => {
      const root = document.querySelector('x-dc > div');
      const r = root.getBoundingClientRect();
      const rows = [...root.children].map(el => { const b = el.getBoundingClientRect();
        const h2 = el.querySelector('h2'); return {tag: el.tagName.toLowerCase(), title: h2 ? h2.textContent.trim().slice(0,50) : '', top: Math.round(b.top), bottom: Math.round(b.bottom), height: Math.round(b.height), scrollH: el.scrollHeight}; });
      const img = document.querySelector('section:nth-of-type(4) img'); const ib = img ? img.getBoundingClientRect() : null;
      return {root: {height: Math.round(r.height), scrollHeight: root.scrollHeight}, rows, fig4: ib ? {w: Math.round(ib.width), h: Math.round(ib.height)} : null};
    }""")
    pg.screenshot(path=str(out/"poster.png"), full_page=True)
    b.close()
print(json.dumps(boxes, indent=1)); print("missing blobs:", missing)
