"""Build the lightning-talk deck from talk/slides/*.html.

Each slide is one 1920x1080 <section> (inline styles only). This script stitches them into
talk/talk.html, renders talk/talk.pdf (one page per slide, fonts embedded), and writes the
speaker notes (the <aside> in each slide) to talk/NOTES.md.

Run from any directory: python talk/build_talk.py
Needs: pip install playwright && playwright install chromium (same as render/render_poster.py).
Fonts (Bricolage Grotesque, Source Serif 4) load from Google Fonts at render time.
"""
import json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
deck = json.loads((HERE / "deck.json").read_text())
ARROW = ('<svg width="56" height="28" viewBox="0 0 56 28" aria-label="leads to"><path d="M0 9h34V0l22 14-22 14V19H0z" fill="#8A94A6"/></svg>')

slides, notes = [], []
for sid in deck["order"]:
    html = (HERE / "slides" / f"{sid}.html").read_text()
    m = re.search(r"<aside>(.*?)</aside>", html, re.S)
    notes.append((sid, m.group(1).strip() if m else ""))
    html = re.sub(r"<aside>.*?</aside>", "", html, flags=re.S)
    html = re.sub(r'<x-shape kind="arrow-right"[^>]*></x-shape>', ARROW, html)
    slides.append(html)

fonts = " ".join(f['href'] for f in deck["faces"].values())
links = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values())
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{deck['title']}</title>
{links}
<style>
@page {{ size: 1920px 1080px; margin: 0 }}
*, *::before, *::after {{ box-sizing: border-box }}
body {{ margin: 0; background: #0b1c33 }}
h1, h2, h3, p {{ margin: 0 }}
section {{ width: 1920px; height: 1080px; position: relative; overflow: hidden; break-after: page }}
@media screen {{ section {{ margin: 0 auto 24px }} }}
</style></head><body>
{chr(10).join(slides)}
</body></html>"""
(HERE / "talk.html").write_text(page)

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
def spoken(n):  # drop the timing cue and any "If asked..." Q&A note; neither is read aloud
    n = n.split("\n\nIf asked")[0]
    return re.sub(r"^About \d+ seconds\.\s*", "", n)
words = sum(len(spoken(n).split()) for _, n in notes)
md = [f"# Lightning talk: speaker notes", "", f"*Written: {now}*", "",
      f"About {words} spoken words in the notes (timing cues and Q&A notes excluded), roughly {words / 150:.1f} minutes at 150 words per minute. "
      "Slides: `talk.pdf` (print/present) and `talk.html`; sources in `slides/`.", ""]
for i, (sid, n) in enumerate(notes, 1):
    md += [f"## {i}. {sid}", "", n, ""]
(HERE / "NOTES.md").write_text("\n".join(md))

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto((HERE / "talk.html").resolve().as_uri()); pg.wait_for_timeout(2500)
    over = pg.evaluate("""() => [...document.querySelectorAll('section')].map(s => ({id: s.id,
        overflow: s.scrollHeight - s.clientHeight, wide: s.scrollWidth - s.clientWidth}))""")
    pg.emulate_media(media="screen")
    pg.pdf(path=str(HERE / "talk.pdf"), width="1920px", height="1080px", print_background=True,
           margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
    b.close()
print(json.dumps(over)); print(f"notes: {words} words")
