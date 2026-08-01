#!/usr/bin/env python3
"""Build a comparison page for an ad-hoc prompt set: rows = checkpoints, columns = prompts.

  python3 build_adhoc_page.py --set rating_ladder_neutral.json --names ep2 ep3 ep4 ep5

Reading across a row gives the dial's behaviour at one checkpoint; reading down a column
shows how that single condition evolved over training. Ad-hoc sets live in their own output
subdirectory (meta.set_name), so they never mix with the frozen set.
"""
import argparse
import html
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = f"{HERE}/outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="set_json", required=True, help="ad-hoc prompt set JSON")
    p.add_argument("--names", nargs="+", required=True, help="checkpoint dirs, in run order")
    p.add_argument("--out", default=None)
    p.add_argument("--thumb_h", type=int, default=340)
    return p.parse_args()


def thumb(src_rel, thumb_h):
    src = os.path.join(OUTPUTS, src_rel)
    if not os.path.exists(src):
        return None
    dst_rel = os.path.join("_thumbs", src_rel.replace(os.sep, "__"))
    dst = os.path.join(OUTPUTS, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        im = Image.open(src)
        w = max(1, round(im.width * thumb_h / im.height))
        im.convert("RGB").resize((w, thumb_h), Image.LANCZOS).save(dst, "JPEG", quality=88)
    return dst_rel


CSS = """
:root { --bg:#f7f7f8; --fg:#16161a; --mut:#6b6b75; --line:#d9d9e0; --card:#fff; --accent:#8a4b00; --accentbg:#fff4e5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#101014; --fg:#e9e9ef; --mut:#9a9aa6; --line:#2c2c36; --card:#17171d; --accent:#ffcf8f; --accentbg:#2a1e0c; }
}
* { box-sizing:border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:21px; margin:0 0 4px; }
.sub { color:var(--mut); margin:0 0 16px; max-width:70ch; }
.note { background:var(--accentbg); color:var(--accent); border-left:3px solid currentColor;
  padding:8px 12px; margin:12px 0; border-radius:0 6px 6px 0; max-width:80ch; }
.scroll { overflow-x:auto; background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:10px; }
table { border-collapse:separate; border-spacing:8px; }
th { font-size:11px; color:var(--mut); font-weight:600; text-align:left; vertical-align:bottom;
  white-space:nowrap; padding:0 4px; }
th.rowh { text-align:right; padding-right:8px; vertical-align:middle; }
th.rowh .ck { font-size:14px; color:var(--fg); font-weight:700; display:block; }
td { padding:0; vertical-align:top; }
td img { display:block; border-radius:6px; border:1px solid var(--line); max-width:none; }
td.miss .ph { width:200px; height:260px; display:grid; place-items:center; color:var(--mut);
  border:1px dashed var(--line); border-radius:6px; font-size:11px; }
.prompts { margin:12px 0 0; }
.prompts details { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:8px 12px; margin:6px 0; }
.prompts summary { cursor:pointer; font-weight:600; font-size:12px; }
.prompts p { margin:6px 0 0; color:var(--mut); font-size:12px; white-space:pre-wrap; word-break:break-word; }
"""


def main():
    args = parse_args()
    spec = json.load(open(args.set_json if os.path.isabs(args.set_json)
                          else os.path.join(HERE, args.set_json)))
    meta = spec["meta"]
    prompts = spec["prompts"]
    set_dir = meta.get("set_name", "adhoc")
    tier = meta.get("default_tier", meta.get("resolution_tiers", [1024])[0])

    out = [f"<style>{CSS}</style>",
           f"<h1>{html.escape(meta['name'])} <span style='font-size:13px;color:var(--mut)'>"
           f"({html.escape(meta['version'])})</span></h1>",
           f'<p class="sub">{html.escape(meta["seed_policy"])} · {tier}px · '
           f'steps {meta["inference_defaults"]["steps"]}, cfg {meta["inference_defaults"]["cfg"]}</p>']
    for g, desc in meta.get("groups", {}).items():
        out.append(f'<div class="note">{html.escape(desc)}</div>')

    out.append('<div class="scroll"><table>')
    out.append("<tr><th></th>" + "".join(
        f'<th>{html.escape(p["id"])}<br><span style="color:var(--fg)">'
        f'{html.escape(p["label"].split("rating: ")[-1])}</span></th>' for p in prompts) + "</tr>")
    for name in args.names:
        cells = []
        for p in prompts:
            rel = f"{name}/{set_dir}/{p['id']}_{tier}.png"
            t = thumb(rel, args.thumb_h)
            if t is None:
                cells.append('<td class="miss"><div class="ph">missing</div></td>')
            else:
                u = t.replace(os.sep, "/")
                cells.append(f'<td><a href="{html.escape(u)}"><img src="{html.escape(u)}" loading="lazy"></a></td>')
        out.append(f'<tr><th class="rowh"><span class="ck">{html.escape(name)}</span></th>'
                   + "".join(cells) + "</tr>")
    out.append("</table></div>")

    out.append('<div class="prompts">')
    for p in prompts:
        out.append(f'<details><summary>{html.escape(p["id"])} — {html.escape(p["label"])}</summary>'
                   f'<p>{html.escape(p["prompt"])}</p></details>')
    out.append(f'<details><summary>notes</summary><p>{html.escape(meta.get("notes",""))}</p></details>')
    out.append("</div>")

    out_path = args.out or f"{OUTPUTS}/review_{set_dir}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
