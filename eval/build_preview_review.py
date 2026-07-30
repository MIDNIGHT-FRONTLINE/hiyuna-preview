#!/usr/bin/env python3
"""Build the preview review page: usability set (resolution rows x prompt columns) + style swap grid.

  python3 build_preview_review.py --names ep1            # single checkpoint
  python3 build_preview_review.py --names base_0step ep1 ep2   # progressive comparison

Layout
- Usability, per group: one table whose ROWS are (checkpoint x resolution tier) and whose
  COLUMNS are the prompts of that group. Scanning a row compares the group's swap (format /
  rating / tag / usage) at fixed resolution; scanning a column compares resolutions and
  epochs for one prompt. 1536 rows are marked as a pre-polish baseline (outside the base
  native band) — collapse is recorded, not corrected.
- Style swap: per base prompt, a labelled grid over the no-tag control + every style tag
  (1024 only).
Thumbnails are written to outputs/_thumbs/ and the full-resolution PNG is one click away.
"""
import argparse
import html
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = f"{HERE}/outputs"
THUMBS = f"{OUTPUTS}/_thumbs"
THUMB_H = 300


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", required=True, help="checkpoint output dirs, in display order")
    p.add_argument("--usability_json", default=f"{HERE}/usability_set_v2.json")
    p.add_argument("--style_swap_json", default=f"{HERE}/style_swap_set.json")
    p.add_argument("--out", default=None, help="output html (default outputs/review_<names>.html)")
    p.add_argument("--thumb_h", type=int, default=THUMB_H)
    return p.parse_args()


def thumb(src_rel, thumb_h):
    """Make/reuse a thumbnail; returns (thumb_rel_url, full_rel_url) relative to OUTPUTS, or None."""
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
    return dst_rel, src_rel


def cell(src_rel, thumb_h, caption=""):
    t = thumb(src_rel, thumb_h)
    if t is None:
        return '<td class="miss"><div class="ph">missing</div></td>'
    trel, frel = t
    cap = f'<div class="cap">{html.escape(caption)}</div>' if caption else ""
    return (f'<td><a href="{html.escape(trel.replace(os.sep, "/"))}" '
            f'data-full="{html.escape(frel.replace(os.sep, "/"))}">'
            f'<img src="{html.escape(trel.replace(os.sep, "/"))}" loading="lazy"></a>{cap}</td>')


CSS = """
:root { --bg:#f7f7f8; --fg:#16161a; --mut:#6b6b75; --line:#d9d9e0; --card:#fff; --warn:#8a4b00; --warnbg:#fff4e5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#101014; --fg:#e9e9ef; --mut:#9a9aa6; --line:#2c2c36; --card:#17171d; --warn:#ffcf8f; --warnbg:#2a1e0c; }
}
* { box-sizing:border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 6px; }
h3 { font-size:14px; margin:20px 0 6px; color:var(--mut); font-weight:600; }
.sub { color:var(--mut); margin:0 0 18px; }
.note { background:var(--warnbg); color:var(--warn); border-left:3px solid currentColor;
  padding:8px 12px; margin:12px 0; border-radius:0 6px 6px 0; }
.scroll { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; }
table { border-collapse:separate; border-spacing:8px; }
th { font-size:11px; color:var(--mut); font-weight:600; text-align:left; vertical-align:bottom;
  white-space:nowrap; padding:0 4px; }
th.rowh { text-align:right; padding-right:8px; white-space:nowrap; vertical-align:middle; }
th.rowh .tier { font-size:13px; color:var(--fg); font-weight:700; display:block; }
th.rowh.oob .tier { color:var(--warn); }
td { padding:0; vertical-align:top; }
td img { display:block; border-radius:6px; border:1px solid var(--line); max-width:none; }
td.miss .ph { width:200px; height:120px; display:grid; place-items:center; color:var(--mut);
  border:1px dashed var(--line); border-radius:6px; font-size:11px; }
.cap { font-size:10px; color:var(--mut); margin-top:3px; }
.prompts { margin:10px 0 0; }
.prompts details { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:8px 12px; margin:6px 0; }
.prompts summary { cursor:pointer; font-weight:600; font-size:12px; }
.prompts p { margin:6px 0 0; color:var(--mut); font-size:12px; white-space:pre-wrap; word-break:break-word; }
.meta { font-size:12px; color:var(--mut); }
.grid { display:flex; flex-wrap:wrap; gap:10px; }
.gcell { text-align:center; }
.gcell .lbl { font-size:11px; color:var(--mut); margin-top:3px; max-width:220px; }
.gcell.ctrl .lbl { color:var(--fg); font-weight:700; }
"""


def main():
    args = parse_args()
    uspec = json.load(open(args.usability_json))
    sspec = json.load(open(args.style_swap_json))
    umeta, smeta = uspec["meta"], sspec["meta"]
    tiers = umeta["resolution_tiers"]
    names = args.names

    groups = {}
    for p in uspec["prompts"]:
        groups.setdefault(p["group"], []).append(p)

    out = [f"<style>{CSS}</style>",
           f"<h1>Hiyuna preview — review: {html.escape(', '.join(names))}</h1>",
           f'<p class="sub">usability set <b>{html.escape(umeta["version"])}</b> '
           f'({umeta["total_prompts"]} prompts x {len(tiers)} tiers) · style swap '
           f'<b>{html.escape(smeta["version"])}</b> · steps {umeta["inference_defaults"]["steps"]}, '
           f'cfg {umeta["inference_defaults"]["cfg"]} · fixed per-prompt seeds</p>',
           f'<div class="note"><b>1536 rows = pre-polish baseline.</b> '
           f'{html.escape(umeta["resolution_note"])}</div>']

    # ---------- usability ----------
    out.append("<h2>Usability set v2</h2>")
    for gname, prompts in groups.items():
        out.append(f"<h3>{html.escape(gname)} — {html.escape(umeta['groups'][gname])}</h3>")
        out.append('<div class="scroll"><table>')
        out.append("<tr><th></th>" + "".join(
            f'<th>{html.escape(p["id"])}<br><span class="meta">seed {p["seed"]} · '
            f'{p["aspect"][0]}:{p["aspect"][1]}</span></th>' for p in prompts) + "</tr>")
        for name in names:
            for tier in tiers:
                oob = " oob" if tier > 1024 else ""
                label = (f'<th class="rowh{oob}"><span class="tier">{tier}</span>'
                         f'<span class="meta">{html.escape(name)}'
                         f'{" · out-of-band" if oob else ""}</span></th>')
                row = "".join(cell(f"{name}/usability/{p['id']}_{tier}.png", args.thumb_h)
                              for p in prompts)
                out.append(f"<tr>{label}{row}</tr>")
        out.append("</table></div>")
        out.append('<div class="prompts">')
        for p in prompts:
            out.append(f'<details><summary>{html.escape(p["id"])} — {html.escape(p["label"])}</summary>'
                       f'<p>{html.escape(p["prompt"])}</p></details>')
        out.append("</div>")

    # ---------- style swap ----------
    out.append("<h2>Style swap set (1024 only)</h2>")
    out.append(f'<p class="sub">{html.escape(smeta["design"])} · {html.escape(smeta["seed_policy"])}</p>')
    for name in names:
        for base in smeta["base_prompts"]:
            out.append(f'<h3>{html.escape(name)} · base "{html.escape(base["key"])}" '
                       f'(seed {base["seed"]}) — {html.escape(base["note"])}</h3>')
            out.append('<div class="grid">')
            for cond in ["notag"] + smeta["style_tags"]:
                slug = cond.replace(" ", "_").replace("'", "")
                rel = f"{name}/style_swap/{base['key']}__{slug}.png"
                t = thumb(rel, args.thumb_h)
                is_ctrl = cond == "notag"
                lbl = "no tag (control)" if is_ctrl else cond
                if t is None:
                    inner = '<div class="ph" style="width:200px;height:120px"></div>'
                else:
                    trel, frel = t
                    inner = (f'<a href="{trel.replace(os.sep, "/")}"><img src="{trel.replace(os.sep, "/")}" '
                             f'loading="lazy" style="border-radius:6px;border:1px solid var(--line)"></a>')
                out.append(f'<div class="gcell{" ctrl" if is_ctrl else ""}">{inner}'
                           f'<div class="lbl">{html.escape(lbl)}</div></div>')
            out.append("</div>")
            out.append(f'<div class="prompts"><details><summary>template</summary>'
                       f'<p>{html.escape(base["template"])}</p></details></div>')

    # ---------- generation logs summary ----------
    out.append("<h2>Generation logs</h2><div class=\"prompts\">")
    for name in names:
        for set_name in ("usability", "style_swap"):
            lp = f"{OUTPUTS}/{name}/generation_log_{set_name}.json"
            if not os.path.exists(lp):
                continue
            lg = json.load(open(lp))
            imgs = lg.get("images", {})
            secs = [v["sec"] for v in imgs.values() if "sec" in v]
            peaks = [v.get("peak_vram_alloc_gib", 0) for v in imgs.values()]
            out.append(f'<details><summary>{html.escape(name)} / {set_name} — {len(imgs)} images'
                       f'{f", {sum(secs)/60:.0f} min total, peak {max(peaks):.1f} GiB" if secs else ""}</summary>'
                       f'<p>dit: {html.escape(str(lg.get("dit")))}\nlora: {html.escape(str(lg.get("lora")))}\n'
                       f'set version: {html.escape(str(lg.get("eval_set_version")))}\n'
                       f'updated: {html.escape(str(lg.get("updated_at")))}</p></details>')
    out.append("</div>")

    out_path = args.out or f"{OUTPUTS}/review_{'_'.join(names)}.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
