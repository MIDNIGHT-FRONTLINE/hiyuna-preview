#!/usr/bin/env python3
"""Pick a reference set of keep images per style tag — the destination the dials aim at.

For each style attribute, take the keep images that carry it and choose the best-scoring
few, at most `--per-artist` from any one artist so a single prolific artist cannot define
the tag on its own. The result is what "moving toward keep" is measured against; without
it, the only available statement is "the output moved", not "it moved somewhere".

Writes eval/style_refs.json: tag -> [{id, artist, score_pr, path}]. Contains post ids and
artist names, so it stays out of the published repo (see .gitignore).
"""
import argparse
import glob
import json
import os
import sqlite3

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW = os.path.dirname(HERE)
PKG = os.path.expanduser("~/Desktop/polish/hf_pkg")
IMAGES = f"{PREVIEW}/train/dataset/images"
MATERIALS = f"{PREVIEW}/caption/preview_materials.sqlite"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=6, help="references per tag (target)")
    p.add_argument("--min-n", type=int, default=4, help="warn if a tag cannot reach this")
    p.add_argument("--per-artist", type=int, default=2, help="max references from one artist")
    p.add_argument("--out", default=f"{HERE}/style_refs.json")
    return p.parse_args()


def main():
    args = parse_args()

    # score_pr lives in the body parquet (the materials DB omits it), so read it back
    score_pr, = [{}],
    score_pr = {}
    for sh in sorted(glob.glob(os.path.join(PKG, "data", "part-*.parquet"))):
        t = pq.read_table(sh, columns=["id", "score_pr"]).to_pydict()
        score_pr.update({int(i): (float(s) if s is not None else 0.0)
                         for i, s in zip(t["id"], t["score_pr"])})

    db = sqlite3.connect(f"file:{MATERIALS}?mode=ro", uri=True)
    rows = db.execute("select id, style_tags, tag_string_artist from captions").fetchall()
    db.close()

    by_tag = {}
    for pid, stj, artist in rows:
        tags = json.loads(stj) if stj else []
        for t in tags:
            by_tag.setdefault(t, []).append(
                {"id": int(pid), "artist": (artist or "").strip(), "score_pr": score_pr.get(int(pid), 0.0)}
            )

    out, report = {}, []
    for tag in sorted(by_tag):
        cands = sorted(by_tag[tag], key=lambda r: -r["score_pr"])
        picked, per_artist = [], {}
        for r in cands:
            a = r["artist"] or f"__unknown_{r['id']}"
            if per_artist.get(a, 0) >= args.per_artist:
                continue
            path = os.path.join(IMAGES, f"{r['id']}.webp")
            if not os.path.exists(path):
                continue
            per_artist[a] = per_artist.get(a, 0) + 1
            picked.append({**r, "path": path})
            if len(picked) >= args.n:
                break
        out[tag] = picked
        report.append((tag, len(picked), len(cands), len({p["artist"] for p in picked})))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_artist_cap": args.per_artist, "target_n": args.n, "refs": out}, f,
                  ensure_ascii=False, indent=1)

    print(f"{'tag':<20}{'picked':>7}{'pool':>7}{'artists':>9}")
    for tag, n, pool, na in report:
        flag = "  <- thin" if n < args.min_n else ""
        print(f"{tag:<20}{n:>7}{pool:>7}{na:>9}{flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
