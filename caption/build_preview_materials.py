#!/usr/bin/env python3
"""Build the preview caption-synth materials sqlite from the HF curation package.

Source: ~/Desktop/polish/hf_pkg  (mirror of H4RUming/hiyuna-curation-internal)
  data/part-*.parquet  -> body, 9439 keep rows (1 per image)
  style_tags.parquet   -> SIDECAR: id -> style_tags list<string> (+ version)

Output: preview_materials.sqlite with table `captions` in the column names the
synthesizer expects (tag_string_*), plus a JSON `style_tags` column and a `meta`
table recording the sidecar version (logged at train time). score/category are
omitted (unused by the preview profile); year is kept for reference only.

The body column names differ from the full-FT captions.db (tag_general vs
tag_string_general, no score/category) — this script does the rename+join so the
unchanged CaptionSynthesizer loader reads it directly.
"""
import glob
import json
import os
import sqlite3
import sys

import pyarrow.parquet as pq

PKG = os.path.expanduser("~/Desktop/polish/hf_pkg")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_materials.sqlite")

BODY_COLS = ["id", "natural_caption", "tag_general", "tag_character",
             "tag_copyright", "tag_artist", "rating", "year"]
RENAME = {
    "tag_general": "tag_string_general",
    "tag_character": "tag_string_character",
    "tag_copyright": "tag_string_copyright",
    "tag_artist": "tag_string_artist",
}


def main():
    # ---- sidecar: id -> style_tags, + version ----
    st = pq.read_table(os.path.join(PKG, "style_tags.parquet")).to_pandas()
    versions = set(st["version"].tolist())
    if len(versions) != 1:
        print(f"WARNING: style_tags sidecar has {len(versions)} versions: {versions}")
    sidecar_version = sorted(versions)[-1]
    style_of = {int(r.id): list(r.style_tags) for r in st.itertuples()}
    n_styled = sum(1 for v in style_of.values() if v)
    print(f"sidecar version={sidecar_version}  rows={len(style_of)}  non-empty style_tags={n_styled}")

    # ---- body: 7 shards -> rows ----
    shards = sorted(glob.glob(os.path.join(PKG, "data", "part-*.parquet")))
    assert shards, "no body shards found"
    rows = []
    for sh in shards:
        t = pq.read_table(sh, columns=BODY_COLS).to_pandas()
        for r in t.itertuples(index=False):
            rec = {RENAME.get(c, c): getattr(r, c) for c in BODY_COLS}
            rec["style_tags"] = json.dumps(style_of.get(int(r.id), []), ensure_ascii=False)
            rows.append(rec)
    print(f"body rows={len(rows)} from {len(shards)} shards")

    # every body id must exist in the sidecar (1:1:1 invariant)
    body_ids = {r["id"] for r in rows}
    missing = body_ids - set(style_of)
    assert not missing, f"{len(missing)} body ids missing from sidecar, e.g. {list(missing)[:5]}"

    # ---- write sqlite ----
    if os.path.exists(OUT):
        os.remove(OUT)
    db = sqlite3.connect(OUT)
    db.execute(
        "CREATE TABLE captions (id INTEGER PRIMARY KEY, natural_caption TEXT, "
        "tag_string_general TEXT, tag_string_character TEXT, tag_string_copyright TEXT, "
        "tag_string_artist TEXT, rating TEXT, year INTEGER, style_tags TEXT)"
    )
    cols = ["id", "natural_caption", "tag_string_general", "tag_string_character",
            "tag_string_copyright", "tag_string_artist", "rating", "year", "style_tags"]
    db.executemany(
        f"INSERT INTO captions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows],
    )
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    db.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("sidecar_version", sidecar_version),
        ("n_rows", str(len(rows))),
        ("n_styled", str(n_styled)),
        ("source_pkg", PKG),
    ])
    db.commit()

    # sanity: NULL captions, refused (should be 0/none per README)
    n_null = db.execute("SELECT count(*) FROM captions WHERE natural_caption IS NULL OR natural_caption=''").fetchone()[0]
    db.close()
    print(f"wrote {OUT}  ({len(rows)} rows, {n_null} null/empty captions)")
    if n_null:
        print(f"WARNING: {n_null} rows have no natural_caption")


if __name__ == "__main__":
    main()
