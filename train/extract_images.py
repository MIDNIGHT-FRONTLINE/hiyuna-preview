#!/usr/bin/env python3
"""Extract the keep-set images from the HF curation package to a flat training tree.

Body parquet carries `image_bytes` = FULL-RES original webp bytes, written out verbatim
(no re-encode) as <id>.webp. The filename stem becomes the trainer's item_key, which is
what krea2_caption_synth looks up in preview_materials.sqlite (keyed by str(id)) — so the
flat <id>.webp naming is what wires pixels to captions.

Output: ~/Desktop/ft-preview/train/dataset/images/<id>.webp  (9439 files)
"""
import glob
import os
import sys

import pyarrow.parquet as pq

PKG = os.path.expanduser("~/Desktop/polish/hf_pkg")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "images")


def main():
    os.makedirs(OUT, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(PKG, "data", "part-*.parquet")))
    assert shards, "no body shards found"
    n_written = n_skipped = 0
    total_bytes = 0
    for sh in shards:
        t = pq.read_table(sh, columns=["id", "image_bytes"])
        ids = t.column("id").to_pylist()
        blobs = t.column("image_bytes").to_pylist()
        for pid, blob in zip(ids, blobs):
            path = os.path.join(OUT, f"{pid}.webp")
            if os.path.exists(path) and os.path.getsize(path) == len(blob):
                n_skipped += 1
                continue
            with open(path, "wb") as f:
                f.write(blob)
            n_written += 1
            total_bytes += len(blob)
        print(f"  {os.path.basename(sh)}: done (written={n_written} skipped={n_skipped})", flush=True)
    n_files = len(glob.glob(os.path.join(OUT, "*.webp")))
    print(f"extracted {n_written} (skipped {n_skipped} existing), {total_bytes/2**30:.2f} GiB new")
    print(f"total files in {OUT}: {n_files}")
    if n_files != 9439:
        print(f"WARNING: expected 9439 files, found {n_files}")
        sys.exit(1)


if __name__ == "__main__":
    main()
