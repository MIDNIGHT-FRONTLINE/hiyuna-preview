#!/usr/bin/env python3
"""How much did the model still move between epochs? (under-training vs converged)

Every checkpoint renders the same frozen prompts at the same seeds, so the images are
directly comparable frame to frame. This measures the size of the change between
consecutive checkpoints:

  structure  RMS difference of 256px grayscale, 0..1 — composition / linework movement
  palette    L1 distance of 32-bin per-channel histograms, 0..1 — colour and tone movement

Read the TREND, not the absolute value. If d(ep2,ep3) is close to d(ep1,ep2) the model was
still moving when the run ended, which is what under-training looks like. If it is much
smaller, the run is converging and more epochs buy less. This says nothing about whether the
movement is an improvement — that judgement is the review page's job.

  python3 convergence_delta.py --names ep1 ep2 ep3
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = f"{HERE}/outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="+", required=True, help="checkpoint dirs in run order")
    p.add_argument("--outputs", default=OUTPUTS)
    p.add_argument("--json_out", default=None, help="also write the numbers as JSON")
    return p.parse_args()


def load(path, size=256):
    im = Image.open(path).convert("RGB")
    small = im.resize((size, size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hist = np.concatenate([
        np.histogram(arr[..., c], bins=32, range=(0, 1))[0] for c in range(3)
    ]).astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    return gray, hist


def main():
    args = parse_args()
    names = args.names
    if len(names) < 2:
        raise SystemExit("need at least two checkpoints to measure a delta")

    # index images per checkpoint: (set, id) -> path
    index = {}
    for name in names:
        for set_name in ("usability", "style_swap"):
            d = os.path.join(args.outputs, name, set_name)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".png"):
                    index[(name, set_name, fn[:-4])] = os.path.join(d, fn)

    keys = sorted({k[1:] for k in index if k[0] == names[0]})
    common = [k for k in keys if all((n,) + k in index for n in names)]
    print(f"comparable images present in all of {names}: {len(common)}")
    if not common:
        raise SystemExit("no images common to every checkpoint yet")

    cache = {}
    def feats(name, key):
        ck = (name,) + key
        if ck not in cache:
            cache[ck] = load(index[ck])
        return cache[ck]

    pairs = list(zip(names, names[1:]))
    # per pair: overall + per group (usability id prefix) + per set
    agg = {p: defaultdict(lambda: [[], []]) for p in pairs}
    for key in common:
        set_name, pid = key
        group = set_name if set_name == "style_swap" else pid[0]  # F / R / T / U
        for a, b in pairs:
            ga, ha = feats(a, key)
            gb, hb = feats(b, key)
            struct = float(np.sqrt(np.mean((ga - gb) ** 2)))
            pal = float(np.abs(ha - hb).sum() / 2.0)
            for bucket in ("ALL", group):
                agg[(a, b)][bucket][0].append(struct)
                agg[(a, b)][bucket][1].append(pal)

    label = {"F": "format equivalence", "R": "rating ladder", "T": "tag isolation",
             "U": "real usage", "style_swap": "style swap", "ALL": "ALL"}
    rows = {}
    print(f"\n{'bucket':22s} " + "  ".join(f"{a}->{b:<14s}" for a, b in pairs))
    print(f"{'':22s} " + "  ".join("struct / palette".ljust(19) for _ in pairs))
    for bucket in ["ALL", "F", "R", "T", "U", "style_swap"]:
        if not any(bucket in agg[p] for p in pairs):
            continue
        cells = []
        for p in pairs:
            s, q = agg[p].get(bucket, [[], []])
            if not s:
                cells.append("—".ljust(19))
                continue
            cells.append(f"{np.mean(s):.4f} / {np.mean(q):.4f}".ljust(19))
            rows.setdefault(bucket, {})[f"{p[0]}->{p[1]}"] = {
                "structure": round(float(np.mean(s)), 5),
                "palette": round(float(np.mean(q)), 5),
                "n": len(s),
            }
        print(f"{label[bucket]:22s} " + "  ".join(cells))

    if len(pairs) >= 2:
        first, last = pairs[0], pairs[-1]
        s0 = np.mean(agg[first]["ALL"][0])
        s1 = np.mean(agg[last]["ALL"][0])
        ratio = s1 / s0 if s0 else float("nan")
        print(f"\nmovement ratio (last pair / first pair, structure): {ratio:.2f}")
        print("  ~1.0 or higher -> still moving at the end (under-trained)")
        print("  well below 1.0 -> converging; further epochs buy less")
        rows.setdefault("_summary", {})["movement_ratio_structure"] = round(float(ratio), 4)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
