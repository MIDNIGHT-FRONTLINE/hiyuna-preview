#!/usr/bin/env python3
"""Are the style dials moving TOWARD the keep set, per tag? (destination metric)

"Moved from base" only says something changed. This measures distance to the destination:
each checkpoint's style-swap output for a tag is compared against that tag's reference set
of keep images (built by build_style_refs.py), across checkpoints, as a time series.

Two distances, per output/reference pair, averaged over the reference set:
  palette   L1 between 32-bin per-channel histograms, 0..1
  dino      cosine distance between DINOv2 CLS embeddings, 0..1

Generated images and references show different subjects, so the absolute level is not
meaningful — the trend across checkpoints is. Two controls make it falsifiable:

  no-tag control  the same checkpoint's untagged output measured against the same
                  references. If the tagged output approaches its references no faster
                  than the untagged one, nothing tag-specific happened.
  off-diagonal    the tagged output measured against OTHER tags' references. Approaching
                  every reference set equally is a global quality drift, not a dial. The
                  discriminative quantity is (off-diagonal - own), which should grow.

  python3 destination_delta.py --names base_0step ep1 ep2 ep3 ep4 ep5
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = f"{HERE}/outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="+", required=True, help="checkpoint dirs, in run order")
    p.add_argument("--refs", default=f"{HERE}/style_refs.json")
    p.add_argument("--bases", nargs="+", default=["simple", "scene"],
                   help="style-swap base prompts to average over")
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json_out", default=f"{OUTPUTS}/destination_delta.json")
    p.add_argument("--no_dino", action="store_true", help="palette only (skip the embedding model)")
    return p.parse_args()


def slug(tag):
    return tag.replace(" ", "_").replace("'", "")


def palette(path, bins=32):
    a = np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.LANCZOS),
                   dtype=np.float32) / 255.0
    h = np.concatenate([np.histogram(a[..., c], bins=bins, range=(0, 1))[0] for c in range(3)])
    h = h.astype(np.float32)
    return h / max(h.sum(), 1.0)


class Dino:
    def __init__(self, name, device):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(device)
        self.device = device
        self.cache = {}

    def __call__(self, path):
        if path not in self.cache:
            im = Image.open(path).convert("RGB")
            with torch.no_grad():
                out = self.model(**{k: v.to(self.device)
                                    for k, v in self.proc(images=im, return_tensors="pt").items()})
            e = out.last_hidden_state[:, 0].float().cpu().numpy()[0]
            self.cache[path] = e / (np.linalg.norm(e) + 1e-8)
        return self.cache[path]


def main():
    args = parse_args()
    refs = json.load(open(args.refs))["refs"]
    tags = sorted(refs)
    dino = None if args.no_dino else Dino(args.model, args.device)

    # reference features
    ref_feat = {}
    for t in tags:
        ref_feat[t] = {
            "pal": [palette(r["path"]) for r in refs[t]],
            "dino": None if dino is None else [dino(r["path"]) for r in refs[t]],
        }

    def dist(out_path, tag):
        """mean palette / dino distance from one output image to a tag's reference set"""
        p = palette(out_path)
        pal = float(np.mean([np.abs(p - q).sum() / 2 for q in ref_feat[tag]["pal"]]))
        dn = None
        if dino is not None:
            e = dino(out_path)
            dn = float(np.mean([1.0 - float(e @ q) for q in ref_feat[tag]["dino"]]))
        return pal, dn

    results = {}
    for name in args.names:
        d = f"{OUTPUTS}/{name}/style_swap"
        if not os.path.isdir(d):
            print(f"skip {name}: no style_swap outputs")
            continue
        per_tag = {}
        for t in tags:
            own_p, own_d, off_p, off_d, ctl_p, ctl_d = [], [], [], [], [], []
            for b in args.bases:
                tagged = f"{d}/{b}__{slug(t)}.png"
                untagged = f"{d}/{b}__notag.png"
                if not (os.path.exists(tagged) and os.path.exists(untagged)):
                    continue
                p, dn = dist(tagged, t)
                own_p.append(p)
                if dn is not None:
                    own_d.append(dn)
                p, dn = dist(untagged, t)          # no-tag control vs the same references
                ctl_p.append(p)
                if dn is not None:
                    ctl_d.append(dn)
                for other in tags:                  # tagged output vs other tags' references
                    if other == t:
                        continue
                    p, dn = dist(tagged, other)
                    off_p.append(p)
                    if dn is not None:
                        off_d.append(dn)
            if not own_p:
                continue
            per_tag[t] = {
                "own_palette": round(float(np.mean(own_p)), 5),
                "own_dino": round(float(np.mean(own_d)), 5) if own_d else None,
                "notag_palette": round(float(np.mean(ctl_p)), 5),
                "notag_dino": round(float(np.mean(ctl_d)), 5) if ctl_d else None,
                "off_palette": round(float(np.mean(off_p)), 5),
                "off_dino": round(float(np.mean(off_d)), 5) if off_d else None,
            }
        results[name] = per_tag
        print(f"computed {name}: {len(per_tag)} tags")

    # ---- report ----
    names = [n for n in args.names if n in results]
    key = "own_dino" if not args.no_dino else "own_palette"
    print(f"\n=== keep 참조까지의 거리 (own {'DINOv2' if not args.no_dino else 'palette'}) — 낮을수록 접근 ===")
    print(f"{'tag':<20}" + "".join(f"{n:>11}" for n in names) + f"{'최종-최초':>11}")
    for t in tags:
        row = [results[n].get(t, {}).get(key) for n in names]
        if any(v is None for v in row):
            continue
        delta = row[-1] - row[0]
        arrow = "접근" if delta < -0.002 else ("이탈" if delta > 0.002 else "정체")
        print(f"{t:<20}" + "".join(f"{v:>11.4f}" for v in row) + f"{delta:>+9.4f} {arrow}")

    print(f"\n=== 판별력: (다른 태그 참조까지의 거리) − (자기 태그 참조까지의 거리) — 클수록 다이얼이 특이적 ===")
    print(f"{'tag':<20}" + "".join(f"{n:>11}" for n in names))
    okey, offkey = ("own_dino", "off_dino") if not args.no_dino else ("own_palette", "off_palette")
    for t in tags:
        row = []
        for n in names:
            r = results[n].get(t, {})
            row.append(None if r.get(okey) is None else r[offkey] - r[okey])
        if any(v is None for v in row):
            continue
        print(f"{t:<20}" + "".join(f"{v:>+11.4f}" for v in row))

    print(f"\n=== 태그 효과: (무태그 출력 거리) − (태그 출력 거리) — 양수면 태그가 참조 쪽으로 끌어당김 ===")
    print(f"{'tag':<20}" + "".join(f"{n:>11}" for n in names))
    nkey = "notag_dino" if not args.no_dino else "notag_palette"
    for t in tags:
        row = []
        for n in names:
            r = results[n].get(t, {})
            row.append(None if r.get(okey) is None else r[nkey] - r[okey])
        if any(v is None for v in row):
            continue
        print(f"{t:<20}" + "".join(f"{v:>+11.4f}" for v in row))

    with open(args.json_out, "w") as f:
        json.dump({"metric": "distance to keep reference set", "names": names,
                   "model": None if args.no_dino else args.model, "results": results}, f, indent=1)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
