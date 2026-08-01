#!/usr/bin/env python3
"""Style-only distance to the keep reference sets — Gram / patch statistics.

destination_delta.py uses DINOv2 CLS embeddings, which describe the whole image and are
therefore dominated by subject and layout when the two images show different things — which
is always the case here, since generated outputs and keep references share no content. This
computes descriptors that discard spatial arrangement, so what survives is texture, palette,
brushwork and line quality: style.

  vgg_gram    Gram matrices of VGG19 relu1_1..relu5_1 feature maps (the Gatys style
              representation), each normalised, cosine distance averaged over layers.
  dino_gram   Gram matrix of L2-normalised DINOv2 patch tokens.
  dino_stat   concat(mean, std) of DINOv2 patch tokens over positions (AdaIN-style
              descriptor).

Same controls as the destination metric: the untagged output against the same references,
and the tagged output against other tags' references.

  python3 style_distance.py --names base_0step ep1 ep2 ep3 ep4 ep5
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = f"{HERE}/outputs"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VGG_STYLE_LAYERS = [1, 6, 11, 20, 29]  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--refs", default=f"{HERE}/style_refs.json")
    p.add_argument("--bases", nargs="+", default=["simple", "scene"])
    p.add_argument("--size", type=int, default=384, help="square resize before featurising")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json_out", default=f"{OUTPUTS}/style_distance.json")
    return p.parse_args()


def slug(tag):
    return tag.replace(" ", "_").replace("'", "")


class Featurizer:
    def __init__(self, device, size):
        import torchvision
        from transformers import AutoImageProcessor, AutoModel
        self.device, self.size = device, size
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.vgg = vgg.features[: max(VGG_STYLE_LAYERS) + 1].eval().to(device)
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
        self.cache = {}

    @staticmethod
    def _gram(feat):
        """feat (C,H,W) -> normalised Gram, flattened and unit-normed."""
        c = feat.shape[0]
        f = feat.reshape(c, -1)
        g = (f @ f.T) / f.shape[1]
        g = g.flatten()
        return (g / (g.norm() + 1e-8)).cpu().numpy()

    def __call__(self, path):
        if path in self.cache:
            return self.cache[path]
        im = Image.open(path).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(self.device)
        grams = []
        h = x
        with torch.no_grad():
            for i, layer in enumerate(self.vgg):
                h = layer(h)
                if i in VGG_STYLE_LAYERS:
                    grams.append(self._gram(h[0]))
            d = self.dino(**{k: v.to(self.device)
                             for k, v in self.dproc(images=im, return_tensors="pt").items()})
            patches = d.last_hidden_state[0, 1:]                     # (N, 768)
            patches = F.normalize(patches, dim=-1)
            dg = (patches.T @ patches) / patches.shape[0]
            dg = dg.flatten()
            dg = (dg / (dg.norm() + 1e-8)).cpu().numpy()
            raw = d.last_hidden_state[0, 1:]
            stat = torch.cat([raw.mean(0), raw.std(0)]).cpu().numpy()
            stat = stat / (np.linalg.norm(stat) + 1e-8)
        self.cache[path] = {"vgg_gram": grams, "dino_gram": dg, "dino_stat": stat}
        return self.cache[path]


def dist(a, b):
    return {
        "vgg_gram": float(np.mean([1.0 - float(x @ y) for x, y in zip(a["vgg_gram"], b["vgg_gram"])])),
        "dino_gram": float(1.0 - float(a["dino_gram"] @ b["dino_gram"])),
        "dino_stat": float(1.0 - float(a["dino_stat"] @ b["dino_stat"])),
    }


def main():
    args = parse_args()
    refs = json.load(open(args.refs))["refs"]
    tags = sorted(refs)
    fz = Featurizer(args.device, args.size)

    ref_feat = {t: [fz(r["path"]) for r in refs[t]] for t in tags}
    print(f"featurised references: {sum(len(v) for v in ref_feat.values())} images")

    def to_refs(out_path, tag):
        f = fz(out_path)
        ds = [dist(f, r) for r in ref_feat[tag]]
        return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}

    metrics = ["vgg_gram", "dino_gram", "dino_stat"]
    results = {}
    for name in args.names:
        d = f"{OUTPUTS}/{name}/style_swap"
        if not os.path.isdir(d):
            print(f"skip {name}")
            continue
        per_tag = {}
        for t in tags:
            acc = {f"{p}_{m}": [] for p in ("own", "notag", "off") for m in metrics}
            for b in args.bases:
                tagged, untagged = f"{d}/{b}__{slug(t)}.png", f"{d}/{b}__notag.png"
                if not (os.path.exists(tagged) and os.path.exists(untagged)):
                    continue
                for m, v in to_refs(tagged, t).items():
                    acc[f"own_{m}"].append(v)
                for m, v in to_refs(untagged, t).items():
                    acc[f"notag_{m}"].append(v)
                for other in tags:
                    if other == t:
                        continue
                    for m, v in to_refs(tagged, other).items():
                        acc[f"off_{m}"].append(v)
            if acc["own_vgg_gram"]:
                per_tag[t] = {k: round(float(np.mean(v)), 5) for k, v in acc.items() if v}
        results[name] = per_tag
        print(f"computed {name}: {len(per_tag)} tags")

    names = [n for n in args.names if results.get(n)]
    from collections import Counter
    for m in metrics:
        print(f"\n=== {m}: keep 참조까지의 화풍 거리 (낮을수록 접근) ===")
        print(f"{'tag':<20}" + "".join(f"{n:>11}" for n in names) + f"{'최소':>12}")
        nearest = Counter()
        for t in tags:
            row = [results[n].get(t, {}).get(f"own_{m}") for n in names]
            if any(v is None for v in row):
                continue
            best = names[int(np.argmin(row))]
            nearest[best] += 1
            print(f"{t:<20}" + "".join(f"{v:>11.4f}" for v in row) + f"{best:>12}")
        print(f"  최근접 집계: " + ", ".join(f"{k}={v}" for k, v in
                                          sorted(nearest.items(), key=lambda x: names.index(x[0]))))

    print(f"\n=== 판별력 (off − own), vgg_gram — 클수록 다이얼이 화풍 특이적 ===")
    print(f"{'tag':<20}" + "".join(f"{n:>11}" for n in names))
    for t in tags:
        row = []
        for n in names:
            r = results[n].get(t, {})
            row.append(None if "own_vgg_gram" not in r else r["off_vgg_gram"] - r["own_vgg_gram"])
        if any(v is None for v in row):
            continue
        print(f"{t:<20}" + "".join(f"{v:>+11.4f}" for v in row))

    with open(args.json_out, "w") as f:
        json.dump({"metric": "style-only distance to keep references",
                   "names": names, "results": results}, f, indent=1)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
