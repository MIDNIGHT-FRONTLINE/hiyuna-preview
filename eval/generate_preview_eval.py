#!/usr/bin/env python3
"""Preview-run image generator — usability set v2 (3 resolution tiers) + style swap set.

Run with the GPU venv:
  ~/Desktop/diffsynth_research/venv-musubi-gpu/bin/python generate_preview_eval.py \
      --dit <ckpt.safetensors> --name ep1 [--set usability|style_swap|both]

- Prompts/seeds/aspects come frozen from usability_set_v2.json / style_swap_set.json.
- Usability: every prompt at 768 / 1024 / 1536, each tier preserving the prompt's aspect
  (dims = area ~tier^2, rounded to multiples of 16 — the krea2 latent step).
  1536 is outside the base native band and is kept as a pre-polish baseline.
- Style swap: 1024 only, 2 base prompts x (12 style tags + no-tag control).
- Output: outputs/<name>/<set>/<id>.png (+ generation_log_<set>.json with real dims,
  seed, timing, VRAM). Existing PNGs are skipped unless --overwrite, so a killed run resumes.
- The model is loaded once and reused for every image.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import torch

from musubi_tuner.krea2_generate_image import build_pipeline, generate, load_text_encoder

DIFFSYNTH = os.path.expanduser("~/Desktop/diffsynth_research")
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit", required=True, help="DiT checkpoint (.safetensors) — base RAW or an epoch ckpt")
    p.add_argument("--vae", default=f"{DIFFSYNTH}/models_smoke/split_files/vae/qwen_image_vae.safetensors")
    p.add_argument("--text_encoder", default=f"{DIFFSYNTH}/models_smoke/text_encoders/qwen3vl_4b_bf16.safetensors")
    p.add_argument("--name", default=None, help="output subdirectory (e.g. base_0step, ep1). default: dit stem")
    p.add_argument("--set", dest="which", default="both", choices=["usability", "style_swap", "both"])
    p.add_argument("--usability_json", default=f"{HERE}/usability_set_v2.json")
    p.add_argument("--style_swap_json", default=f"{HERE}/style_swap_set.json")
    p.add_argument("--outdir", default=f"{HERE}/outputs")
    p.add_argument("--out_set", default=None,
                   help="output subdirectory for the prompt-set images (default: the set's meta "
                        "set_name, else 'usability'). Keeps ad-hoc sets out of the frozen set's dir.")
    p.add_argument("--tiers", nargs="*", type=int, default=None, help="override resolution tiers (usability)")
    p.add_argument("--ids", nargs="*", default=None, help="only these prompt ids")
    p.add_argument("--attn_mode", default="torch", choices=["torch", "flash", "sageattn", "xformers"])
    p.add_argument("--lora", nargs="*", default=None, help="LoRA weight file(s) to merge (task 3 verification)")
    p.add_argument("--lora_multiplier", nargs="*", type=float, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def dims_for(aspect, tier: int, step: int = 16):
    """Width/height whose area ~= tier^2 at the given aspect ratio, both multiples of `step`."""
    aw, ah = aspect
    ar = aw / ah
    w = (tier * ar**0.5)
    h = w / ar
    w = max(step, int(round(w / step)) * step)
    h = max(step, int(round(h / step)) * step)
    return w, h


def build_jobs(args):
    """[(set_name, id, prompt, seed, width, height, meta_dict), ...]"""
    jobs = []
    if args.which in ("usability", "both"):
        spec = json.load(open(args.usability_json))
        meta = spec["meta"]
        sweep_groups = set(meta.get("sweep_groups", []))
        default_tier = meta.get("default_tier", 1024)
        set_dir = args.out_set or meta.get("set_name", "usability")
        for p in spec["prompts"]:
            # the full resolution sweep applies only to sweep_groups; others get the default tier
            tiers = args.tiers or (meta["resolution_tiers"] if p["group"] in sweep_groups else [default_tier])
            for tier in tiers:
                w, h = dims_for(p["aspect"], tier)
                jobs.append((set_dir, f"{p['id']}_{tier}", p["prompt"], p["seed"], w, h,
                             {"base_id": p["id"], "group": p["group"], "label": p["label"],
                              "tier": tier, "aspect": p["aspect"]}))
    if args.which in ("style_swap", "both"):
        spec = json.load(open(args.style_swap_json))
        meta = spec["meta"]
        tier = meta["resolution_tiers"][0]
        for base in meta["base_prompts"]:
            conditions = [("notag", "")] + [(t, t + ", ") for t in meta["style_tags"]]
            for cond_key, prefix in conditions:
                slug = cond_key.replace(" ", "_").replace("'", "")
                pid = f"{base['key']}__{slug}"
                w, h = dims_for(base["aspect"], tier)
                jobs.append(("style_swap", pid, base["template"].format(style=prefix), base["seed"], w, h,
                             {"base_key": base["key"], "condition": cond_key, "tier": tier}))
    if args.ids:
        want = set(args.ids)
        jobs = [j for j in jobs if j[1] in want or j[6].get("base_id") in want]
    return jobs


def main():
    args = parse_args()
    name = args.name or os.path.splitext(os.path.basename(args.dit))[0]
    jobs = build_jobs(args)

    # defaults (shared by both sets)
    u_spec = json.load(open(args.usability_json))
    defaults = u_spec["meta"]["inference_defaults"]

    todo = []
    for set_name, pid, prompt, seed, w, h, m in jobs:
        out_png = os.path.join(args.outdir, name, set_name, pid + ".png")
        if args.overwrite or not os.path.exists(out_png):
            todo.append((set_name, pid, prompt, seed, w, h, m, out_png))
    print(f"[preview_eval] {name}: {len(todo)}/{len(jobs)} to generate "
          f"({'overwrite' if args.overwrite else 'skipping existing'})", flush=True)
    if not todo:
        print("[preview_eval] nothing to do")
        return
    for d in {os.path.dirname(t[7]) for t in todo}:
        os.makedirs(d, exist_ok=True)

    device, dtype = "cuda", torch.bfloat16
    t0 = time.time()
    encoder = load_text_encoder(args.text_encoder, dtype)
    dit, ae = build_pipeline(args.dit, args.vae, device=device, dtype=dtype, attn_mode=args.attn_mode,
                             lora_weights=args.lora, lora_multipliers=args.lora_multiplier)
    load_sec = time.time() - t0
    print(f"[preview_eval] model load {load_sec:.1f}s", flush=True)

    logs = {}
    run_t0 = time.time()
    for i, (set_name, pid, prompt, seed, w, h, m, out_png) in enumerate(todo):
        g = SimpleNamespace(
            prompt=prompt, negative_prompt=defaults.get("negative_prompt", ""), num_images=1,
            seed=seed, steps=defaults["steps"], guidance_scale=defaults["cfg"],
            width=w, height=h, y1=0.5, y2=1.15, mu=None, blocks_to_swap=0,
        )
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        images = generate(g, dit, ae, encoder, device, dtype, te_device=device)
        sec = time.time() - t
        images[0].save(out_png)
        peak = torch.cuda.max_memory_allocated() / 2**30
        entry = {"prompt": prompt, "seed": seed, "steps": g.steps, "cfg": g.guidance_scale,
                 "width": w, "height": h, "sec": round(sec, 1),
                 "peak_vram_alloc_gib": round(peak, 2), **m}
        logs.setdefault(set_name, {})[pid] = entry
        # flush the log for this set after every image so a killed run keeps its record
        log_path = os.path.join(args.outdir, name, f"generation_log_{set_name}.json")
        prev = json.load(open(log_path)) if os.path.exists(log_path) else {}
        prev.setdefault("images", {}).update({pid: entry})
        prev.update({"checkpoint_name": name, "dit": os.path.abspath(args.dit),
                     "lora": args.lora, "lora_multiplier": args.lora_multiplier,
                     "set": set_name, "gpu": torch.cuda.get_device_name(0),
                     "torch": torch.__version__, "dtype": "bf16",
                     "eval_set_version": (json.load(open(args.style_swap_json))["meta"]["version"]
                                          if set_name == "style_swap" else u_spec["meta"]["version"]),
                     "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(log_path, "w") as f:
            json.dump(prev, f, ensure_ascii=False, indent=1)
        print(f"[{i+1}/{len(todo)}] {set_name}/{pid} {w}x{h} {sec:.1f}s peak {peak:.1f}GiB", flush=True)

    print(f"[preview_eval] done: {len(todo)} images / {time.time()-run_t0:.0f}s (model load excluded)")


if __name__ == "__main__":
    main()
