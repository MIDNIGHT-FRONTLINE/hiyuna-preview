#!/usr/bin/env python3
"""Extract a LoRA from the weight difference between a fine-tuned Krea-2 DiT and its base.

Ported from musubi-tuner's qwen_extract_lora.py; only the key-selection and naming rules are
architecture-specific. For Krea-2 the target set is exactly the 2-D `.weight` tensors — 264 of
them, matching the full Linear set that networks/lora_krea2.py wraps (its target list is None,
i.e. every Linear). Everything else in the checkpoint is 1-D (RMSNorm scales, modulation
parameters) and cannot carry a low-rank factorisation. `last.modulation.lin` is the one 2-D
tensor that is not a `.weight`; it is modulation, never wrapped as LoRA, and so excluded.

Each delta W_tuned - W_base is factorised by SVD, truncated to `--dim`, and split into
    lora_down (rank x in)  = sqrt(S) V^T
    lora_up   (out x rank) = U sqrt(S)
with alpha = dim, so the loader's scale (alpha/dim) is 1.0 and the extracted LoRA reproduces
the delta at multiplier 1.0. Singular values are clamped at --clamp_quantile before the split
to keep a few outliers from dominating the factorisation.

Naming is `lora_unet_<module path with dots replaced by underscores>`, the convention
networks/lora_krea2.py builds at load time and the one Comfy's LoRA loaders expect.

  python3 krea2_extract_lora.py --model_org raw.safetensors --model_tuned ep2.safetensors \
      --save_to out.safetensors --dim 128 --device cuda
"""
import argparse
import json
import logging
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _str_to_dtype(p):
    return {"float": torch.float, "fp16": torch.float16, "bf16": torch.bfloat16}.get(p)


def _lora_name_from_key(key: str) -> str:
    return "lora_unet_" + key[: -len(".weight")].replace(".", "_")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_org", required=True, help="base DiT (pristine Krea-2-Raw)")
    p.add_argument("--model_tuned", required=True, help="fine-tuned DiT checkpoint")
    p.add_argument("--save_to", required=True)
    p.add_argument("--dim", type=int, default=128, help="LoRA rank")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clamp_quantile", type=float, default=0.99)
    p.add_argument("--save_precision", default="bf16", choices=["float", "fp16", "bf16"])
    p.add_argument("--min_diff", type=float, default=0.0,
                   help="skip a module whose delta max-abs is below this (0 = keep all)")
    return p.parse_args()


def main():
    args = parse_args()
    dtype_save = _str_to_dtype(args.save_precision)
    device = torch.device(args.device)

    with safe_open(args.model_org, framework="pt") as f_org, \
         safe_open(args.model_tuned, framework="pt") as f_tun:
        org_keys, tun_keys = set(f_org.keys()), set(f_tun.keys())
        if org_keys != tun_keys:
            only_org, only_tun = org_keys - tun_keys, tun_keys - org_keys
            raise SystemExit(f"key sets differ: {len(only_org)} only in base, {len(only_tun)} only in tuned")

        targets = sorted(k for k in org_keys
                         if k.endswith(".weight") and len(f_org.get_slice(k).get_shape()) == 2)
        logger.info(f"target Linear weights: {len(targets)} of {len(org_keys)} tensors")

        lora_sd = {}
        skipped, ranks_capped = [], 0
        t0 = time.time()
        for key in tqdm(targets, desc=f"SVD rank {args.dim}"):
            w_org = f_org.get_tensor(key).to(device, torch.float32)
            w_tun = f_tun.get_tensor(key).to(device, torch.float32)
            diff = w_tun - w_org
            del w_org, w_tun

            if args.min_diff > 0 and diff.abs().max().item() < args.min_diff:
                skipped.append(key)
                del diff
                continue

            out_dim, in_dim = diff.shape
            rank = min(args.dim, out_dim, in_dim)
            if rank < args.dim:
                ranks_capped += 1

            U, S, Vh = torch.linalg.svd(diff)
            U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

            # clamp the singular-value tail so a handful of directions cannot dominate
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi = torch.quantile(dist.abs(), args.clamp_quantile)
            U = U.clamp(-hi, hi)
            Vh = Vh.clamp(-hi, hi)

            sqrt_s = torch.sqrt(S)
            up = (U * sqrt_s.unsqueeze(0)).contiguous()        # (out, rank)
            down = (sqrt_s.unsqueeze(1) * Vh).contiguous()     # (rank, in)

            name = _lora_name_from_key(key)
            lora_sd[f"{name}.lora_up.weight"] = up.to("cpu", dtype_save)
            lora_sd[f"{name}.lora_down.weight"] = down.to("cpu", dtype_save)
            lora_sd[f"{name}.alpha"] = torch.tensor(float(rank)).to(dtype_save)
            del diff, U, S, Vh, up, down

    n_mod = sum(1 for k in lora_sd if k.endswith(".alpha"))
    logger.info(f"extracted {n_mod} modules in {time.time()-t0:.0f}s"
                f"{f', {ranks_capped} rank-capped by layer size' if ranks_capped else ''}"
                f"{f', {len(skipped)} skipped below --min_diff' if skipped else ''}")

    metadata = {
        "ss_network_module": "networks.lora_krea2",
        "ss_network_dim": str(args.dim),
        "ss_network_alpha": str(args.dim),
        "ss_extracted_from": os.path.basename(args.model_tuned),
        "ss_extracted_base": os.path.basename(args.model_org),
        "ss_extract_clamp_quantile": str(args.clamp_quantile),
        "ss_output_name": os.path.splitext(os.path.basename(args.save_to))[0],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.save_to)), exist_ok=True)
    save_file(lora_sd, args.save_to, metadata=metadata)
    size_gb = os.path.getsize(args.save_to) / 2**30
    logger.info(f"saved {args.save_to} ({size_gb:.2f} GiB, {len(lora_sd)} tensors)")
    logger.info("metadata: " + json.dumps(metadata))


if __name__ == "__main__":
    main()
