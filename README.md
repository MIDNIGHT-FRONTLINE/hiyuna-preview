# Hiyuna — preview fine-tune

**Hiyuna** is an anime-specialised text-to-image model built on **Krea-2** (12.9B
single-stream DiT + Qwen3-VL-4B text encoder) by full fine-tuning, not LoRA.

This repository records the **preview stage**, which runs *before* the main fine-tune: a
deliberately small full FT over a hand-curated keep set, from which the weight delta is
extracted and distributed as a LoRA. Its purpose is to show the model's direction early and to
settle the caption and evaluation methodology before committing to the long run.

> **Code and method only.** No dataset, no prompts, no captions, no tag strings, no
> checkpoints, no generated samples — see [Not included](#not-included). Record-keeping
> repository; not accepting contributions.

## Status

| stage | what | state |
|---|---|---|
| 1 | preview caption profile — style dials, artist attribution removed | done, verified |
| 2 | preview training run — 3 epochs from pristine Krea-2-Raw | **running** |
| 3 | weight-delta → LoRA extraction (rank 128 / 256) + verification | queued |
| — | main fine-tune | not started (separate effort) |

## The preview run

- **Base**: pristine `Krea-2-Raw` — not a pilot checkpoint.
- **Data**: 9,439 hand-curated images (private), single layer, `num_repeats = 1`.
- **Recipe**, validated by a 10k-step pilot and reused unchanged: full bf16, Adafactor with a
  fused backward pass (per-parameter `register_post_accumulate_grad_hook`, stochastic
  rounding, gradients freed immediately), gradient checkpointing, `--max_grad_norm 0`, LR 1e-5
  constant after a 100-step warmup, resolution-aware timestep shift, batch 4 at 1024² with
  aspect bucketing.
- **Schedule**: 3 epochs ≈ 7,100 steps, run as one segment per epoch (`--resume` chained) so a
  review page exists at every epoch boundary — a single GPU cannot sample and train at once.
  Each segment's terminal save is already the model-only DiT `state_dict`, so no extraction
  step is needed to get a loadable checkpoint.
- **Hardware**: one RTX PRO 6000 Blackwell 96GB. ~56 GB in use while training.

### Captions are synthesised at training time

No text sidecars. Each step composes its captions on the fly from structured material, seeded
deterministically per `(run, image, epoch)` — reproducible, but different every epoch. The
preview profile removes artist attribution entirely and puts **curated style attributes** in
that leading slot instead, drawn by a **count-limited** sampler (0 / 1 / 2 / all at
10 / 35 / 35 / 20 %, then shuffled) so each dial gets solo exposure rather than the model
learning a fixed co-occurrence as house grammar. Verified across 28,317 synthesised captions:
zero invariant violations, and 32.9% of styled captions carry exactly one style attribute.

Full profile, rationale and verification numbers: [`caption/PROFILE.md`](caption/PROFILE.md).

### Evaluation

A frozen 20-prompt usability set — format equivalence, a content-rating ladder, single-tag
isolation, and real operator prompts — regenerated at every epoch boundary with identical
seeds, each prompt at **768 / 1024 / 1536** with its aspect ratio preserved. 1536 sits outside
the base model's native band and is kept deliberately as a **pre-polish baseline**: the
breakdown is recorded rather than tuned away. A separate style-swap sweep at 1024 includes a
no-attribute control so dial effects are falsifiable.

Structure, seed policy and page layout: [`eval/EVAL_DESIGN.md`](eval/EVAL_DESIGN.md).

## Layout

```
caption/   profile config, materials builder, and a snapshot of the synthesis
           module so the run's caption behaviour is reproducible from this repo
train/     image extraction, dataset config, per-epoch run orchestrator
eval/      generator (3 resolution tiers, LoRA-capable) and review page builder
```

## Not included

The training set is private, and neither it nor anything derived from it is published here:
no images, no captions, no tag strings, no artist names, no post ids. Also excluded: the frozen
prompt sets, every generated sample, all run logs, and all checkpoints. Nothing adult-rated is
in this repository.

`caption/build_preview_materials.py` shows exactly which fields the trainer consumes, so the
pipeline is auditable without the data.

## License

Apache-2.0 (`LICENSE`), with upstream attribution in `NOTICE`. Covers the code in this
repository only — not the model weights, and not the dataset.
