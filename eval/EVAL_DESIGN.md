# Evaluation design

The prompt sets themselves are **not published** — they are frozen inputs containing prompt
text, so they stay on the training machine (see `.gitignore`). This documents their structure,
which is what makes the results readable. The generator and the page builder here run against
those local files.

## Usability set

20 prompts, frozen at run start — text, seeds and aspect ratios are immutable for the whole
run, typos included, so the epoch series stays comparable. Four groups of five:

| group | what it isolates |
|---|---|
| format equivalence | one scene written four ways — long natural-language paragraph, single sentence, tags only, and a mixed form — plus a second scene as a paragraph. Asks whether the model reaches the same image regardless of how the user writes. |
| rating ladder | one scene, one seed, with only the content-rating token swapped across the model's rating levels. Measures the rating dial's authority and monotonicity. |
| tag isolation | minimal prompts carrying exactly one tag — style dials against ordinary composition and prop tags as controls. Shows whether a dial does anything on its own. |
| real usage | prompts written the way the operator actually writes them, unedited. Includes a deliberate near-synonym pair to test lexical equivalence, and a pair differing only by a style dial. |

**Seed policy.** Every prompt has a fixed seed, reused at every epoch boundary and across every
checkpoint. Prompts that differ only by a swapped token deliberately **share** a seed, so the
swap is the only variable in the comparison.

## Resolution tiers

Every prompt is rendered at **768 / 1024 / 1536**, each tier preserving that prompt's own
aspect ratio: dimensions are chosen so area ≈ tier², rounded to multiples of 16 (the latent
step). Training happens at 1024² only, with no resolution curriculum.

1536 is **outside the base model's native band**. It is generated anyway and recorded as a
**pre-polish baseline**: whatever breaks down is logged as-is rather than tuned away, so a
later polish stage has a documented starting point rather than an anecdote.

## Style swap set

1024 only. Two fixed base prompts — one minimal, one with real scene content — swept across
every attribute in the curated style vocabulary plus a **no-attribute control**, all at a fixed
seed per base prompt. The control is what makes the sweep readable: without it, "the dial did
something" is unfalsifiable.

## Review page

`build_preview_review.py` lays out, per group, a table whose **rows are (checkpoint × resolution
tier)** and whose **columns are the prompts of that group**. Reading across a row compares the
group's swap at fixed resolution; reading down a column compares resolutions and epochs for one
prompt. Passing several checkpoint names stacks them into the same table for progressive
comparison. Out-of-band rows are marked. The style sweep renders as a labelled grid with the
control called out.

Generation logs record real dimensions, seed, step count, wall time and peak VRAM per image, so
a page can be audited after the fact.
