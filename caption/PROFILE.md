# Preview caption profile

The trainer bakes no text sidecars. Captions are composed per step at training time
(`--te_online`) by a deterministic stochastic synthesiser: every random choice derives from
`random.Random(sha256(global_seed | item_key | epoch))`, so a rerun reproduces the caption
stream byte for byte while each epoch sees a different phrasing of the same image.

The preview profile is a **config-only overlay** on that synthesiser
(`caption_synth_preview_config.json`). Module defaults still reproduce the main fine-tune
profile, so the two runs share one code path with no branching.

## What the preview changes

| dial | main fine-tune | preview |
|---|---|---|
| artist attribution | emitted, with a drop rate | **removed entirely** — never emitted |
| leading identity slot | artist attribution | **per-image curated style attributes** (sidecar), count-limited + shuffled |
| quality-percentile token | emitted in bands | unused |
| year token | emitted, with a drop rate | unused |
| content-rating token | always emitted | always emitted, in every non-empty format |
| format mix | 6 formats incl. tag-only families | mixed .55 / long-NL .25 / short-NL .15 / empty .05 |
| general-tag shuffle, drop, exempt families, underscore handling | — | unchanged from the main profile |

Rationale for each removal: artist attribution is out of scope for a public preview; the keep
set is uniformly top-curated, so a quality percentile carries no signal; the year dial is
dropped to keep the preview's conditioning surface small.

## Style attributes are count-limited, not independently dropped

Naively dropping each attribute independently means an image tagged with five attributes
almost always shows all five, and the model learns the *co-occurrence* as house grammar
rather than learning each dial separately. So each synthesis instead draws a **count** from
the image's attribute pool and then shuffles order:

| drawn count | probability |
|---|---|
| 0 | 10% |
| 1 | 35% |
| 2 | 35% |
| all | 20% |

A curator flag, where present, is drawn separately (independent 15% drop) so it also gets
solo exposure instead of always riding along with the others.

## Verification

`build_preview_materials.py` joins the body table to the style sidecar and records the
sidecar's version timestamp into a `meta` table; the run logs that timestamp, so every
checkpoint is traceable to the exact tagging snapshot it trained on.

A verifier (not published — it quotes caption and tag text) checks the profile over the whole
keep set × 3 epochs, 28,317 synthesised captions:

- **Format mix**: .5510 / .2518 / .1467 / .0505 against the .55 / .25 / .15 / .05 spec.
- **Invariant violations: 0** — no artist attribution, no year token, no quality token, and no
  non-empty caption missing its rating token.
- **Count sampling** on pools of ≥3 attributes (n = 1,709): 9.8 / 32.9 / 36.1 / 21.2 % against
  the 10 / 35 / 35 / 20 spec. Curator-flag keep rate 85.5% (= 1 − 0.15). Leading slot is a
  style attribute in 100% of styled captions.
- **Single-dial exposure**: 32.9% of styled captions carry exactly one style attribute — the
  separability the count draw was introduced for.
- **General-tag handling** unchanged: mean drop 0.173 against E[U(0.1, 0.3)] = 0.20, full
  reshuffle every sample, exempt families never dropped, no underscore corruption.
- **Determinism**: identical seed reproduces exactly; a different seed diverges on 9/10 draws.

Two apparent violations in the first pass were bugs in the *verifier's* regexes, not the
synthesiser — a sentence-final punctuation boundary, and a substring match against text that
the vision-language captioner had itself transcribed from watermarks in the image. Both were
confirmed against the data and fixed; the synthesiser was correct in both cases.

A later log-label fix (the startup line hardcoded the wrong profile name) was verified to
leave caption output byte-identical on this profile and unchanged on the main profile.
