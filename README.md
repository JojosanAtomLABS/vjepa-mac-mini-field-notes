# V-JEPA on a Mac Mini M4 — Garage Field Notes

Independent field notes on running Meta's V-JEPA 2.1 and building a miniature JEPA from scratch on Apple Silicon, in October 2026. Not affiliated with Meta or the V-JEPA team. This repo documents what actually ran, what failed in an instructive way, and the limits of the numbers — not a benchmark, and not a claim that a toy model rivals the real thing.

> Upstream code and weights are Meta's: https://github.com/facebookresearch/vjepa2 (V-JEPA 2 / 2.1). No upstream code, weights, or personal videos are republished here. The miniature model described below is a from-scratch toy, trained on synthetic bouncing-ball clips only.

## Verdict in one paragraph

V-JEPA 2.1 ViT-B/16 runs fine on a 24 GB Mac Mini M4, and its frozen features are genuinely useful for simple motion tasks. The more interesting result is the negative one: a miniature JEPA trained without BatchNorm collapses to a trivial solution — loss near zero, cosine similarity 1.00 vs 1.00, delta exactly 0 — while the same idea with BatchNorm learns real trajectory structure. In other words, a near-zero loss proves nothing on its own. EMA protects against one kind of collapse, not against a predictor that learns to output the same unit vector for everything.

## What was run

| Item | Value |
|---|---|
| Host | Mac Mini M4, 24 GB RAM |
| Python / PyTorch | Python 3.12.13 (Homebrew), PyTorch 2.14.1, MPS |
| Upstream | V-JEPA 2.1, ViT-B/16 (~1.55 GB weights, `ema_encoder` checkpoint keys) |
| First inference | 429 ms cold; **89 ms warm** on a real video, features `[1, 4608, 768]` (8 × 576 tokens, embed_dim 768) |
| Input | 16 frames, 384 × 384, tubelet_size 2, patch_size 16 |

### Apple Silicon gotchas, verified

- **`decord` is broken on ARM64** — replaced with PyAV (`av`) for video loading. PyAV worked, including on a real video (16 frames in 2.23 s including resize/crop).
- **MPS fallback:** `PYTORCH_ENABLE_MPS_FALLBACK=1` was set as a precaution for a `torch.bmm` MPS issue; it was not ultimately required.
- **Upstream test URL bug:** `VJEPA_BASE_URL` in the hub code pointed at a localhost test URL; weights were downloaded manually from Meta's public files instead.
- **Model path:** V-JEPA 2.1 uses `app.vjepa_2_1.models`, not the older `src.models` path, and must be launched from the repo root for imports to resolve.

## What JEPA actually does (dissected, not just run)

- A **context encoder** encodes only the visible (unmasked) tubelets.
- A **target encoder** — a deepcopy of the encoder, `requires_grad=False`, updated **only by EMA** — encodes the full clip and provides the prediction targets, layer-normalised.
- A **predictor** takes context tokens plus learnable mask tokens carrying the *target positions*, and predicts latent representations for the masked regions.
- The loss is computed **in latent space only** (L1/L2 on predicted vs target latents), never on pixels. That is the whole point: predicting structure, not texture.

The EMA teacher is the anti-collapse ingredient — but, as the miniature shows below, it is not sufficient on its own.

## Experiment 1 — Frozen V-JEPA features are useful (linear probe)

Method: frozen V-JEPA 2.1 features, mean-pooled to 768D, StandardScaler + logistic regression (seed 42), stratified 80/20 split — **72 clips train, 18 clips held-out test**.

| Task | Train accuracy | Test accuracy | Baseline |
|---|---|---|---|
| Fast vs slow | 100% | 100% | 50% |
| Motion vs static | 100% | 100% | 50% |
| Fast / slow / static (3 classes) | 100% | 100% | 33% |

Verdict: the features carry real motion/dynamic signal, well above baseline.

**Honest caveat, kept in the headline, not the footnote:** 18 test clips is a small held-out set — roughly ±23% at a 95% confidence interval. 100% on three independent tasks is very unlikely to be luck, but it is not a statistically definitive benchmark. There were also `overflow in matmul` warnings (very large-scale features, not fully absorbed by the scaler) that did not prevent convergence. A full validation (300 clips, 5-fold CV) was judged unnecessary at this stage — and is still the obvious next step before any stronger claim.

## Experiment 2 — Mini-JEPA from scratch: it learns (Phase 4)

Synthetic data only: 2,000 clips × 8 frames × 64×64 RGB, 2–3 coloured balls with elastic bounces.

- Architecture: 3-layer CNN encoder with BatchNorm (32→64→128 channels) + linear head to 128D, EMA target encoder (momentum 0.99), MSE loss on L2-normalised targets. ~1.2M trainable parameters.
- Masking: frames 0–3 = context, frames 4–7 = targets.
- Training: 1,000 epochs, 21.5 minutes on MPS, 2.2 GB RAM. Loss 0.007317 → 0.000026 by epoch 100.
- **No collapse:** `std(z_target)` stayed ~0.0882 throughout (alert threshold 0.01).
- **It learned trajectory structure:** cosine similarity same-clip 0.5704 vs different-clip 0.4930, **Δ = +0.0774**. Modest, but a random latent space would give Δ ≈ 0. Context frames of a clip are measurably closer to that clip's own future than to another clip's future.

## Experiment 3 — Mini-JEPA without BatchNorm: the trivial solution (Phase 4b)

Same idea, stricter spec: 2 conv layers, **no BatchNorm**, AdaptiveAvgPool, 413,504 parameters, 200 epochs in 13.9 minutes (4.15 s/epoch).

- Loss: 0.001287 → ~0.000000 by epoch 10. Looks perfect.
- **Collapse check:** `std(z_target)` never tripped the 0.01 threshold — so it is not the classic EMA collapse.
- **But:** cosine similarity 1.00 vs 1.00, **Δ = 0.0000**. After L2 normalisation, every embedding points the same way. The predictor found the trivial solution: project everything to one constant unit vector, and MSE against L2-normalised targets is ~0.

**Lesson:** BatchNorm in the CNN encoder was critical here. Without it, the architecture is vulnerable to representational collapse even while EMA works and the loss looks ideal. Loss ≈ 0 is not evidence of learning; the same-clip vs different-clip delta is the check that matters.

## What was NOT done (Phase 3 verdicts)

- **Fine-tuning V-JEPA on personal videos: NO-GO.** Estimated ~45 hours for 500 clips × 30 epochs, plus 3–4 hours adapting the distributed/SLURM training framework first, for a marginal and possibly degrading gain on a domain-general model.
- **Mini-JEPA (Option B) and linear probe (Option C): GO** — both completed above, in a few hours each.

## Scope and disclaimer

These are garage experiments on a single Mac Mini, with synthetic data for the miniature and a small personal clip set for the probe. They are not a reproduction of Meta's V-JEPA results, not a benchmark, and not a product. The toy model uses naive temporal masking (first half vs second half of each clip); real V-JEPA masks random spatio-temporal blocks, which forces spatial understanding as well. Reproduce the probe with more clips before citing the 100% figures anywhere serious.

## Sources

- https://github.com/facebookresearch/vjepa2 — V-JEPA 2 / 2.1, Meta
- Field notes and training logs: October 2026, Mac Mini M4 (this repo's experiments)
- Yann LeCun's JEPA direction (predict latent representations, not pixels or next tokens) — the paradigm these experiments were built to touch, not just read about
