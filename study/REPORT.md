# Retrieval-v4 Scorer-Validity Study — Report

Study `products-language-validity-20261009` · locked in `study/lock.yaml` (commit `9cc48e2`) before any arm was evaluated · recorded on branch `obv/products-language-validity-20261009-r1`.

## Question

Can a submission exploit lookup shortcuts through retrieval-v4's published scoring, and does the scoring miss real failures (false negatives)? The scorer (`plugins.LanguageSearchBenchmark.scorer_version: retrieval-v4`) averages three components — text MRR (5-decimal-rounded submission embeddings vs reference embeddings), retrieval MRR, and search MRR — over the keyword, truncated, and typo rewrite rungs.

## Design (frozen before evaluation)

- **Tiers:** test (20 queries, 100-pool, 75 captions) and evaluation (150 queries, 700-pool, 500 captions), pinned public manifests with sha256-verified artifacts.
- **Arms (15):** a reference reimplementation (IDF-weighted GloVe-200 caption embeddings, ridge projection from ResNet-18 descriptors, seeded validation λ), three lookup variants (derived — realistic cheater; oracle — upper bound; exact — what current guards measure), five planted mistakes, and five inert controls. Full definitions in `study/lock.yaml` and `study/arms.py`.
- **Seeds:** all fixed at lock (20261009 / 20261001); every arm deterministic.
- **Uncertainty:** locked paired bootstrap, 10,000 resamples, 95% CI, query as unit for retrieval/search, image for text.
- **Selection rule:** every arm ran every case on every tier; nothing dropped after seeing scores.

## Recorded run

`python experiments/language-validity/run.py --manifest experiments/language-validity/manifest.json` · 30/30 tier-arm results, 28,830 recorded units, **zero unavailable arms** · per-query evidence (raw ranks, first outcomes, gold) saved per arm under `experiments/language-validity/results/` · runlog in `results/runlog.json`.

Replay (no arm rerun): `python experiments/language-validity/run.py --replay experiments/language-validity/results`

## Headline: shortcut success

Overall score (equal-weight component mean, retrieval-v4 definitions):

| arm | test overall | evaluation overall | text MRR |
|---|---|---|---|
| reference (reimplementation) | 0.6862 | 0.4200 | 0.8700 / 0.7849 |
| **lookup-derived (realistic cheater)** | **0.9092 (+0.2230)** | **0.8819 (+0.4619)** | 0.9871 / 0.9744 |
| lookup-oracle (upper bound) | 0.8962 | 0.8878 | same as derived |
| lookup-exact (what guards measure) | 0.3516 (−0.3345) | 0.3299 (−0.0901) | same as derived |

A submission that never embeds anything — it repairs typos, resolves keyword/truncated rewrites against the corpus vocabulary, and answers through a 512-d Gaussian codebook — **beats the reference by +0.2230 on test and +0.4619 on evaluation**. The oracle arm (which may use the benchmark's own perturb module) adds nothing over the derived arm: paired differences are exactly zero, so the realistic cheater already captures the full shortcut.

The exact-string lookup the existing guards catch scores **below** the reference on both tiers. Current guard tests measure a threat that is not competitive while missing the one that is.

Rung detail (evaluation tier, lookup-derived vs lookup-exact, paired bootstrap at the locked 10,000 draws / seed 20261009): retrieval-keywords +0.9076 [+0.8554, +0.9528], retrieval-truncated +0.5365 [+0.4566, +0.6165], retrieval-typo +0.7152 [+0.6368, +0.7882], search-keywords +0.9809 [+0.9646, +0.9942], search-truncated +0.8522 [+0.8070, +0.8935], search-typo +0.9753 [+0.9557, +0.9915] (all CIs exclude zero). Only search-verbatim is unchanged (+0.0300 [−0.0033, +0.0667]): verbatim is guarded, rewrites are not.

## False-negative risk (does scoring miss real failures?)

- **Planted mistakes survive scoring when the failure is a weighting choice.** mistake-no-idf scores 0.6627 on test and mistake-perword-norm 0.6663 — within ~0.02 of the reference, i.e. a submission with the wrong word weighting is nearly indistinguishable from a correct one. By contrast mistake-raw-count collapses (0.1923 test / 0.0574 evaluation), as do wrong-space mistakes (raw descriptors: retrieval MRR 0.0106; memorized search: search MRR 0.0025). The scorer detects gross representation failures but is weakly sensitive to weighting-quality failures.
- **Inert controls floor correctly.** constant/zero/random/shuffled/pool-order arms land at 0.01–0.10 overall — the chance floors are real, so the pipeline would notice a broken submission.
- **Rounding probe:** replicating the driver's normalize→round(5dp) path, raw vs rounded mean reciprocal rank is identical — **0 rank flips** on all 20 test and 150 evaluation queries. Published precision is not hiding rank changes.
- **Multi-relevant audit (established):** exact caption-text duplicates are absent in-pool on both tiers (corpus-level: 1 test / 10 evaluation queries, never in-pool). Interpretive Jaccard bands on evaluation: zero queries with a top non-gold pool image at Jaccard ≥ 0.6 or ≥ 0.8, zero exact duplicates — no near-duplicate false-negative signal in top-ranked results.
- **Natural-query audit (established, annotation-based):** 300 co-captions of evaluation query images (seed 20261009, ≤2 per image) score retrieval-verbatim 0.3070 / search-verbatim 0.3060 for the reference — lower than rewritten-run scores, as expected for unmodified natural sentences.

Interpretive outputs (agent-judged shortlists, Jaccard bands) are reported separately in `results/audit/` and are **not** independent human truth.

## Validation gate

Independent validation (commit `77df427`, `study/validate_metrics.py`, pure-python reference implementations, no dependency on the metric code under test) — **7/7 families PASS** at full scale (`python -m study.validate_metrics --scale full` → `study/validation_results.json`), with 5 recorded discrepancies, all at noise scale:

| # | family | result |
|---|---|---|
| 1 | Per-rung MRR/bit-reproducibility | PASS — pairwise replica matches the np.mean-based implementation bit-for-bit through n=5000, beyond the locked 1024 bound |
| 2 | Rung-grid mean summation | PASS with caveat — `metrics.py:479,574` uses CPython≥3.12 compensated builtin `sum`, not np.mean; last-bit diffs (≤2.2e-16) on 4/50 random 3-rung grids, so grid means reproduce bits only on the same CPython |
| 3 | Tie semantics | PASS — ties, −0.0, and −inf agree with tuple sort on all 21 probe cases |
| 4 | Chance floors | PASS — rational vs published floor differ at ~2.4e-16 relative ([6,2,2,2]: 0.4717471561627406 vs …405); comparisons need ~1e-12 tolerance |
| 5 | 5-dp rounding noise | PASS — flips are bounded by the rounding noise bound with a constructible near-tie counterexample (gold rank 1→2); float32 error (~4e-8) is 3 orders below payload rounding noise |

The rounding-flip probe inside the recorded run additionally showed 0 actual flips on both tiers (20 test / 150 evaluation queries). Two bugs in the validation harness itself were found and fixed during construction (first-relevant reference looked up self instead of first same-group; two inverted probe expectations). A third hardening landed after CI review: the original exact-ties check ranked gemm output of constant rows, but gemm accumulation order varies by output position across BLAS kernels (measured span 2.2e-16 under the Haswell kernel vs exactly 0 under SkylakeX), so mathematically equal entries could differ in their last bits and reshuffle the seeded tie order — a scorer-environment property of gemm, now recorded as a measurement (`constant_gemm_score_spans`) rather than gated. The gated check verifies the rounding property directly (elementwise rounding maps constant rows to constant rows; equal score matrices rank identically). The family passes 7/7 under forced Haswell, SkylakeX, and default kernels. 17 validation tests added; full suite: **222 passed** (including the 35 discovery tests that skip without the pinned cogbench SDK), 0 failed. Study files lint clean (`ruff check study/ experiments/` + new test files: all pass; 5 pre-existing findings remain in baseline test files, untouched).

All family results PASS, so the variant differences above stand as interpreted under the locked analysis plan.

## Deviations from lock

- `study/arms.py` stopword list was corrected after initial runs (the list now covers the benchmark rewrite filter's dropped words, so rewritten queries resolve against the lookup index). Affected arms were rerun; all results in this report come from the fixed code.
- The paired bootstrap was first executed with 2,000 draws and seed 20261001, contradicting the lock (10,000 draws, seed 20261009); the discrepancy was found in review. `study/analyze.py` is now bound to the locked settings, the analysis was regenerated by replay (no arm rerun; per-unit evidence unchanged), and every interval in this report comes from the locked-settings run. Point estimates and all conclusions are unchanged; interval endpoints moved by at most ~0.004.
- The staff reference submission is not in this repository. The reference arm is a reimplementation; quoted historical numbers (0.6925/0.6594 test) come from the unavailable implementation and are cited, not reproduced.



## Environment

Locked environment was Python 3.13.14 / NumPy 2.3.5 (outside the package's declared ranges; the source was run directly). The full repository suite additionally passes on the package's declared floor: **222 passed, 0 failed** under Python 3.8.20 with the pinned cogbench SDK installed (187 passed / 35 skipped without it), including all study, experiment, and validation tests.
