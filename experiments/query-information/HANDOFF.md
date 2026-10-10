# Handoff — Query-Information Study

**Branch.** `obv/products-l2-query-information-20261010`, based on
`fix/weights-exact-commit-20261001` (4b17554). Draft PR into the base
branch; do not merge, deploy, or change workflows or repository settings.

**Question.** When a rewritten rung scores below verbatim, is that
information the rewrite removed, or language understanding the submission
lacks?

## Complete

- `grammar.py`, `decoding.py`, `manifest.json`, `run.py` — frozen 40-scene
  grammar; multiset-aware compatibility (qi-compat-1); lexical and
  grammar-aware decoders; exact-grammar and lexical-idf ranking systems;
  run/replay runner that refuses to run if `perturb.py` or `metrics.py`
  drift from their manifest hashes.
- `results/` — all eight artifacts committed, including the content-hash
  witness and the committed replay verdict (full agreement: all seven
  files byte-identical on regeneration).
- `tests/` — 36 tests, all passing: hand-derived rewrite strings for the
  real perturbation transforms, hand-computed compatible-target sets,
  manifest counts and hashes, system guarantees, and committed-results
  consistency.
- `README.md` — method, findings with the MRR/ceiling table, and exact
  reproduction commands.

## Findings (short form)

- The real stopword list deletes `near` on all 8 near-family keywords
  rewrites; truncated inherits the loss. Information removal starts at
  the stopword list, not at truncation.
- keywords/truncated drops are ~entirely information removal: exact-grammar
  lands within 0.01–0.02 of the conditional-identifiability ceilings
  (0.8913 keywords, 0.6317 truncated).
- The typo rung is recoverable language robustness: every corruption has
  a unique one-substitution correction, and exact-grammar attains the
  perfect 1.0 ceiling while lexical-idf realizes 0.75.
- Truncated rewrites collide (16 groups / 32 queries); "red square blue"
  spans 10 candidates. Collisions reported, never dropped.
- Small-adapter baseline: not run — no `cogworks.submissions.v2` entry
  points are registered in this environment (recorded in
  `results/systems.json`).

## Not run / not done

- No paid-service or live-model step was run; the study is fully offline
  by design.
- No course data files were read; nothing imports the benchmark's
  public manifests.
- Not merged, not deployed; no workflow or repository-settings changes.

## Resume

```bash
cd /home/user/work/cogworks-week3-language-benchmark
git checkout obv/products-l2-query-information-20261010
python -m pytest experiments/query-information/tests -q   # 36 passed
PYTHONPATH=. python experiments/query-information/run.py --replay \
  experiments/query-information/results                    # replay AGREES
```
