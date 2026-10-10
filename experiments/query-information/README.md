# Query-Information Study

**Question.** When a rewritten rung scores below verbatim, is that
information the rewrite removed, or language understanding the submission
lacks?

**Method in one paragraph.** A finite synthetic grammar (40 candidate
scenes over color/shape/relation families, frozen candidate order,
`manifest.json`) supplies both candidates and queries. Every rung rewrite
is produced by the benchmark's own, unmodified perturbation functions —
the study imports `language_search_benchmark.perturb` and pins its SHA-256
in the manifest; `run.py` refuses to run if those files drift. Two offline
systems rank the full pool per rung: `exact-grammar` (grammar-aware
decoding with unique in-vocabulary typo correction, attribute-overlap
scoring) and `lexical-idf` (IDF-weighted bag-of-words cosine, the course's
zero-vector rule, no network). Compatibility is multiset containment of
decoded query tokens in a candidate's attribute tokens (compat version
`qi-compat-1`): multiplicity matters, order does not, and corrupted tokens
are dropped by the lexical decoder or uniquely corrected by the grammar
decoder. Everything is offline; no course data, no adapters, no model
endpoints.

## Findings

1. **The stopword list deletes the relation word on `near` queries.** The
   real `perturb.STOPWORDS` contains `near`, so the keywords rung drops it
   from all 8 near-family queries, and truncated rewrites inherit the same
   loss. Information removal starts before truncation does. This is
   measured from the transforms, not assumed.

2. **Rewritten-rung drops are almost entirely information removal, not
   missing language understanding.** MRR (40 queries, whole pool):

   | system | verbatim | keywords | truncated | typo | scored mean |
   |---|---|---|---|---|---|
   | exact-grammar | 1.0000 | 0.8821 | 0.6127 | 1.0000 | 0.8316 |
   | lexical-idf   | 1.0000 | 0.9021 | 0.6687 | 0.7500 | 0.7736 |

   The conditional-identifiability ceilings (E[MRR] with uniform
   tie-breaks inside each compatible set) under lexical decoding are
   1.0 / 0.8913 / 0.6317 / 0.84375 for verbatim/keywords/truncated/typo.
   exact-grammar lands at 0.8821 and 0.6127 on keywords and truncated —
   within ~0.01–0.02 of the ceiling, the residue of seeded within-tie
   tie-breaks — so nearly the whole drop from verbatim is information the
   rewrite no longer carries, not understanding the system lacks.

3. **The typo rung splits cleanly.** Lexically, typo rewriting costs
   0.25 MRR below verbatim for lexical-idf (1.0 → 0.75). Every corruption
   on this pool has exactly one one-substitution correction in the
   attribute vocabulary (see `results/identifiability.json` correction
   log), so the grammar-aware decoder restores unique identification and
   exact-grammar attains a perfect 1.0 — the ceiling itself. The typo gap
   is recoverable system-side; it is a language-robustness cost, not
   information the query lost.

4. **Truncated rewrites collide.** 16 collision groups cover 32 of 40
   queries on the truncated rung (e.g. "red circle below" is the shared
   rewrite of two distinct scenes). One rewrite, "red square blue", is
   compatible with 10 of 40 candidates: deleting "near" also deletes the
   relation, and the token `square` does double duty as first-scene shape
   and second-scene shape, so the shape constraint stops binding.
   Collisions are reported, never dropped.

5. **Both systems' answers are always justified.** Justified-answer
   coverage (top answer compatible with the rewritten query under the
   system's own decoder) is 1.0 for both systems on every rung — measured,
   not assumed; the lexical system had no guarantee a priori.

6. **Adapter baseline not run.** No `cogworks.submissions.v2` entry points
   are registered in this environment, so the small-adapter column is
   recorded as not run (`results/systems.json`) rather than filled with a
   stand-in.

## Layout

- `grammar.py` — the frozen pool, query template, and rung rewrites
  (delegated to `language_search_benchmark.perturb`).
- `decoding.py` — compatibility semantics (qi-compat-1), the two
  decoders, the two systems, adapter discovery.
- `manifest.json` — the frozen input list: seeds, family counts, rungs,
  reference hashes of `perturb.py`/`metrics.py`, provenance, limits.
- `run.py` — run mode (`--manifest`) and replay mode (`--replay`), which
  recomputes everything and compares content hashes against the witness.
- `results/` — committed artifacts: `raw_ranks.json`,
  `justified_coverage.json`, `rewrite_ambiguities.json`,
  `identifiability.json`, `systems.json`, `summary.json`, `witness.json`
  (content hashes), `replay_report.json` (committed replay verdict).
- `tests/` — 36 tests pinning hand-derived rewrite strings,
  hand-computed compatible sets, manifest hashes, and system behavior.

## Reproduce

```bash
# unit suite (offline)
python -m pytest experiments/query-information/tests -q

# regenerate results from the frozen manifest
PYTHONPATH=. python experiments/query-information/run.py \
  --manifest experiments/query-information/manifest.json

# verify the committed results regenerate byte-identically
PYTHONPATH=. python experiments/query-information/run.py \
  --replay experiments/query-information/results
```

The replay compares the SHA-256 of every results artifact and the
manifest against the committed witness and exits nonzero on any
disagreement. The committed `results/replay_report.json` records the
verdict: full agreement.

## Limits

Conditional identifiability here means: what a rewritten query still
determines over these 40 candidates, inside this finite grammar. It is
not general semantic truth. Nothing in this directory changes
`metrics.py`, `perturb.py`, or any benchmark manifest; the study measures
the frozen transforms, it does not replace them.
