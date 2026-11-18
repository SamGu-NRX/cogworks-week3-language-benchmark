# Progress snapshot: CogPortal-language-lifecycle (20261118)

The local checkout `~/Programming Projects/CogPortal-language-lifecycle` was removed on 2026-11-18 to free disk space on Sam's Mac, after everything in it was pushed here. This file records where the work stood.

## What it is

CogWorks Week 3 benchmark: semantic image search: Your capstone builds a system that takes a caption and finds the images
that match it. This benchmark measures that system the same way for every
team, so results are comparable: it hands your code pinned caption sets and
pinned ResNet-18 descriptors, your code embeds them, and the benchmark
computes every similarity and ranking itself against COCO's own
caption-to-image pairs. Your training choices are yours; the measurement is
shared.



## State

- Checkout branch: `fix/runtime-encoder-lifecycle` at `94c7e64f7e`
- Snapshot branch: `archive/wip-20261118-CogPortal-language-lifecycle` (HEAD plus the uncommitted work listed below)
- Last commits:
- 94c7e64 2026-09-29 ci: install the published SDK so discovery-backed tests run
- a2fdcfd 2026-09-29 test: bind projected stores' search through discovery, not by hand
- f5f7347 2026-09-29 fix: build the database before an image encoder that is its method
- 6dc63fe 2026-09-15 docs: describe the search score the way it is computed
- ed6e4e7 2026-09-15 fix(diagnostics): stop leading a good run with its own floors
- daa254c 2026-09-15 test: drop a __getattr__ that only restated the default
- 04925d0 2026-09-15 fix(diagnostics): stop withholding and claiming what this run can see
- a7525bf 2026-09-15 docs: name the diagnostic cap that actually binds

Uncommitted work captured in this branch (3 paths):

- `.DS_Store`
- `language_search_benchmark/.DS_Store`
- `tests/.DS_Store`

## What's left

Nothing recorded beyond the commits above.

## How to resume

```sh
git clone https://github.com/SamGu-NRX/cogworks-week3-language-benchmark.git "$HOME/Programming Projects/CogPortal-language-lifecycle"
git -C "$HOME/Programming Projects/CogPortal-language-lifecycle" checkout archive/wip-20261118-CogPortal-language-lifecycle
```

## Thread

No T3 thread was found for this checkout.
