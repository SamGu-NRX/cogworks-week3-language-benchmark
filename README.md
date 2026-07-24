# CogWorks Week 3 benchmark: semantic image search

Your capstone builds a system that takes a caption and finds the images
that match it. This benchmark measures that system the same way for every
team, so results are comparable: it hands your code pinned caption sets and
pinned ResNet-18 descriptors, your code embeds them, and the benchmark
computes every similarity and ranking itself against COCO's own
caption-to-image pairs. Your training choices are yours; the measurement is
shared.

## What gets scored

Three parts of your system are observed separately, and the overall score
is their average. If one part is broken you still get credit for the parts
that work.

| Component | What runs | What it measures |
| --- | --- | --- |
| Text MRR | your `embed_text` on a pinned caption pool | whether captions of the same image land near each other in your space |
| Retrieval MRR | your `embed_text` + `embed_images` on pinned queries and descriptors | whether your trained encoder maps images near their captions |
| Search MRR | your own `search(query, k)` over a pinned pool | whether the whole application, database and all, actually works |

MRR is the mean of `1/rank` of the right answer, so rank 3 earns more than
rank 9 and a working pipeline always beats chance. The run also reports
Recall@1/5/10, median rank, and the chance baseline so you can see what
"better than guessing" means for the pool size.

## What your repo provides

Add a `benchmark_adapter.py` (about ten lines) and register it in your
`pyproject.toml`:

```python
# benchmark_adapter.py
from my_project import load_my_model, MySearchApp

def create_search_adapter(resources):
    # resources.captions_path / resources.descriptors_path / resources.glove_path
    # point at the course files; resources.load_glove() loads GloVe fast.
    return MySearchApp(load_my_model())
```

```toml
[project.entry-points."cogworks.submissions.v2"]
language-search = "benchmark_adapter:create_search_adapter"
```

The returned object needs:

- `embed_text(captions)` — list of caption strings in, `(N, D)` array out.
- `embed_images(descriptors)` — `(M, 512)` descriptor array in, `(M, D)` out.
- `prepare_database(image_ids, descriptors)` then `search(query, k)` — your
  database and query path, for the search component.

`D` is whatever your model uses; the benchmark never assumes a dimension, a
normalization, or an architecture. It also never looks for your weights:
your adapter loads them from your own repository, so commit them (a linear
map is well under a megabyte). If your object already exposes close course
names like `text_embedding` or `se_image`, the benchmark maps them and says
so in the run notes; anything it can't map by exact name produces a report
telling you what to add, never a guess.

## Data

The three course files you already have are the only inputs: the COCO 2014
captions, the ResNet-18 descriptor pickle, and the 200-d GloVe embeddings.
The benchmark verifies them by checksum and caches them. Files the course's
`cogworks-data` package already fetched are found and reused automatically;
you can also point `COGWORKS_LANGUAGE_DATA` at a folder that holds them.
No image files are needed to score a run.

The evaluation splits are fixed files in `language_search_benchmark/manifests/`,
built once with a recorded seed. The official run on the portal uses a
separate split, disjoint from these by construction.

## Running it

From your repo, with the course environment active:

```
python -m pip install -e .
cogworks check --benchmark language-search
cogworks test  --benchmark language-search   # small split, a couple of minutes
cogworks run   --benchmark language-search   # the public evaluation split
```

The run ends with ten demo captions and the image urls your system picked
for them. If those look right, that's your system working end to end.
