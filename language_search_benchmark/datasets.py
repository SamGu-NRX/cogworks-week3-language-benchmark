"""Pinned artifacts, the local cache, manifests, and case materialization.

Three course artifacts carry everything the benchmark needs (no image files;
COCO urls are only for display):

- ``captions_train2014.json`` — COCO 2014 captions: 82,783 images and
  414,113 annotations; the gold caption_id -> image_id map.
- ``resnet18_features.pkl`` — ``image_id -> (1, 512) float32`` descriptor
  dict covering 82,612 of those images (images without a descriptor are
  excluded everywhere, per the course).
- ``glove.6B.200d.txt.w2v`` — 200-d GloVe vectors in word2vec text format
  (distributed zipped; used by student adapters, never by the controller).

Artifacts are verified by size then sha256 and cached under a platformdirs
directory. ``COGWORKS_LANGUAGE_DATA`` may point at a directory that already
holds the files (students have them from the capstone); they are verified
once and used in place.

Manifests pin the evaluation splits by caption/image id with a fixed seed;
they are versioned files, never resampled at run time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from . import perturb
from .contracts import Resources

CACHE_APP = "cogworks-language-search"
CACHE_VERSION = "v1"
DATA_ENV = "COGWORKS_LANGUAGE_DATA"

#: Sizes and sha256 pinned 2026-07-24 from the fetched files; the GitHub
#: release assets and the CogWeb Dropbox links carry byte-identical files
#: (verified by hash from both sources). GitHub releases lead because the
#: course's own cogworks-data package fetches from them and Dropbox links
#: are unversioned.
_RELEASE = "https://github.com/rsokl/cog_data/releases/download/language-files/"

ARTIFACTS: Dict[str, Dict[str, Any]] = {
    "captions": {
        "filename": "captions_train2014.json",
        "urls": [
            _RELEASE + "captions_train2014.json",
            "https://www.dropbox.com/s/0e4fpk8wppyojyk/captions_train2014.json?dl=1",
        ],
        "size": 66782097,
        "sha256": "dd8c9636dc11740f956e36728866ea0c4ebe4988dcbdc5e712b7c2267f152d12",
    },
    "descriptors": {
        "filename": "resnet18_features.pkl",
        "urls": [
            _RELEASE + "resnet18_features.pkl",
            "https://www.dropbox.com/s/5gklm1ar3tz84rm/resnet18_features.pkl?dl=1",
        ],
        "size": 174540061,
        "sha256": "d56e267dacd39608b4aae581595f8fa8ec55457ab59b713df1eafa93e1023450",
    },
    "glove": {
        "filename": "glove.6B.200d.txt.w2v",
        "urls": [_RELEASE + "glove.6B.200d.txt.w2v"],
        "size": 693432839,
        "sha256": "dcee6ecdefebb5a884b23f2353561cc5f0527e59592525c0e3f4a31d81d91272",
        "archive": {
            "filename": "glove.6B.200d.txt.w2v.zip",
            "urls": ["https://www.dropbox.com/s/3clt5qi13fxkg3g/glove.6B.200d.txt.w2v.zip?dl=1"],
            "size": 264337080,
            "sha256": "6cbe88628045658c4175c50121b9ad6c61c39777ee42bdb19a255d26b0472b3a",
            "member": "glove.6B.200d.txt.w2v",
        },
    },
}

#: Where the course's cogworks-data package caches the same files; adopted
#: (after hash verification) so students never download twice.
COURSE_CACHE_APP = "cog_data"

GLOVE_KV_FILENAME = "glove.6B.200d.kv"

#: How many results the search component asks for; gold beyond this rank
#: scores zero for that query. Bounds the search payload.
SEARCH_K = 50


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheStatus:
    ready: bool
    path: Path
    message: str


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


@dataclass
class TextCase:
    """Caption pool for the text-understanding component.

    ``group_rows[i]`` is the pool row group (one group per source image) of
    caption ``i``; co-captions of the same image are the relevant items.
    ``None`` inside the evaluation sandbox.
    """

    kind: str
    captions: List[str]
    group_rows: Optional[List[int]]
    tie_break_seed: int


@dataclass
class RetrievalCase:
    """Query captions plus the descriptor pool for controller-side ranking.

    ``gold_rows[i]`` is the pool row of query ``i``'s true image; ``None``
    inside the evaluation sandbox.
    """

    kind: str
    queries: List[str]
    descriptors: np.ndarray
    gold_rows: Optional[List[int]]
    tie_break_seed: int
    #: Which query rewrite this case carries, matching SearchCase. "verbatim"
    #: is the caption exactly as it appears in the annotations file the
    #: submission is handed, which is why it is reported and not scored; see
    #: docs/decisions/week3-verbatim-probes.md.
    rung: str = "verbatim"


@dataclass
class SearchCase:
    """The same queries driven through the submission's own search path.

    ``rung`` names the rewrite applied to the queries. ``"verbatim"`` is the
    caption unchanged; the others rewrite it the way a person would type it.
    Since scorer version retrieval-v4 the three rewritten rungs are scored and
    ``search_mrr`` is their mean; ``verbatim`` is reported and not scored, so
    the search component measures whether end-to-end search survives the
    queries a user writes rather than whether it reproduces a caption it was
    handed. Each rung is still reported on its own as
    ``search_mrr_<rung>``; see ``perturb.py`` for what each one predicts.
    """

    kind: str
    queries: List[str]
    image_ids: List[int]
    descriptors: np.ndarray
    gold_image_ids: Optional[List[int]]
    k: int
    tie_break_seed: int
    rung: str = "verbatim"


# --------------------------------------------------------------------------
# Cache and downloads
# --------------------------------------------------------------------------


def cache_root() -> Path:
    override = os.environ.get(DATA_ENV)
    if override:
        return Path(override).expanduser()
    import platformdirs

    return Path(platformdirs.user_cache_path(CACHE_APP)) / CACHE_VERSION


def _state_path(root: Path) -> Path:
    return root / "cache-state.json"


def _load_state(root: Path) -> Dict[str, Any]:
    try:
        with open(str(_state_path(root)), "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return {}


def _save_state(root: Path, state: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(root), delete=False, suffix=".tmp"
    ) as stream:
        json.dump(dict(state), stream, indent=2, sort_keys=True)
        temp_name = stream.name
    os.replace(temp_name, str(_state_path(root)))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify(path: Path, size: int, sha256: str, root: Path, key: str) -> bool:
    """Size gate always; full hash once per file identity, then remembered."""

    if not path.is_file() or path.stat().st_size != size:
        return False
    state = _load_state(root)
    record = state.get("verified", {}).get(key)
    stamp = {"size": size, "mtime": int(path.stat().st_mtime)}
    if record and record.get("sha256") == sha256 and record.get("stamp") == stamp:
        return True
    digest = _sha256_file(path)
    if digest != sha256:
        return False
    state.setdefault("verified", {})[key] = {"sha256": sha256, "stamp": stamp}
    _save_state(root, state)
    return True


def _download(urls: Sequence[str], dest: Path, size: int, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as stream:
                temp_name = stream.name
                with urllib.request.urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            actual_size = os.path.getsize(temp_name)
            if actual_size != size:
                raise DatasetError(
                    "Downloaded {} has {} bytes; expected {}. The upstream file may have "
                    "changed; do not trust it.".format(dest.name, actual_size, size)
                )
            digest = _sha256_file(Path(temp_name))
            if digest != sha256:
                raise DatasetError(
                    "Downloaded {} fails its sha256 pin. The upstream file changed or the "
                    "download was corrupted; not keeping it.".format(dest.name)
                )
            os.replace(temp_name, str(dest))
            return
        except (OSError, DatasetError) as error:
            last_error = error
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    raise DatasetError(
        "Could not fetch {}: {}".format(dest.name, last_error)
    ) from last_error


def _adopt_from_course_cache(filename: str, dest: Path, size: int, sha256: str) -> bool:
    """Take a byte-identical file from the cogworks-data cache if present."""

    import platformdirs

    candidate = Path(platformdirs.user_cache_path(COURSE_CACHE_APP)) / filename
    if not candidate.is_file() or candidate.stat().st_size != size:
        return False
    if _sha256_file(candidate) != sha256:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(candidate), str(dest))
    except OSError:
        shutil.copyfile(str(candidate), str(dest))
    return True


def _ensure_glove(root: Path) -> Path:
    spec = ARTIFACTS["glove"]
    target = root / str(spec["filename"])
    if _verify(target, int(spec["size"]), str(spec["sha256"]), root, "glove"):
        return target
    if _adopt_from_course_cache(
        str(spec["filename"]), target, int(spec["size"]), str(spec["sha256"])
    ) and _verify(target, int(spec["size"]), str(spec["sha256"]), root, "glove"):
        return target
    try:
        _download(
            [str(url) for url in spec["urls"]],
            target,
            int(spec["size"]),
            str(spec["sha256"]),
        )
        if _verify(target, int(spec["size"]), str(spec["sha256"]), root, "glove"):
            return target
    except DatasetError:
        pass  # fall through to the zipped Dropbox copy
    archive = spec["archive"]
    zip_path = root / str(archive["filename"])
    if not _verify(zip_path, int(archive["size"]), str(archive["sha256"]), root, "glove-zip"):
        _download(
            [str(url) for url in archive["urls"]],
            zip_path,
            int(archive["size"]),
            str(archive["sha256"]),
        )
        if not _verify(zip_path, int(archive["size"]), str(archive["sha256"]), root, "glove-zip"):
            raise DatasetError("The GloVe archive failed verification after download.")
    with zipfile.ZipFile(str(zip_path)) as bundle:
        with tempfile.NamedTemporaryFile(dir=str(root), delete=False) as stream:
            temp_name = stream.name
            with bundle.open(str(archive["member"])) as member:
                shutil.copyfileobj(member, stream, length=1024 * 1024)
    os.replace(temp_name, str(target))
    if not _verify(target, int(spec["size"]), str(spec["sha256"]), root, "glove"):
        raise DatasetError("The extracted GloVe file failed verification.")
    # The zip is 264 MB of redundancy once the member is verified on disk.
    try:
        os.unlink(str(zip_path))
    except OSError:
        pass
    return target


def _missing_message(target: Path) -> str:
    """Distinguish "no file" from "file exists but fails its checksum"."""

    if target.is_file():
        return (
            "{} exists but does not match its checksum pin (wrong or corrupted "
            "copy). Delete it and run `cogworks test` to fetch a clean one.".format(target)
        )
    return (
        "{} is not cached. Run `cogworks test` to fetch it, or point {} at a "
        "directory that already has the course files.".format(target, DATA_ENV)
    )


def ensure_artifact(name: str, download: bool = True) -> Path:
    root = cache_root()
    if name == "glove":
        target = root / str(ARTIFACTS["glove"]["filename"])
        if _verify(
            target,
            int(ARTIFACTS["glove"]["size"]),
            str(ARTIFACTS["glove"]["sha256"]),
            root,
            "glove",
        ):
            return target
        if not download:
            raise DatasetError(_missing_message(target))
        return _ensure_glove(root)
    spec = ARTIFACTS[name]
    target = root / str(spec["filename"])
    if _verify(target, int(spec["size"]), str(spec["sha256"]), root, name):
        return target
    if not download:
        raise DatasetError(_missing_message(target))
    if _adopt_from_course_cache(
        str(spec["filename"]), target, int(spec["size"]), str(spec["sha256"])
    ) and _verify(target, int(spec["size"]), str(spec["sha256"]), root, name):
        return target
    _download([str(url) for url in spec["urls"]], target, int(spec["size"]), str(spec["sha256"]))
    if not _verify(target, int(spec["size"]), str(spec["sha256"]), root, name):
        raise DatasetError("{} failed verification after download.".format(spec["filename"]))
    return target


def ensure_glove_kv(glove_path: Path) -> Optional[Path]:
    """Build the fast-loading gensim ``.kv`` cache once, if gensim is present.

    The controller itself never needs GloVe, so a missing gensim is not an
    error here; adapters then fall back to the slow text parse.
    """

    kv_path = glove_path.parent / GLOVE_KV_FILENAME
    npy_path = kv_path.with_name(kv_path.name + ".vectors.npy")
    if kv_path.is_file():
        return kv_path
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        return None
    vectors = KeyedVectors.load_word2vec_format(str(glove_path), binary=False)
    vectors.save(str(kv_path))
    # gensim writes kv (+ .npy sidecar for large arrays); nothing to clean up.
    del npy_path
    return kv_path


def build_resources(download: bool = True, build_kv: bool = True) -> Resources:
    captions = ensure_artifact("captions", download)
    descriptors = ensure_artifact("descriptors", download)
    glove = ensure_artifact("glove", download)
    kv_path: Optional[Path] = None
    if build_kv:
        kv_path = ensure_glove_kv(glove)
    else:
        candidate = glove.parent / GLOVE_KV_FILENAME
        kv_path = candidate if candidate.is_file() else None
    return Resources(
        captions_path=captions,
        descriptors_path=descriptors,
        glove_path=glove,
        glove_kv_path=kv_path,
    )


def artifact_status() -> CacheStatus:
    root = cache_root()
    missing: List[str] = []
    for name, spec in sorted(ARTIFACTS.items()):
        target = root / str(spec["filename"])
        if not _verify(target, int(spec["size"]), str(spec["sha256"]), root, name):
            missing.append(str(spec["filename"]))
    if missing:
        return CacheStatus(
            ready=False,
            path=root,
            message="missing or unverified: {}".format(", ".join(missing)),
        )
    return CacheStatus(ready=True, path=root, message="ready")


# --------------------------------------------------------------------------
# Manifests and materialization
# --------------------------------------------------------------------------


def load_manifest(tier: str) -> Dict[str, Any]:
    if tier not in ("test", "evaluation"):
        raise ValueError("Public tier must be 'test' or 'evaluation'.")
    name = "public-{}.json".format(tier)
    from importlib import resources as importlib_resources

    package = "language_search_benchmark.manifests"
    try:
        text = importlib_resources.files(package).joinpath(name).read_text(encoding="utf-8")
    except AttributeError:  # Python 3.8
        with importlib_resources.open_text(package, name, encoding="utf-8") as stream:
            text = stream.read()
    return json.loads(text)


def caption_maps(captions_blob: Mapping[str, Any]) -> Dict[str, Dict[int, Any]]:
    caption_text: Dict[int, str] = {}
    caption_image: Dict[int, int] = {}
    for annotation in captions_blob["annotations"]:
        caption_text[int(annotation["id"])] = str(annotation["caption"])
        caption_image[int(annotation["id"])] = int(annotation["image_id"])
    image_url: Dict[int, str] = {
        int(image["id"]): str(image.get("coco_url") or image.get("flickr_url") or "")
        for image in captions_blob["images"]
    }
    return {"caption_text": caption_text, "caption_image": caption_image, "image_url": image_url}


def materialize_cases(
    manifest: Mapping[str, Any], resources: Resources
) -> List[Any]:
    """Build the three component cases (with gold) from a manifest."""

    maps = caption_maps(resources.load_captions())
    caption_text = maps["caption_text"]
    caption_image = maps["caption_image"]
    descriptors_blob = resources.load_descriptors()
    seed = int(manifest["tie_break_seed"])

    text_block = manifest["text"]
    text_caption_ids = [int(value) for value in text_block["caption_ids"]]
    text_captions = [caption_text[cid] for cid in text_caption_ids]
    group_of_image: Dict[int, int] = {}
    group_rows: List[int] = []
    for cid in text_caption_ids:
        image_id = caption_image[cid]
        group_rows.append(group_of_image.setdefault(image_id, len(group_of_image)))

    query_block = manifest["queries"]
    query_caption_ids = [int(value) for value in query_block["query_caption_ids"]]
    pool_image_ids = [int(value) for value in query_block["pool_image_ids"]]
    pool_row = {image_id: row for row, image_id in enumerate(pool_image_ids)}
    queries = [caption_text[cid] for cid in query_caption_ids]
    gold_image_ids = [caption_image[cid] for cid in query_caption_ids]
    missing_gold = [iid for iid in gold_image_ids if iid not in pool_row]
    if missing_gold:
        raise DatasetError(
            "Manifest is inconsistent: {} query images are not in the pool.".format(
                len(missing_gold)
            )
        )
    gold_rows = [pool_row[iid] for iid in gold_image_ids]
    matrix = np.zeros((len(pool_image_ids), 512), dtype=np.float32)
    missing_desc = []
    for row, image_id in enumerate(pool_image_ids):
        value = descriptors_blob.get(image_id)
        if value is None:
            missing_desc.append(image_id)
            continue
        matrix[row] = np.asarray(value, dtype=np.float32).reshape(-1)
    if missing_desc:
        raise DatasetError(
            "Manifest is inconsistent: {} pool images have no descriptor.".format(
                len(missing_desc)
            )
        )

    return attach_gold(
        build_cases(
            text_captions=text_captions,
            queries=queries,
            pool_image_ids=pool_image_ids,
            pool_descriptors=matrix,
            tie_break_seed=seed,
            search_k=SEARCH_K,
        ),
        text_group_rows=group_rows,
        retrieval_gold_rows=gold_rows,
        search_gold_image_ids=gold_image_ids,
    )


def build_cases(
    *,
    text_captions: Sequence[str],
    queries: Sequence[str],
    pool_image_ids: Sequence[int],
    pool_descriptors: np.ndarray,
    tie_break_seed: int,
    search_k: int,
) -> List[Any]:
    """The week's case grid, gold-free, from values already resolved.

    One place builds this list. The controller reaches it through
    `materialize_cases`, which resolves a manifest first; the evaluation
    sandbox reaches it from the payload zip, which cannot carry a manifest
    because gold is `caption_image[caption_id]` and the captions file is baked
    into the image. Both used to assemble the grid themselves and drifted: the
    sandbox rebuilt the search rewrites and not the retrieval ones, so it ran
    six cases against a nine-case tier. Scoring pairs outputs to cases by
    position, so that is not a recoverable difference.

    Keyword-only on purpose. The three sequence arguments are the same type,
    and every failure this function exists to prevent is a positional one.

    Two sharing rules, both measured rather than assumed:

    * Every search case holds the same `image_ids` list and the same
      descriptor array, because `drivers.run_with_adapter` decides whether to
      rebuild a submission's index by comparing those objects by identity. A
      fresh list per rung made the sandbox build the index four times where
      the local run built it once, and on an append-style prepare that moved
      the score (search_mrr_truncated 0.406667 local against 0.250000 hosted).
    * Search works on a copy, so it does not share an array with retrieval.
      Nothing in the contract asks a submission's `prepare_database` to leave
      the pool alone, and the retrieval rewrites are ranked after it runs;
      sharing one array let an in-place prepare move all three of them.
    """

    seed = int(tie_break_seed)
    queries = list(queries)
    rewrites = [rung for rung in perturb.RUNGS if rung != "verbatim"]
    search_image_ids = [int(value) for value in pool_image_ids]
    search_descriptors = np.asarray(pool_descriptors, dtype=np.float32).copy()

    def retrieval(rung: str) -> Any:
        return RetrievalCase(
            kind="retrieval",
            queries=perturb.rewrite_all(queries, rung),
            descriptors=pool_descriptors,
            gold_rows=None,
            tie_break_seed=seed,
            rung=rung,
        )

    def search(rung: str) -> Any:
        return SearchCase(
            kind="search",
            queries=perturb.rewrite_all(queries, rung),
            image_ids=search_image_ids,
            descriptors=search_descriptors,
            gold_image_ids=None,
            k=int(search_k),
            tie_break_seed=seed,
            rung=rung,
        )

    return (
        [
            TextCase(
                kind="text",
                captions=list(text_captions),
                group_rows=None,
                tie_break_seed=seed,
            ),
            retrieval("verbatim"),
            search("verbatim"),
        ]
        + [retrieval(rung) for rung in rewrites]
        + [search(rung) for rung in rewrites]
    )


def attach_gold(
    cases: Sequence[Any],
    *,
    text_group_rows: Sequence[int],
    retrieval_gold_rows: Sequence[int],
    search_gold_image_ids: Sequence[int],
) -> List[Any]:
    """The same cases with the answers on, for the controller that scores them.

    Separate from `build_cases` so the sandbox path cannot produce gold: it
    calls the constructor and never this. Every rewrite shares the verbatim
    case's answers, because a rewrite changes the query text and not which
    image is correct.

    Maps over the input in order rather than rebuilding the list, so the order
    is stated once, in `build_cases`.
    """

    group_rows = [int(value) for value in text_group_rows]
    gold_rows = [int(value) for value in retrieval_gold_rows]
    gold_image_ids = [int(value) for value in search_gold_image_ids]
    restored: List[Any] = []
    for case in cases:
        kind = getattr(case, "kind", "?")
        if kind == "text":
            if len(group_rows) != len(case.captions):
                raise ValueError("Gold text groups do not match the caption count.")
            restored.append(replace(case, group_rows=group_rows))
        elif kind == "retrieval":
            if len(gold_rows) != len(case.queries):
                raise ValueError("Gold rows do not match the query count.")
            restored.append(replace(case, gold_rows=gold_rows))
        elif kind == "search":
            if len(gold_image_ids) != len(case.queries):
                raise ValueError("Gold rows do not match the query count.")
            restored.append(replace(case, gold_image_ids=gold_image_ids))
        else:
            raise ValueError("Unknown case kind {!r}.".format(kind))
    return restored


def tier_status(tier: str) -> CacheStatus:
    try:
        load_manifest(tier)
    except (OSError, ValueError, KeyError) as error:
        return CacheStatus(
            ready=False, path=cache_root(), message="manifest unreadable: {}".format(error)
        )
    return artifact_status()


def assert_disjoint(manifests: Iterable[Mapping[str, Any]]) -> None:
    """Queries and pools must not overlap across split manifests."""

    seen_queries: Dict[int, str] = {}
    seen_pool: Dict[int, str] = {}
    for manifest in manifests:
        label = str(manifest.get("manifest_id", "?"))
        for cid in manifest["queries"]["query_caption_ids"]:
            previous = seen_queries.get(int(cid))
            if previous is not None and previous != label:
                raise DatasetError(
                    "Caption {} appears in query sets of {} and {}.".format(cid, previous, label)
                )
            seen_queries[int(cid)] = label
        for iid in manifest["queries"]["pool_image_ids"]:
            previous = seen_pool.get(int(iid))
            if previous is not None and previous != label:
                raise DatasetError(
                    "Image {} appears in pools of {} and {}.".format(iid, previous, label)
                )
            seen_pool[int(iid)] = label
