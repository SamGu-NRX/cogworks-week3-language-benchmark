"""What Week 3 asks for, described so a repository can be searched for it.

The capstone is four surfaces over one embedding space: captions become
vectors, ResNet descriptors become vectors in that same space, an image
database is built from ids and descriptors, and a query string comes back as
ranked image ids. Every team writes those under their own names, in their own
files, so this file says what each step does in terms of what goes in and what
comes back and never in terms of what it is called.

Three things here are not stage shapes and are worth reading first.

``compute_idfs`` and its cousins take the whole caption corpus and return a
table their embedder then needs on every call. That is a side input, not a
link in the chain, which is what ``Stage.fit`` is for: it runs once against
the benchmark's corpus and joins the extras pool. The corpus is ours to hand
over; the formula is theirs.

The trained ``W_embed`` lives in the repository as a file, and finding it is
neither a stage nor a resource the benchmark owns. ``weights_in`` does that
search: every array-shaped file under the root, loaded in a child process so a
malformed pickle cannot take the run with it, kept when it has the shape a
512-to-D projection has, and disambiguated only by the path their own source
names in a string literal. Never by mtime, and never by "the largest one":
either their code names the file or the repository is ambiguous and says so.

``accepts`` is the week's own end-to-end test and the only thing that may
accept a binding. It runs the benchmark's nine driver cases on the composed
object and requires every case the bound branches cover to come back ok. It
is deliberately not a score threshold: a team whose embedding is weak should
be scored badly by the benchmark, not refused by discovery.
"""

from __future__ import annotations

import ast
import inspect
import fnmatch
import os
import subprocess
import sys
from dataclasses import replace, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cogbench.pipeline import Role, Stage

__all__ = [
    "AmbiguousWeights",
    "Weights",
    "looks_like_idf_table",
    "looks_like_matrix_for",
    "looks_like_token_lists",
    "looks_like_W",
    "looks_like_store",
    "looks_like_id_list",
    "as_matrix",
    "search_role",
    "accepts",
    "weights_in",
    "weight_files",
    "named_in_source",
    "save_call",
    "gitignore_line",
    "weights_diagnostic",
    "TEXT_BRANCH",
    "IMAGE_BRANCH",
    "PREPARE_BRANCH",
    "SEARCH_BRANCH",
]


#: The descriptor width every submission is handed. A projection out of it is
#: the only thing that can be the image half of the shared space, which is why
#: it is the one number `looks_like_W` insists on.
DESCRIPTOR_WIDTH = 512

#: The embedding widths a shared space may plausibly have. Wide enough for the
#: course's 200 and for a team that picked 50 or 300; narrow enough that a
#: (512, 512) autoencoder weight and a (512, 82612) score table are not
#: mistaken for a projection.
MIN_WIDTH = 8
MAX_WIDTH = 512

#: Extensions a trained projection is saved under. Read from the corpus:
#: Lashika writes `data/W_embed.npy`, Bagel pickles `[W, b]` under
#: `model_tests/*.pkl`, and the course's own notebooks use `.npz`.
WEIGHT_SUFFIXES = (".npy", ".npz", ".pkl", ".pt", ".pth")

#: A file larger than this is not opened. `model_tests/trainingdb_large.pkl`
#: is 31 MB and loads in a second, but the same glob reaches 174 MB descriptor
#: dumps and 693 MB GloVe exports in other checkouts, and unpickling those to
#: learn they are not a weight matrix costs more than the search is worth.
MAX_WEIGHT_BYTES = 200 * 1024 * 1024

#: How long one candidate file may take to load. It runs in a child process,
#: so a pickle that hangs or aborts the interpreter costs this and nothing
#: else.
LOAD_TIMEOUT_SECONDS = 20

#: Directories never walked when looking for weights.
_SKIP_DIRECTORIES = (".git", "__pycache__", ".venv", "venv", "node_modules")


class AmbiguousWeights(RuntimeError):
    """Several files could be the trained projection and their code names none.

    Deliberately an error rather than a choice. Picking the newest file would
    bind differently against a squashed clone of the same repository, and
    picking the largest would be a guess about their training script. The
    image side is refused and every candidate is listed, which is a thing a
    student can act on in one commit.
    """

    def __init__(self, candidates: Sequence[Path]) -> None:
        self.candidates = [Path(path) for path in candidates]
        super().__init__(
            "several files could be the trained image projection and no source "
            "file names one of them: {}".format(
                ", ".join(sorted(str(path) for path in self.candidates))
            )
        )


@dataclass(frozen=True)
class Weights:
    """The trained projection this repository committed, and where it came from."""

    path: Path
    #: The (512, D) projection itself.
    matrix: np.ndarray
    #: The bias saved beside it, when their save wrote a pair.
    bias: Optional[np.ndarray] = None
    #: ``file:line`` of the string literal in their own source that names this
    #: file, when one exists. Empty when there was only one candidate and
    #: nothing had to be disambiguated.
    cited: str = ""


# --------------------------------------------------------------------------
# What each stage's output looks like, loosely
# --------------------------------------------------------------------------


def as_matrix(value: Any) -> Optional[np.ndarray]:
    """The two-dimensional float array in a value, or None.

    Three representations, all from the corpus and all the same answer. A
    plain ndarray (Lashika's `embed_captions_batch`). A mygrad `Tensor`, which
    `np.asarray` reads without copying the graph (Bagel's `ImageToCaption`,
    CogFinder's `Model`, rutvim's `ImageEmbedder`). And a list of one-row
    arrays, which is what a per-item embedder gathers into.
    """

    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            rows = [np.asarray(row, dtype=np.float64).reshape(-1) for row in value]
            if not rows or len({row.shape[0] for row in rows}) != 1:
                return None
            array = np.stack(rows)
        else:
            array = np.asarray(getattr(value, "data", value))
    except BaseException:  # noqa: BLE001 - a student value may be anything
        return None
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        return None
    return array.astype(np.float64, copy=False)


def looks_like_idf_table(value: Any) -> bool:
    """A non-empty mapping of word to number.

    All four repositories return exactly this from `compute_idfs`,
    `compute_idf`, and `find_idfs`. The values are checked rather than assumed
    because their vocabulary builders also return word-to-count dictionaries,
    and a count table handed to their embedder would weight every word by how
    common it is, which is the opposite of what they wrote.
    """

    if not isinstance(value, dict) or not value:
        return False
    whole = True
    for word, weight in value.items():
        if not isinstance(word, str):
            return False
        if isinstance(weight, bool) or not isinstance(weight, (int, float, np.floating, np.integer)):
            return False
        if isinstance(weight, (float, np.floating)) and not float(weight).is_integer():
            whole = False
    # A table whose every weight is a whole number is a count table, not an
    # IDF table: log10(N / count) is whole only when N / count is a power of
    # ten, which no vocabulary manages for every word. The first draft let
    # ints through, so the check above did not do what its own docstring
    # says, and rutvim's module-scope `vocab` (a defaultdict of counts)
    # would have been offered ahead of their `idf`.
    return not whole


def looks_like_matrix_for(rows: int):
    """One row per item, in a width a shared embedding space could have.

    Loose on purpose, in both directions. The width is a range rather than the
    course's 200 because the benchmark never assumes a dimension, and the
    contents are only checked for being finite, because a caption made
    entirely of unseen words is supposed to embed as zeros.
    """

    def check(value: Any) -> bool:
        array = as_matrix(value)
        if array is None or array.ndim != 2 or array.shape[0] != rows:
            return False
        width = array.shape[1]
        if not MIN_WIDTH <= width <= MAX_WIDTH:
            return False
        return bool(np.isfinite(array).all())

    return check


def looks_like_token_lists(value: Any) -> bool:
    """One list of words per item.

    The step before embedding for a team that wrote them separately: rutvim's
    `caption_processor` turns one caption into its tokens and their
    `embed_text` takes those tokens, not the caption.
    """

    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, list) or not item:
            return False
        if not all(isinstance(word, str) for word in item):
            return False
    return True


def looks_like_W(value: Any) -> bool:
    """A 512-to-D projection, however it was saved.

    Four shapes, all measured. A bare `(512, D)` array is Lashika's
    `data/W_embed.npy`. A `[W, b]` list is what Bagel's `ImageToCaption.save`
    writes and what CogFinder's `save_model` would write. A mapping or an
    `.npz` holding one of those covers the course notebook's own save. And a
    class with a `load` method that is callable is the fourth: Bagel's model
    class is the only thing in that repository that can install its own
    pickled parameters, so the class is a way of applying W as much as the
    array is.
    """

    if isinstance(value, type):
        loader = getattr(value, "load", None)
        return callable(loader) and callable(getattr(value, "__call__", None))
    if isinstance(value, (list, tuple)):
        return bool(value) and looks_like_W(value[0])
    if isinstance(value, dict):
        return any(looks_like_W(item) for item in value.values())
    if hasattr(value, "files") and hasattr(value, "__getitem__"):
        # An `.npz` archive, read by the names it carries.
        try:
            return any(looks_like_W(value[name]) for name in value.files)
        except BaseException:  # noqa: BLE001 - a malformed archive is not a W
            return False
    array = as_matrix(value)
    if array is None or array.ndim != 2:
        return False
    return (
        array.shape[0] == DESCRIPTOR_WIDTH
        and MIN_WIDTH <= array.shape[1] <= MAX_WIDTH
        and bool(np.isfinite(array).all())
    )


def looks_like_store(value: Any) -> bool:
    """Anything at all. Kept for callers that only need a non-None value."""

    return value is not None


def looks_like_store_of(image_ids: Sequence[Any]):
    """Something that kept the images it was handed.

    A database is whatever their prepare step returned, and the corpus writes
    it as an object (Lashika's `ImageDatabase`, Bagel's `CaptionImageQuery`)
    and as a dict keyed by id (rutvim). What they share is not a type but a
    fact: the ids the benchmark passed in are in there, or one row per id
    is. That is the check, read off the value's own contents to depth two
    (attributes, dict keys and values, tuple elements), and nothing else.

    It exists because "anything at all" let CogFinder's
    `LanguageModels.generate_letter(ids, descriptors)` bind as the prepare
    step: it returned a letter, the stage accepted it, and the search branch
    then reported that nothing took the query. A store that forgot the
    images is not a store, and the search step is the only other thing that
    could have said so.
    """

    wanted = set()
    for item in image_ids:
        try:
            wanted.add(int(item))
        except (TypeError, ValueError):
            return lambda _value: False
    count = len(list(image_ids))

    def holds(part: Any) -> bool:
        if isinstance(part, dict):
            keys = set()
            for key in part:
                try:
                    keys.add(int(key))
                except (TypeError, ValueError):
                    return False
            return keys == wanted
        if isinstance(part, (set, frozenset)):
            try:
                return {int(item) for item in part} == wanted
            except (TypeError, ValueError):
                return False
        array = as_matrix(part)
        if array is not None and array.ndim == 2:
            return array.shape[0] == count
        if isinstance(part, np.ndarray) and part.ndim == 1:
            try:
                return {int(item) for item in part.tolist()} == wanted
            except (TypeError, ValueError):
                return False
        if isinstance(part, (list, tuple)) and len(part) == count:
            try:
                return {int(item) for item in part} == wanted
            except (TypeError, ValueError):
                return False
        return False

    def parts_of(value: Any) -> List[Any]:
        if isinstance(value, dict):
            return [value] + list(value.values())
        if isinstance(value, (list, tuple)):
            return [value] + list(value)
        found: List[Any] = [value]
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            found.extend(state.values())
        return found

    def check(value: Any) -> bool:
        if value is None or isinstance(value, (str, bytes)):
            return False
        for part in parts_of(value):
            if holds(part):
                return True
            for inner in parts_of(part):
                if inner is not part and holds(inner):
                    return True
        return False

    return check


def looks_like_id_list(pool: Sequence[Any]):
    """Ids drawn from the pool the benchmark just handed over.

    The one place a Week 3 validator can be strict, because the ids are ours:
    a search that answers with something outside the pool has not answered.
    Rows of dicts are read for an id, since Bagel's `search` returns
    `{"image_id": ..., "score": ...}` per hit and the id is what the ranking
    already chose.
    """

    allowed = set()
    for item in pool:
        try:
            allowed.add(int(item))
        except (TypeError, ValueError):
            return lambda _value: False

    def check(value: Any) -> bool:
        if value is None or isinstance(value, (str, bytes, dict)):
            return False
        try:
            items = list(value)
        except TypeError:
            return False
        if not items:
            return False
        for item in items:
            if isinstance(item, dict):
                item = item.get("image_id", item.get("id"))
            try:
                if int(item) not in allowed:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    return check


# --------------------------------------------------------------------------
# The trained projection, found in the repository rather than supplied
# --------------------------------------------------------------------------


def weight_files(root: Path) -> List[Path]:
    """Every file under ``root`` that could hold a trained projection.

    Sorted by path so two runs enumerate the same candidates in the same
    order, which is what makes the tiebreak below reproducible.
    """

    root = Path(root)
    found: List[Path] = []
    for folder, directories, names in os.walk(str(root)):
        directories[:] = sorted(d for d in directories if d not in _SKIP_DIRECTORIES)
        for name in sorted(names):
            if not name.endswith(WEIGHT_SUFFIXES):
                continue
            path = Path(folder) / name
            try:
                if path.is_symlink() or path.stat().st_size > MAX_WEIGHT_BYTES:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


#: The child program. It loads one file and answers a single question, so a
#: pickle that imports half a training stack, hangs, or aborts the interpreter
#: costs one subprocess rather than the run.
_PROBE = (
    "import sys, json\n"
    "sys.path[:0] = json.loads(sys.argv[1])\n"
    "from language_search_benchmark.roles import looks_like_W, load_weight_file\n"
    "print('yes' if looks_like_W(load_weight_file(sys.argv[2])) else 'no')\n"
)


def load_weight_file(path: str) -> Any:
    """Read one candidate file, whatever it was saved with."""

    name = str(path)
    if name.endswith(".npy") or name.endswith(".npz"):
        return np.load(name, allow_pickle=True)
    if name.endswith(".pkl"):
        import pickle

        with open(name, "rb") as stream:
            return pickle.load(stream)
    import pickle

    try:
        import torch  # noqa: PLC0415 - optional, and only for .pt/.pth

        return torch.load(name, map_location="cpu")
    except ImportError:
        with open(name, "rb") as stream:
            return pickle.load(stream)


def _holds_weights(path: Path, search_path: Sequence[str]) -> bool:
    """Whether this file holds a projection, decided in a child process."""

    import json

    try:
        done = subprocess.run(
            [sys.executable, "-c", _PROBE, json.dumps(list(search_path)), str(path)],
            capture_output=True,
            text=True,
            timeout=LOAD_TIMEOUT_SECONDS,
            cwd=str(path.parent),
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return done.returncode == 0 and done.stdout.strip().endswith("yes")


_LOAD_WORDS = ("load", "read", "open")


def named_in_source(root: Path, basename: str) -> List[Tuple[Path, int, bool]]:
    """Where their own code LOADS this file, as ``(path, line, at_module_scope)``.

    An AST walk rather than a text search: a basename inside a comment or a
    README is not their code loading the file, and a docstring that mentions
    `W_embed.npy` is not a load call either. Only a string constant that is
    an argument to a call whose name says it loads counts, matched by
    basename because their path is written from wherever they run.

    A save call does not count, and the distinction decides one repository.
    Bagel's `model_tests/test_db.py` saves `test2.pkl` on line 53 and loads
    `test1.pkl` on line 63 inside a round-trip test, while their application
    script `get_model_embeddings.py:13` loads `testd_50.pkl` at module scope
    with the comment "change filename if needed". Counting every literal made
    three files look equally named and refused the repository as ambiguous;
    counting only loads leaves two, and the third value carried here, whether
    the call runs when the file runs, is what separates the script they run
    from a test function nobody calls.
    """

    root = Path(root)
    hits: List[Tuple[Path, int, bool]] = []
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        if any(part in _SKIP_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        inside_def: set = set()
        for scope in ast.walk(tree):
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for inner in ast.walk(scope):
                    inside_def.add(id(inner))
        never_run = _dead_when_loaded(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee(node.func).lower()
            if not any(word in callee for word in _LOAD_WORDS):
                continue
            for argument in list(node.args) + [word.value for word in node.keywords]:
                if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                    continue
                if argument.value.replace("\\", "/").rsplit("/", 1)[-1] == basename:
                    runs = id(node) not in inside_def and id(node) not in never_run
                    hits.append((path, node.lineno, runs))
                    break
    return hits


def _dead_when_loaded(tree: ast.Module) -> set:
    """Ids of the nodes under a module-scope `if` that cannot run.

    Only the simplest guard is read: a comparison between a name assigned a
    literal earlier in the same file and another literal. That guard is a
    switch a student flips by editing one line, and it decides one
    repository. Bagel's `model_tests/test_db.py` sets `t = 0` on line 17,
    trains and saves `test2.pkl` under `if t == 0`, and loads `test1.pkl`
    under `elif t == 1`. Both loads are at module scope, and reading them as
    equals made the repository ambiguous between `test1.pkl` and the
    `testd_50.pkl` their application script loads unconditionally. Anything
    the folder cannot decide (`if __name__ == "__main__"`, a name bound by
    a call) is left alone and counts as running.
    """

    dead: set = set()
    env: Dict[str, Any] = {}

    def fold(test: ast.AST, env: Dict[str, Any]) -> Optional[bool]:
        if isinstance(test, ast.Constant):
            return bool(test.value)
        if isinstance(test, ast.Name):
            return bool(env[test.id]) if test.id in env else None
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = fold(test.operand, env)
            return None if inner is None else not inner
        if isinstance(test, ast.BoolOp):
            parts = [fold(part, env) for part in test.values]
            # Short-circuit: one false operand makes an `and` false and one
            # true operand makes an `or` true, whatever the others are. The
            # first draft returned unknown whenever any operand was, so
            # `if t == 0 and x:` with `t = 1` read as live.
            if isinstance(test.op, ast.And):
                if any(part is False for part in parts):
                    return False
                return None if any(part is None for part in parts) else True
            if any(part is True for part in parts):
                return True
            return None if any(part is None for part in parts) else False
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left = literal(test.left, env)
            right = literal(test.comparators[0], env)
            if left is _UNKNOWN or right is _UNKNOWN:
                return None
            op = test.ops[0]
            try:
                if isinstance(op, ast.Eq):
                    return left == right
                if isinstance(op, ast.NotEq):
                    return left != right
                if isinstance(op, ast.Lt):
                    return left < right
                if isinstance(op, ast.LtE):
                    return left <= right
                if isinstance(op, ast.Gt):
                    return left > right
                if isinstance(op, ast.GtE):
                    return left >= right
            except TypeError:
                return None
        return None

    def literal(node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        return _UNKNOWN

    def walk(statements: Sequence[ast.stmt], env: Dict[str, Any]) -> Dict[str, Any]:
        env = dict(env)
        for statement in statements:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    if isinstance(statement.value, ast.Constant):
                        env[target.id] = statement.value.value
                    else:
                        env.pop(target.id, None)
            elif isinstance(statement, ast.If):
                verdict = fold(statement.test, env)
                if verdict is False:
                    for inner in statement.body:
                        dead.update(id(node) for node in ast.walk(inner))
                    env = walk(statement.orelse, env)
                elif verdict is True:
                    for inner in statement.orelse:
                        dead.update(id(node) for node in ast.walk(inner))
                    env = walk(statement.body, env)
                else:
                    # Either branch may run. Each is walked from the same
                    # starting point and only a name both leave with the same
                    # literal survives; the first draft let the else branch's
                    # assignment decide a later guard, so a load under
                    # `if choice == 0` read as dead when `choice` depended
                    # on something the folder could not see.
                    left = walk(statement.body, env)
                    right = walk(statement.orelse, env)
                    env = {
                        name: value
                        for name, value in left.items()
                        if name in right and right[name] == value
                    }
        return env

    walk(tree.body, env)
    return dead


_UNKNOWN = object()


def weights_in(root: Path, search_path: Sequence[str] = ()) -> Optional[Weights]:
    """The trained projection this repository committed, or None.

    Returns None when nothing under the root loads as one, which is the answer
    for two of the four audited 2026 repositories: CogFinder's
    `results/modelweights.pkl` was never committed and rutvim's
    `image_embedder_weights.pkl` does not exist in the tree or in its history.
    Their image side is then unmeasured, which is a different claim from an
    image side that scored badly, and `plugins.score` keeps those two apart.

    Raises ``AmbiguousWeights`` when several files load as a projection and
    their own source names none of them or names more than one. Bagel is the
    case that decides the rule: `model_tests/` holds `test1.pkl`, `test2.pkl`,
    and `testd_50.pkl`, all `[W (512, 200), b (1, 200)]`, and only
    `get_model_embeddings.py:13` says which one they meant.
    """

    root = Path(root)
    path_for_child = list(search_path) or [str(Path(__file__).resolve().parents[1])]
    holding = [path for path in weight_files(root) if _holds_weights(path, path_for_child)]
    if not holding:
        return None
    if len(holding) == 1:
        return _read(holding[0], "")

    # The file their code loads, and only that. When loads at module scope
    # exist, those decide: they are what runs when the script runs. Only when
    # no load is at module scope do loads inside functions count. Two
    # different files loaded at the same level is a disagreement the
    # instrument reports rather than resolves.
    loads: Dict[Path, List[Tuple[str, bool]]] = {}
    for path in holding:
        for source, line, at_module_scope in named_in_source(root, path.name):
            loads.setdefault(path, []).append(
                ("{}:{}".format(source.relative_to(root).as_posix(), line), at_module_scope)
            )
    top = {p: [c for c, at_top in cites if at_top] for p, cites in loads.items()}
    top = {p: cites for p, cites in top.items() if cites}
    chosen = top if top else {p: [c for c, _ in cites] for p, cites in loads.items()}
    if len(chosen) != 1:
        raise AmbiguousWeights(holding)
    path, citations = next(iter(chosen.items()))
    return _read(path, citations[0])


def loaded_model(modules: Sequence[Any], path: Path) -> Optional[Tuple[str, Any]]:
    """One of their model objects holding this weights file, or None.

    A team whose encoder is an object rather than a matrix saves its
    parameters with a method and reads them back with another: Bagel's
    `ImageToCaption()` builds for free, `.load(path)` fills its layer from
    their pickle, and the instance is what their scripts call on
    descriptors. The matrix inside is theirs and so is the object; the only
    thing supplied is the path. Any class of theirs that constructs with no
    arguments, has a `load` taking one argument, accepts this file, and is
    callable afterwards is that object. The first such in module order
    wins, which is deterministic and, on the audited corpus, unique.
    """

    from cogbench.pipeline import Candidate, _call, _reaches_outside

    # Three calls into their code, each under the search's own clock
    # (`pipeline._call`): a constructor, a loader, or an encoder that never
    # returns would otherwise hang the whole resolve, and catching
    # exceptions does nothing about a call that does not come back.
    for module in modules:
        module_name = getattr(module, "__name__", "?")
        for name in sorted(dir(module)):
            owner = getattr(module, name, None)
            if not isinstance(owner, type) or name.startswith("_"):
                continue
            if getattr(owner, "__module__", None) != module_name:
                continue
            loader = getattr(owner, "load", None)
            if not callable(loader) or _reaches_outside(loader):
                continue
            label = "{}.{}".format(module_name, name)
            ok, instance = _call(Candidate(label, owner, module_name, self_only=True), ())
            if not ok or instance is None or not callable(instance):
                continue
            ok, _ = _call(Candidate(label + ".load", instance.load, module_name), (str(path),))
            if not ok:
                continue
            # It has to encode. A class whose `load` swallowed the file
            # without complaint and whose call then fails on a descriptor
            # is not the encoder; the test fixture found that reading
            # `"not a list"[0]` gives `"n"` and nothing had objected.
            probe = np.zeros((1, DESCRIPTOR_WIDTH), dtype=np.float32)
            ok, encoded = _call(Candidate(label, instance, module_name), (probe,))
            if not ok or as_matrix(encoded) is None:
                continue
            return label, instance
    return None


def _read(path: Path, cited: str) -> Weights:
    """The projection and its bias, in this process, once one file has won."""

    loaded = load_weight_file(str(path))
    matrix, bias = _split(loaded)
    return Weights(path=path, matrix=matrix, bias=bias, cited=cited)


def _split(loaded: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(loaded, (list, tuple)) and loaded:
        first = np.asarray(loaded[0])
        rest = np.asarray(loaded[1]) if len(loaded) > 1 else None
        return first, rest
    if isinstance(loaded, dict):
        for value in loaded.values():
            if looks_like_W(value):
                return _split(value)
    if hasattr(loaded, "files"):
        for name in loaded.files:
            if looks_like_W(loaded[name]):
                return _split(loaded[name])
    return np.asarray(loaded), None


#: Callee names that mean "this line writes the model out". `dump` is here for
#: `pickle.dump`, which is how three of the four repositories save.
_SAVE_WORDS = ("save", "dump")


def save_call(root: Path) -> Optional[Tuple[str, int, str]]:
    """Where their own code writes the model, as ``(file, line, path)``.

    Read statically, because the point is to tell a student what their
    training script would have produced, and running their training script is
    a separate and much more expensive question. Only calls that name a path
    as a string literal are read; `pickle.dump(weights, f)` names a file
    object and says nothing a student could act on.
    """

    root = Path(root)
    found: List[Tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        if any(part in _SKIP_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee(node.func)
            if not name or not any(word in name.lower() for word in _SAVE_WORDS):
                continue
            for argument in list(node.args) + [word.value for word in node.keywords]:
                if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                    continue
                text = argument.value
                if not text.endswith(WEIGHT_SUFFIXES):
                    continue
                found.append(
                    (path.relative_to(root).as_posix(), node.lineno, text)
                )
                break
    return sorted(found)[0] if found else None


def _callee(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def gitignore_line(root: Path, target: str) -> Optional[int]:
    """The ``.gitignore`` line that keeps this path out of the repository.

    One-based, matching what an editor shows. Matched against the basename
    and against the path as written, so `*.pkl` and `results/` both answer for
    `results/modelweights.pkl`.
    """

    ignore = Path(root) / ".gitignore"
    try:
        lines = ignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    cleaned = target.replace("\\", "/").lstrip("./")
    basename = cleaned.rsplit("/", 1)[-1]
    for number, line in enumerate(lines, start=1):
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        pattern = pattern.rstrip("/")
        if not pattern:
            continue
        if fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(cleaned, pattern):
            return number
        if "/" in cleaned and fnmatch.fnmatch(cleaned.split("/", 1)[0], pattern):
            return number
    return None


def weights_diagnostic(root: Path) -> str:
    """One sentence naming, in their words, what the image side is missing.

    Read from their own save call and their own `.gitignore`, because "no
    trained weights were found" is true and useless, while "your training.py
    writes results/modelweights.pkl and *.pkl is ignored" is one commit away
    from being fixed.
    """

    lead = (
        "overall withheld: the image side has no trained weights to measure."
    )
    where = save_call(root)
    if where is None:
        return (
            lead
            + " Nothing under this repository loads as a (512, D) projection and no"
            " source file saves one, so there is no image embedding to score."
            " Commit the weights your training run produces and try again."
        )
    source, line, target = where
    sentence = "{} Your {} saves to {} ({}:{})".format(lead, source, target, source, line)
    number = gitignore_line(root, target)
    if number is not None:
        sentence += ", and it matches .gitignore line {}".format(number)
    return sentence + ". Commit that file and run again."


# --------------------------------------------------------------------------
# The role
# --------------------------------------------------------------------------


def text_branch(captions: Sequence[str]) -> Role:
    """Captions in, one row per caption out.

    Two stages rather than one because two of the audited repositories split
    it. rutvim's `caption_processor` turns a caption into tokens and their
    `embed_text` takes tokens, never the caption; Lashika's
    `embed_captions_batch` takes the captions and does both. The first stage
    is `fusible`, which is what lets the second be probed against the
    benchmark's own captions for the teams who wrote one function.
    """

    return Role(
        "text",
        (
            Stage(
                "tokens",
                prefers=("token", "caption_processor", "process", "strip"),
                produces=looks_like_token_lists,
                per_item=True,
                # Two of the four audited repositories embed straight from
                # the caption (Lashika's `embed_captions_batch`, CogFinder's
                # `embed_caption`), so this step is one their code may not
                # have. Marking it fusible lets the search probe the embed
                # step against the captions themselves; the first draft
                # marked it fusible and it still refused, because a fusible
                # first stage is only skipped when the SECOND stage is
                # probed against the fixture, and the second stage's own
                # `accepts` then refused a list of strings. Both flags are
                # needed and both are here.
                fusible=True,
            ),
            Stage(
                "text",
                prefers=("embed_caption", "embed_text", "text_embedding", "embed"),
                # A list of captions or a list of token lists. The tokens
                # step is skipped for a team who embeds from the caption,
                # and the search then hands this step the captions.
                accepts=lambda value: isinstance(value, (list, tuple)) and bool(value),
                produces=looks_like_matrix_for(len(captions)),
                per_item=True,
                extras=("glove", "idfs"),
            ),
        ),
        fixture=(list(captions),),
    )


def image_branch(descriptors: np.ndarray) -> Role:
    """ResNet descriptors in, one row per image out, in the text space.

    Declared with the weights in the extras pool under `W`, because every
    audited image encoder takes the projection as an argument or holds it on
    an object built with it.
    """

    return Role(
        "image",
        (
            Stage(
                "image",
                prefers=("descriptor_to_embedding", "embed_image", "image_embedding", "embed"),
                accepts=lambda value: as_matrix(value) is not None,
                produces=looks_like_matrix_for(int(np.asarray(descriptors).shape[0])),
                per_item=True,
                extras=("W", "weights_model"),
            ),
        ),
        fixture=(np.asarray(descriptors),),
    )


def prepare_forms(
    image_ids: Sequence[int], descriptors: Any, projected: Any = None
) -> List[Tuple[Any, Any]]:
    """The argument forms a prepare step may be offered, in a fixed order.

    One function for the search and for the scored run, because the form
    the search bound with is an index into this list and a scored run has
    to hand the step the same shape. Measured on Bagel before this was
    shared: the search bound `CaptionImageQuery(EMBEDDINGS, ids)` on form 3
    and the scored run built it from `(ids, descriptors)`, the constructor
    accepted either, and only the search step failed, one branch later.
    The projected forms come last and only when the image branch produced
    something, so the index of a raw form never moves.
    """

    ids = list(image_ids)
    raw = np.asarray(descriptors)
    forms: List[Tuple[Any, Any]] = [(ids, raw), (raw, ids)]
    if projected is not None:
        matrix = np.asarray(getattr(projected, "data", projected))
        forms.extend([(ids, matrix), (matrix, ids)])
    return forms


def prepare_branch(image_ids: Sequence[int], descriptors: np.ndarray) -> Role:
    """Ids and descriptors in, whatever they search against out.

    Four forms of the same input, because the corpus wrote all four. Lashika's
    `ImageDatabase(image_ids, descriptors, W)` takes the raw descriptors and
    projects them itself; Bagel's `CaptionImageQuery(image_embeddings,
    image_ids)` takes the PROJECTED matrix first and the ids second. The
    projected matrix is the image branch's output, which exists only once
    that branch has bound, so this fixture is made when the branch resolves
    and offers the projected forms only when the pool holds them.
    """

    from cogbench.pipeline import Fixtures

    ids = list(image_ids)
    raw = np.asarray(descriptors)

    def _forms(pool_values: Dict[str, Any], chains: Dict[str, Any]) -> Any:
        return Fixtures(tuple(prepare_forms(ids, raw, pool_values.get("image"))))

    return Role(
        "prepare",
        (
            Stage(
                "prepare",
                prefers=("database", "prepare", "build", "index", "query"),
                produces=looks_like_store_of(ids),
                extras=("W", "weights_model"),
            ),
        ),
        fixture=_forms,
    )


def search_branch(query: str, k: int, pool: Sequence[int]) -> Role:
    """A query in, ids from the pool out.

    The query is handed over EMBEDDED, by the text branch's own chain, not
    as a string. Every audited search step takes a vector (Lashika's
    `ImageDatabase.query(caption_embedding, k)`, Bagel's
    `CaptionImageQuery.search(caption_vector, top_k)`, rutvim's
    `query_database(caption_embedding, db, k)`), and probing them with the
    raw string was the one thing that stopped the whole role on Lashika
    after the other three branches had bound. So the fixture is made when
    the branch is resolved, from the text chain the search already found.
    """

    def _embedded_query(pool_values: Dict[str, Any], chains: Dict[str, Any]) -> Any:
        steps = chains["text"]
        value: Any = [query]
        for step in steps:
            value = step.bound(value)
        rows = np.asarray(getattr(value, "data", value), dtype=np.float64)
        row = rows[0] if rows.ndim == 2 else rows
        return (row, k)

    return Role(
        "search",
        (
            Stage(
                "search",
                prefers=("search", "query", "find", "top"),
                produces=looks_like_id_list(pool),
                extras=("prepare", "glove", "idfs"),
            ),
        ),
        fixture=_embedded_query,
    )


#: The four surfaces, for a reader and for the tests that check their shapes.
#: `search_role` is what discovery actually resolves, and it carries only the
#: branches this engine can bind; see the note there.
TEXT_BRANCH = text_branch
IMAGE_BRANCH = image_branch
PREPARE_BRANCH = prepare_branch
SEARCH_BRANCH = search_branch


def search_role(
    captions: Sequence[str],
    corpus: Sequence[str],
    descriptors: Any,
    image_ids: Sequence[int],
    query: str,
    k: int,
) -> Role:
    """The week's task: an IDF table computed once, then four surfaces.

    The IDF stage is `fit`: nothing downstream takes its value as input and
    everything downstream takes it alongside one, so it runs once against the
    caption corpus the benchmark owns and joins the extras pool under `idfs`.
    That name is load-bearing for one repository: CogFinder's
    `embed_caption(caption, idfs)` cannot take the table positionally next to
    GloVe, and the only shape that binds it is the keyword one, which matches
    on the parameter's own name.

    Text is required. Image, prepare, and search are optional, which is the
    decided policy for a repository with no trained weights (see
    docs/design/discovery-v2-brief.md, "Absent weights"): its text side is
    measured and reported, its image side is named as missing, and the
    overall is withheld rather than averaged over a surface nobody measured.
    The branches resolve to a fixpoint, because Lashika's image step is a
    method of the object PREPARE builds and Bagel's prepare takes IMAGE's
    output: no single declared order serves both.
    """

    return Role(
        "search",
        (
            Stage(
                "idfs",
                prefers=("idf", "inverse_document", "document_frequency"),
                fit=True,
                fixture=(list(corpus),),
                produces=looks_like_idf_table,
            ),
        ),
        branches=(
            text_branch(captions),
            replace(image_branch(descriptors), optional=True),
            replace(prepare_branch(image_ids, descriptors), optional=True),
            replace(search_branch(query, k, image_ids), optional=True),
        ),
    )


# --------------------------------------------------------------------------
# Acceptance: the only thing that may accept a binding
# --------------------------------------------------------------------------


def accepts(chains: Any, cases: Sequence[Any], extras: Dict[str, Any]):
    """Run the benchmark's own nine cases on the composed object.

    Passes when every case the bound branches cover comes back ok and the text
    matrix is finite. A repository with no image side is allowed through: its
    retrieval and search cases fail, `plugins.score` withholds the three
    image-side numbers rather than zeroing them, and the run reports the text
    half it actually measured. Refusing here instead would report "your code
    is not wired up" to a team whose caption embedding works.

    No score threshold anywhere. Returns ``(passed, detail)``.
    """

    from .drivers import run_with_adapter
    from .adapters import adapt_search
    from .discovered import DiscoveredSearch

    chains = dict(chains) if isinstance(chains, dict) else {"text": tuple(chains)}
    try:
        composed = DiscoveredSearch(chains, extras)
    except BaseException as error:  # noqa: BLE001 - student code raises anything
        return False, "composing their functions raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )

    try:
        outputs = run_with_adapter(adapt_search(composed), list(cases))
    except BaseException as error:  # noqa: BLE001
        return False, "the driver raised {}: {}".format(
            type(error).__name__, str(error)[:120]
        )

    covered = composed.covered_kinds()
    for case, output in zip(cases, outputs):
        kind = getattr(case, "kind", "?")
        if kind not in covered:
            continue
        if not output.get("ok"):
            return False, "the {} case did not run: {}".format(
                kind, str(output.get("error", ""))[:160]
            )
        if kind == "text":
            matrix = np.asarray(output.get("embeddings", []), dtype=np.float64)
            if matrix.size == 0 or not np.isfinite(matrix).all():
                return False, "the caption embeddings are not finite"
            # Seventy-five different captions that all embed to one vector
            # were not embedded. A single caption of unseen words is allowed
            # to be zeros (see `looks_like_matrix_for`); the whole fixture
            # is not. Measured on CogFinder: `tokenize -> embed_caption`
            # binds too, and `embed_caption` re-tokenizes the token list
            # into one word no vocabulary has, so every row came back zero
            # and the chain passed. Not a score: it reads no ranking.
            if matrix.shape[0] > 1 and np.allclose(matrix, matrix[0]):
                return False, "every caption embedded to the same vector"
    return True, "every case the bound surfaces cover ran"
