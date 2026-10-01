"""`python -m language_search_benchmark.fetch` against small local files.

The artifact table is pointed at `file://` copies of three tiny fixture files,
so every fetch goes through the real `_download`, size gate and sha256 pin
with no network. The course's cogworks-data cache is redirected to a temporary
folder, because the real one on a developer's machine may hold the real files.
"""

import hashlib
import os
import subprocess
import sys

import pytest

from language_search_benchmark import datasets, fetch

CONTENT = {
    "captions": b'{"images": [], "annotations": []}',
    "descriptors": b"descriptor bytes",
    "glove": b"the 0.1 0.2\n",
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Upstream files, an empty data folder, and an empty course cache."""

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    table = {}
    for name, body in CONTENT.items():
        filename = str(datasets.ARTIFACTS[name]["filename"])
        (upstream / filename).write_bytes(body)
        table[name] = {
            "filename": filename,
            "urls": [(upstream / filename).as_uri()],
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
    monkeypatch.setattr(datasets, "ARTIFACTS", table)

    data = tmp_path / "data"
    monkeypatch.setenv(datasets.DATA_ENV, str(data))

    course = tmp_path / "cog_data"
    import platformdirs

    real = platformdirs.user_cache_path
    monkeypatch.setattr(
        platformdirs,
        "user_cache_path",
        lambda app, *args, **kwargs: (
            course if app == datasets.COURSE_CACHE_APP else real(app, *args, **kwargs)
        ),
    )
    return {"upstream": upstream, "data": data, "course": course, "table": table}


def _cut_upstream(world):
    """Any download attempted after this fails, so reuse is observable."""

    for spec in world["table"].values():
        spec["urls"] = [(world["upstream"] / "gone" / str(spec["filename"])).as_uri()]


def test_cold_cache_fetches_everything_check_needs(world, capsys):
    with pytest.raises(datasets.DatasetError, match="language_search_benchmark.fetch"):
        datasets.build_resources(download=False, build_kv=False)

    assert fetch.main([]) == 0

    out = capsys.readouterr().out
    assert str(world["data"]) in out
    for name, body in CONTENT.items():
        filename = str(datasets.ARTIFACTS[name]["filename"])
        assert "{}: getting 0 MB... ready".format(filename) in out
        assert (world["data"] / filename).read_bytes() == body
    assert "cogworks check --benchmark language-search" in out
    # The call `cogworks check` makes, which refused before the fetch.
    resources = datasets.build_resources(download=False, build_kv=False)
    assert resources.captions_path == world["data"] / "captions_train2014.json"
    assert not list(world["data"].glob("tmp*"))


def test_second_run_reuses_the_cache_without_downloading(world, capsys):
    assert fetch.main([]) == 0
    capsys.readouterr()
    _cut_upstream(world)

    assert fetch.main([]) == 0

    out = capsys.readouterr().out
    assert out.count("already here") == 3
    assert "getting" not in out


def test_course_cache_copies_are_adopted_instead_of_downloaded(world, capsys):
    world["course"].mkdir()
    for name, body in CONTENT.items():
        (world["course"] / str(datasets.ARTIFACTS[name]["filename"])).write_bytes(body)
    _cut_upstream(world)

    assert fetch.main([]) == 0

    for name, body in CONTENT.items():
        assert (world["data"] / str(datasets.ARTIFACTS[name]["filename"])).read_bytes() == body


def test_a_folder_that_already_has_the_files_is_used_in_place(world, capsys):
    world["data"].mkdir()
    for name, body in CONTENT.items():
        (world["data"] / str(datasets.ARTIFACTS[name]["filename"])).write_bytes(body)
    _cut_upstream(world)

    assert fetch.main([]) == 0
    assert capsys.readouterr().out.count("already here") == 3


def test_a_corrupt_copy_is_named_and_replaced(world, capsys):
    world["data"].mkdir()
    captions = world["data"] / "captions_train2014.json"
    # Right size, wrong bytes: only the sha256 pin can tell.
    captions.write_bytes(b"x" * len(CONTENT["captions"]))
    with pytest.raises(datasets.DatasetError, match="does not match its checksum"):
        datasets.ensure_artifact("captions", download=False)

    assert fetch.main([]) == 0

    assert "captions_train2014.json: replacing a copy that fails its checksum" in (
        capsys.readouterr().out
    )
    assert captions.read_bytes() == CONTENT["captions"]


def test_a_changed_upstream_file_is_refused_and_finished_files_kept(world, capsys):
    (world["upstream"] / "resnet18_features.pkl").write_bytes(b"tampered upstream")

    assert fetch.main([]) == 1

    err = capsys.readouterr().err
    assert "resnet18_features.pkl" in err
    assert "fails its sha256 pin" in err or "bytes; expected" in err
    assert "run `python -m language_search_benchmark.fetch` again" in err
    assert (world["data"] / "captions_train2014.json").is_file()
    assert not (world["data"] / "resnet18_features.pkl").exists()
    assert not list(world["data"].glob("tmp*"))


def test_an_unreachable_source_fails_with_one_next_step(world, capsys):
    _cut_upstream(world)

    assert fetch.main([]) == 1

    err = capsys.readouterr().err
    assert "Could not fetch captions_train2014.json" in err
    assert "Traceback" not in err


def test_an_interrupted_download_leaves_no_partial_file(world, monkeypatch, capsys):
    def interrupted(source, target, length=0):
        target.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(datasets.shutil, "copyfileobj", interrupted)

    assert fetch.main([]) == 130

    assert "Stopped" in capsys.readouterr().err
    assert list(world["data"].iterdir()) == []


def test_help_explains_without_downloading(tmp_path):
    data = tmp_path / "data"
    result = subprocess.run(
        [sys.executable, "-m", "language_search_benchmark.fetch", "--help"],
        env={**os.environ, datasets.DATA_ENV: str(data)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "reusing copies you already have" in result.stdout
    assert not data.exists()
