"""Fetch the three public course files this benchmark reads, and nothing else.

    python -m language_search_benchmark.fetch

pip installs the scorer, not the data: the captions, descriptors and GloVe
files are close to a gigabyte together. ``cogworks check`` reads them without
downloading, because a check should not start a gigabyte download as a side
effect, so on a fresh machine this command is the step between installing the
package and checking a repository. It runs no student code and scores nothing.

Every file goes through ``datasets.ensure_artifact``, the same path a local
run takes: a verified copy already in the cache (``COGWORKS_LANGUAGE_DATA`` or
the default) is kept, a byte-identical copy in the course's cogworks-data
cache is adopted, and anything else is downloaded and checked against its
size and sha256 pin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import datasets
from .datasets import DATA_ENV, FETCH_COMMAND, FETCH_TOTAL, DatasetError, megabytes

#: Smallest first, so a broken connection shows up before the 693 MB file.
ORDER = ("captions", "descriptors", "glove")


def _parser() -> argparse.ArgumentParser:
    # Parsed even though it takes no options, so `--help` explains the
    # command instead of starting the download it describes.
    return argparse.ArgumentParser(
        prog=FETCH_COMMAND,
        description=(
            "Fetch the public course files the language-search benchmark reads "
            "(about {}), reusing copies you already have. Set {} to keep them in a "
            "folder of your choice.".format(FETCH_TOTAL, DATA_ENV)
        ),
    )


def _get(name: str, root: Path) -> None:
    """Make one artifact present and verified, saying which way it went."""

    spec = datasets.ARTIFACTS[name]
    filename = str(spec["filename"])
    try:
        datasets.ensure_artifact(name, download=False)
        print("  {}: already here".format(filename))
        return
    except DatasetError:
        pass
    # "Getting" rather than "downloading": a copy in the cogworks-data cache
    # is adopted without touching the network, and which of the two happens
    # is only known afterwards.
    replacing = (root / filename).is_file()
    what = "replacing a copy that fails its checksum, " if replacing else "getting "
    print(
        "  {}: {}{}...".format(filename, what, megabytes(int(spec["size"]))),
        end="",
        flush=True,
    )
    datasets.ensure_artifact(name, download=True)
    print(" ready")


def main(argv: Optional[List[str]] = None) -> int:
    _parser().parse_args(argv)
    root = datasets.cache_root()
    print("Course data folder: {}".format(root))
    try:
        for name in ORDER:
            _get(name, root)
    # OSError as well as DatasetError: checking a file already here reads it
    # and records the result in the folder, which fails on a full disk or a
    # folder this account cannot write.
    except (DatasetError, OSError) as error:
        print()
        print("  {}".format(error), file=sys.stderr)
        print(
            "Files that finished are kept. Check your connection, disk space and "
            "that you can write to the folder above, then run `{}` again.".format(
                FETCH_COMMAND
            ),
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print()
        print(
            "Stopped. Files that finished are kept; run this again to continue.",
            file=sys.stderr,
        )
        return 130
    print("All three files are in place. Next: cogworks check --benchmark language-search")
    return 0


if __name__ == "__main__":
    sys.exit(main())
