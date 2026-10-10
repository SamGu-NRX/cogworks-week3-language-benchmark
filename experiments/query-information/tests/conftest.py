import os
import sys

# The study imports the frozen grammar modules by file location and the
# benchmark package from the repo root; pytest does not know either path.
EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
for path in (REPO_ROOT, EXPERIMENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
