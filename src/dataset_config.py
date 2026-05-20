"""
Dataset path resolution for training and analysis scripts.

Default training file is the extended realistic vulnerability CSV. Override with:

  set CYBER_DATASET_PATH=C:\\path\\to\\file.csv   (Windows)
  export CYBER_DATASET_PATH=/path/to/file.csv     (Unix)

The legacy scanner-style CSV remains supported and is documented in README.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root (parent of src/)
ROOT = Path(__file__).resolve().parent.parent

# Primary default: extended text + severity dataset
DEFAULT_DATASET_PATH: Path = ROOT / "data" / "extended_realistic_vulnerability_dataset_10000.csv"

# Optional legacy path (tabular scanner simulation); documented for overrides
LEGACY_SCANNER_DATASET_PATH: Path = ROOT / "data" / "security_vulnerabilities.csv"

ENV_DATASET_VAR = "CYBER_DATASET_PATH"


def resolve_dataset_path(explicit: str | Path | None = None) -> Path:
    """
    Return CSV path for training. Order: explicit arg > env CYBER_DATASET_PATH > default extended CSV.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(ENV_DATASET_VAR, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATASET_PATH.resolve()
