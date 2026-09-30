"""Project path configuration.

Machine-specific data locations are read from ``local_paths.py`` in the
repository root (see ``local_paths.example.py``). All other paths are derived
from the repository root.

Importing this module validates the configured data paths and creates the
``temp/`` and ``output/`` directories if they do not exist.

Attributes
----------
ROOT : Path
    Repository root.
DATA : DataPaths
    Validated paths to the raw data.
PATHS : Paths
    All project paths; the main object imported by other modules.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ROOT", "DataPaths", "Paths", "DATA", "PATHS"]

ROOT: Path = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from local_paths import DATA_PATHS
except ImportError as e:
    raise ImportError(
        "\n\nlocal_paths.py not found at the repo root.\n"
        "This file holds machine-specific paths and is not tracked in git.\n\n"
        "To fix this, copy\n"
        f"  {ROOT / 'local_paths.example.py'}\n"
        "to\n"
        f"  {ROOT / 'local_paths.py'}\n"
        "and edit the paths inside it for your machine.\n"
    ) from e

OUTPUT: Path = ROOT / "output"

# ---------------------------------------------------------------------------
# Verify path structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS: set[str] = {"dst", "crime"}

_missing = REQUIRED_KEYS - DATA_PATHS.keys()
_extra = DATA_PATHS.keys() - REQUIRED_KEYS

if _missing:
    raise RuntimeError(f"Missing required DATA_PATHS keys: {_missing}")

if _extra:
    raise RuntimeError(f"Unexpected DATA_PATHS keys: {_extra}")


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataPaths:
    """Container for raw-data directories.

    Attributes
    ----------
    dst : Path
        Statistics Denmark raw registers (BEF, UDDA, DOD, ...).
    crime : Path
        Crime registers (placements, convictions).
    """

    dst: Path
    crime: Path


def _resolve_path(p: str | Path) -> Path:
    """Normalise a configured path and check that it exists.

    UNC paths (``\\\\server\\share``) are returned unresolved; if their
    existence cannot be checked due to permissions, they are accepted as is.

    Parameters
    ----------
    p : str or Path
        Path as given in ``local_paths.py``.

    Returns
    -------
    Path
        Expanded (and, for local paths, resolved) path.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(p).expanduser()

    if path.drive.startswith("\\\\"):
        try:
            if not path.exists():
                raise FileNotFoundError(f"Configured path does not exist: {path}")
        except PermissionError:
            pass  # cannot verify existence; move on
        return path

    if not path.exists():
        raise FileNotFoundError(f"Configured path does not exist: {path}")

    return path.resolve()


DATA: DataPaths = DataPaths(
    dst=_resolve_path(DATA_PATHS["dst"]),
    crime=_resolve_path(DATA_PATHS["crime"]),
)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Paths:
    """Container for all project paths.

    Attributes
    ----------
    root : Path
        Repository root.
    data : DataPaths
        Raw-data directories.
    temp : Path
        Intermediate files (e.g. ``population.parquet``).
    output : Path
        Parent directory of all outputs.
    figures : Path
        Figures and the data behind them.
    tables : Path
        Tables.
    """

    root: Path
    data: DataPaths
    temp: Path
    output: Path
    figures: Path
    tables: Path

    def ensure_dirs(self) -> None:
        """Create ``temp``, ``output``, ``figures`` and ``tables`` if missing."""
        for dir_path in (self.temp, self.output, self.figures, self.tables):
            dir_path.mkdir(parents=True, exist_ok=True)


PATHS: Paths = Paths(
    root=ROOT,
    data=DATA,
    temp=ROOT / "temp",
    output=OUTPUT,
    figures=OUTPUT / "figures",
    tables=OUTPUT / "tables",
)

PATHS.ensure_dirs()
