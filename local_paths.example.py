"""Example machine-specific path configuration.

Copy this file to ``local_paths.py`` in the repository root and edit the
values below to match your local environment. ``local_paths.py`` is read by
:mod:`sanction_mortality.config` and should not be tracked in version control.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Required: paths to the data on your server
# ---------------------------------------------------------------------------

DATA_PATHS: dict[str, Path] = {
    "dst": Path(r"path\to\DST\raw\data"),  # Statistics Denmark raw registers
    "crime": Path(r"path\to\crime\registers"),  # crime registers
}
