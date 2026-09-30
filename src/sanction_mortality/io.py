"""I/O helpers for loading data files into :mod:`pandas`.

Provides:

- :func:`load_lookup` – read a ``.csv`` from the ``lookups/`` directory.
- :func:`fetch` – load a single Stata ``.dta`` with column/row filtering.
- :func:`gather` – load multiple ``.dta`` files into a dict or stacked frame.
- :func:`require` – assert that expected files exist.
- :func:`list_columns` – inspect ``.dta`` column names without reading data.
- :func:`resolve` – map short names to file names by prefix or suffix rule.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import pandas as pd

__all__ = ["load_lookup", "fetch", "gather", "require", "list_columns", "resolve"]

# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------
Scalar = Union[str, int, float, bool]
Rule = Union[Scalar, Iterable, Callable]

_LOOKUP_DIR = Path(__file__).resolve().parent / "lookups"


# ---------------------------------------------------------------------------
# Simple utilities
# ---------------------------------------------------------------------------

def list_columns(path: str | Path) -> list[str]:
    """Return the column names of a ``.dta`` file without loading its data.

    Parameters
    ----------
    path : str or Path
        Path to the Stata file.

    Returns
    -------
    list of str
        Variable names in file order.
    """
    with pd.io.stata.StataReader(path) as reader:
        return list(reader.variable_labels())


def resolve(
    want: Iterable[str],
    available: Iterable[str],
    matching: str = "startswith",
) -> dict[str, str]:
    """Map each name in *want* to a unique file in *available*.

    Parameters
    ----------
    want : iterable of str
        Short names to resolve.
    available : iterable of str
        Pool of file names to match against.
    matching : str
        Strategy – ``"startswith"`` or ``"yearsuffix"`` (e.g. ``"income2023"``).

    Returns
    -------
    dict of str to str
        Mapping from resolved file name to requested name.

    Raises
    ------
    ValueError
        For an unknown *matching* strategy.
    KeyError
        If a name has zero or multiple matches.
    """
    strategies: dict[str, Callable[[str, str], bool]] = {
        "startswith": lambda w, a: a.startswith(w),
        "yearsuffix": lambda w, a: bool(re.fullmatch(rf"{re.escape(w)}\d{{4}}", a)),
    }
    if matching not in strategies:
        raise ValueError(f"`matching` must be one of {list(strategies)}; got {matching!r}.")

    pool = list(available)
    out: dict[str, str] = {}
    match = strategies[matching]

    for w in want:
        if w in pool:
            hit = w
        else:
            hits = [a for a in pool if match(w, a)]
            if len(hits) != 1:
                what = "no match" if not hits else f"multiple matches {hits}"
                raise KeyError(f"{what} for {w!r} under {matching!r}.")
            hit = hits[0]

        if hit in out:
            raise ValueError(f"{out[hit]!r} and {w!r} both resolve to {hit!r}.")
        out[hit] = w

    return out


def load_lookup(filename: str, *, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read a ``.csv`` file from the package ``lookups/`` directory.

    Parameters
    ----------
    filename : str
        File name, e.g. ``"countries.csv"``.
    dtype : dict, optional
        Column dtypes forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    DataFrame
        Contents of the lookup file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = _LOOKUP_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Lookup file not found: {path}")
    return pd.read_csv(path, dtype=dtype)


def require(*need: str, directory: str | Path, extension: str = ".parquet") -> None:
    """Assert that ``{name}{extension}`` exists in *directory* for every *need*.

    Parameters
    ----------
    *need : str
        File names without extension.
    directory : str or Path
        Directory to check.
    extension : str
        File extension, including the leading dot.

    Raises
    ------
    FileNotFoundError
        Listing all missing files at once.

    Examples
    --------
    >>> require("train", "test", directory="./data")  # doctest: +SKIP
    >>> require("logs", directory="./out", extension=".json")  # doctest: +SKIP
    """
    d = Path(directory)
    missing = [str(d / (n + extension)) for n in need if not (d / (n + extension)).is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))


# ---------------------------------------------------------------------------
# Stata loaders
# ---------------------------------------------------------------------------

def fetch(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    filters: Mapping[str, Rule] | None = None,
    pairs: Iterable[tuple[Any, ...]] | None = None,
    pair_cols: Sequence[str] | None = None,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Load a Stata ``.dta`` file with optional column selection and row filters.

    Parameters
    ----------
    path : str or Path
        Path to the Stata file.
    columns : sequence of str, optional
        Columns to keep. ``None`` → all.
    filters : mapping, optional
        ``{col: rule}`` – scalar (``==``), iterable (``.isin``), or
        callable (vectorised over the column). Multiple rules are AND-ed.
    pairs : iterable of tuples, optional
        Exact value combinations to retain across *pair_cols*.
    pair_cols : sequence of str, optional
        Column order corresponding to each tuple in *pairs*.
    **read_kwargs
        Forwarded to :func:`pandas.read_stata`.

    Returns
    -------
    DataFrame
        Filtered data with a reset index.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *pairs* is given without *pair_cols*.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Stata file not found: {path}")

    filters = dict(filters or {})

    # Minimal column set to read from disk
    needed: set[str] = set(columns or [])
    needed.update(filters)
    if pairs is not None:
        if not pair_cols:
            raise ValueError("`pair_cols` is required when `pairs` is given.")
        needed.update(pair_cols)

    read_kwargs.setdefault("convert_categoricals", False)
    df = pd.read_stata(path, columns=sorted(needed) or None, **read_kwargs)

    # Build boolean mask
    mask = pd.Series(True, index=df.index)
    for col, rule in filters.items():
        mask &= _to_bool(rule, df[col])
    if pairs is not None:
        mask &= pd.MultiIndex.from_frame(df[list(pair_cols)]).isin(set(pairs))

    df = df.loc[mask]
    if columns is not None:
        df = df[list(columns)]
    return df.reset_index(drop=True)


def gather(
    path: str | Path,
    names: Sequence[str] | None = None,
    *,
    concatenate: bool = False,
    add_name: str | None = None,
    add_values: Sequence[Any] | None = None,
    file_pattern: str = "{name}.dta",
    recursive: bool = False,
    **fetch_kwargs: Any,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Load multiple ``.dta`` files via :func:`fetch`.

    Parameters
    ----------
    path : str or Path
        Directory to search, or a single ``.dta`` file.
    names : sequence of str, optional
        Explicit names (formatted into *file_pattern*). ``None`` → glob ``*.dta``.
    concatenate : bool
        Stack results into one DataFrame instead of returning a dict.
    add_name : str, optional
        Column to stamp with a provenance value.
    add_values : sequence, optional
        Values for *add_name* (defaults to the dataset name).
    file_pattern : str
        Template mapping name → file name (default ``"{name}.dta"``).
    recursive : bool
        Search subdirectories when auto-discovering.
    **fetch_kwargs
        Forwarded to :func:`fetch`.

    Returns
    -------
    dict of str to DataFrame, or DataFrame
        One frame per dataset, or a single stacked frame if *concatenate*.

    Raises
    ------
    FileNotFoundError
        If no files are found or a resolved path is missing.
    ValueError
        If *add_values* length ≠ number of datasets.
    """
    base = Path(path)
    files = _resolve_files(base, names=names, file_pattern=file_pattern, recursive=recursive)

    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))

    if add_name is not None and add_values is not None and len(add_values) != len(files):
        raise ValueError(
            f"`add_values` length ({len(add_values)}) ≠ number of datasets ({len(files)})."
        )

    out: dict[str, pd.DataFrame] = {}
    for i, (name, fpath) in enumerate(files.items()):
        df = fetch(fpath, **fetch_kwargs)
        if add_name is not None:
            df[add_name] = add_values[i] if add_values is not None else name
        out[name] = df

    return pd.concat(out.values(), ignore_index=True) if concatenate else out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_bool(rule: Rule, series: pd.Series) -> pd.Series:
    """Convert a filter *rule* to a boolean Series aligned with *series*.

    * **scalar** → ``series == rule``
    * **iterable** (non-str) → ``series.isin(rule)``
    * **callable** → called with the full Series (vectorised); falls back to
      ``series.apply(rule)`` if the result is not a Series.
    """
    if callable(rule):
        result = rule(series)
        if isinstance(result, pd.Series):
            return result.astype(bool)
        return series.apply(rule).astype(bool)
    if isinstance(rule, Iterable) and not isinstance(rule, (str, bytes)):
        return series.isin(rule)
    return series == rule


def _resolve_files(
    base: Path,
    *,
    names: Sequence[str] | None = None,
    file_pattern: str = "{name}.dta",
    recursive: bool = False,
) -> dict[str, Path]:
    """Build a ``{dataset_name: Path}`` mapping.

    If *names* is given, paths are formatted via *file_pattern*; otherwise
    ``*.dta`` files under *base* are globbed.

    Raises
    ------
    FileNotFoundError
        If auto-discovery finds no ``.dta`` files.
    """
    if names:
        return {n: base / file_pattern.format(name=n) for n in names}
    if base.is_file():
        return {base.stem: base}

    pattern = "**/*.dta" if recursive else "*.dta"
    found = sorted(base.glob(pattern))
    if not found:
        raise FileNotFoundError(f"No .dta files found under {base}")
    return {p.stem: p for p in found}
