"""Project-wide defaults: parameters, variable labels and plotting setup."""

from __future__ import annotations

__all__ = [
    "YEARS",
    "SEED",
    "VAR_NAMES",
    "DEFAULT_FIGSIZE",
    "GRAYS",
    "LINESTYLES",
    "HATCHES",
    "COLORS",
    "setup_matplotlib",
    "scale_figsize",
]

# ---------------------------------------------------------------------------
# Parameter values
# ---------------------------------------------------------------------------

YEARS: tuple[int, int] = (2011, 2021)  # first and last year of observation
SEED: int = 42  # random seed for subsampling

# ---------------------------------------------------------------------------
# Variable labels
# ---------------------------------------------------------------------------

VAR_NAMES: dict[str, str] = {
    "age": "Age",
    "married": "Married",
    "cohabiting": "Cohabiting",
    "children": "Children",
    "has_children": "Has children",
    "lives_w_parents": "Living home <25",
    "immigrant": "Immigrant",
    "descendant": "Descendant",
    "college": "College degree",
}

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

DEFAULT_FIGSIZE: tuple[float, float] = (8, 6)
GRAYS: list[str] = ["0.4", "0.6", "0.0", "0.8", "0.5", "0.7"]
LINESTYLES: list[str] = ["-", "--", ":", "-.", "-", "--"]
HATCHES: list[str] = ["//", "..", "xx", "++", "oo", "\\\\"]
COLORS: list[str] = ["0.2", "0.5", "firebrick", "indianred", "0.8", "#e29a9a"]


def setup_matplotlib() -> None:
    """Apply the project's matplotlib configuration via ``rcParams``."""
    import matplotlib as mpl  # pylint: disable=import-outside-toplevel
    from cycler import cycler  # pylint: disable=import-outside-toplevel

    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.constrained_layout.use": True,
            "font.family": "serif",
            "font.serif": ("Times New Roman",),
            "font.size": 12,
            "legend.frameon": False,
            "legend.loc": "upper center",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlepad": 10,
            "savefig.format": "png",
            "axes.prop_cycle": (cycler(color=GRAYS) + cycler(linestyle=LINESTYLES)),
        }
    )


def scale_figsize(
    base: tuple[float, float] = DEFAULT_FIGSIZE,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[float, float]:
    """Scale a base figure size to a panel of ``nrows`` x ``ncols`` subplots.

    Parameters
    ----------
    base : tuple of float
        Width and height of a single panel, in inches.
    nrows, ncols : int
        Number of panel rows and columns.

    Returns
    -------
    tuple of float
        ``(width * ncols, height * nrows)``.
    """
    w, h = base
    return (w * ncols, h * nrows)
