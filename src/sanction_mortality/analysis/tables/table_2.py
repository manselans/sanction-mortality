"""Create Table 2: Demographics by Sanction.

Output
------
``PATHS.tables / "table_2.csv"``
"""

from __future__ import annotations

import pandas as pd

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import VAR_NAMES
from sanction_mortality.io import require

__all__ = ["make_table"]

# ── Settings ────────────────────────────────────────────────────────────────

OUT_NAME: str = "table_2"

ROWS: list[str] = [  # variables to include, in order of appearance
    "age",
    "married",
    "cohabiting",
    "has_children",
    "lives_w_parents",
    "immigrant",
    "descendant",
    "college",
]

STATS: dict[str, str] = {"mean": "Mean", "std": "SD"}

# ── Helpers ─────────────────────────────────────────────────────────────────


def _display_format(x: object) -> object:
    """Format a table cell for display: blank if missing, thousands separator if > 1000."""
    if pd.isna(x):
        return ""
    return f"{x:,.0f}" if x > 1000 else x


# ── Functions ───────────────────────────────────────────────────────────────


def make_table(save: bool = True, show: bool = True) -> pd.DataFrame | None:
    """Create Table 2: Demographics by Sanction.

    Parameters
    ----------
    save : bool
        Write the table to ``PATHS.tables / "table_2.csv"``.
    show : bool
        Return a display-formatted copy of the table.

    Returns
    -------
    DataFrame or None
        Formatted table if *show*, else ``None``.
    """
    require("population", directory=paths.temp)

    pop = pd.read_parquet(paths.temp / "population.parquet")
    pop["college"] = pop["education"].eq("College degree")

    # Summary statistics by sanction, side by side
    tbl = pd.concat(
        {
            name: data[ROWS].agg(list(STATS)).T
            for name, data in pop.groupby("group", observed=False)
        },
        axis=1,
    )

    # Add N under each sanction type's mean
    counts = pop.groupby("group", observed=False).size()
    vals = [int(counts.loc[g]) if stat == "mean" else pd.NA for (g, stat) in tbl.columns]
    nrow = pd.DataFrame([vals], columns=tbl.columns, index=["N"])

    tbl = pd.concat([tbl, nrow])

    # Labels and rounding
    tbl = tbl.rename(index=VAR_NAMES)
    tbl = tbl.rename(columns=STATS)
    tbl = tbl.round(2)

    if save:
        tbl.to_csv(paths.tables / f"{OUT_NAME}.csv")

    if show:
        return tbl.map(_display_format)
    return None


if __name__ == "__main__":
    make_table(show=False)
