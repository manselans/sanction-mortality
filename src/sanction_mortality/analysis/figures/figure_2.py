"""Create Figure 2: Annual Mortality Rates by Sanction.

Output
------
``PATHS.figures / "figure_2.csv"`` (data behind the figure).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import YEARS, setup_matplotlib
from sanction_mortality.io import require

setup_matplotlib()

__all__ = ["prepare_data", "make_plot"]

# ── Settings ────────────────────────────────────────────────────────────────

SCALE: int = 1000  # rates per SCALE persons
MIN_CELL: int = 4  # cells with fewer deaths or persons at risk are suppressed
OUT_NAME: str = "figure_2"

# ── Functions ───────────────────────────────────────────────────────────────


def prepare_data(save: bool = True) -> pd.DataFrame:
    """Compute annual mortality rates per :data:`SCALE` by sanction group.

    The rate in year *y* is deaths in *y* divided by persons alive at the
    start of *y*. Cells with fewer than :data:`MIN_CELL` deaths or persons
    at risk are suppressed (set to missing).

    Parameters
    ----------
    save : bool
        Write the data to ``PATHS.figures / "figure_2.csv"``.

    Returns
    -------
    DataFrame
        Mortality rates indexed by year, one column per sanction group.
    """
    require("population", directory=paths.temp)

    pop = pd.read_parquet(paths.temp / "population.parquet")
    out = {}

    for y in range(YEARS[0], YEARS[1] + 1):
        alive = pop["death"].isna() | pop["death"].dt.year.ge(y)
        died = pop["death"].dt.year.eq(y)

        alive = alive.groupby(pop["group"], observed=False).sum().rename("alive")
        died = died.groupby(pop["group"], observed=False).sum().rename("died")

        out[y] = pd.concat([alive, died], axis=1)

    out = pd.concat(out, names=["year", "group"])
    out["rate"] = (out["died"] / out["alive"]) * SCALE

    # Statistical disclosure control
    out = out.where(out["died"].ge(MIN_CELL) & out["alive"].ge(MIN_CELL))

    out = out[["rate"]].unstack(level="group").droplevel(0, axis=1)
    if save:
        out.to_csv(paths.figures / f"{OUT_NAME}.csv")
    return out


def make_plot(show: bool = True, save_data: bool = True) -> tuple[Figure, Axes]:
    """Create Figure 2.

    Parameters
    ----------
    show : bool
        Keep the figure open for display; if ``False`` it is closed.
    save_data : bool
        Passed to :func:`prepare_data` as *save*.

    Returns
    -------
    tuple of (Figure, Axes)
    """
    df = prepare_data(save=save_data)

    fig, ax = plt.subplots()

    ax.plot(df.index, df, label=df.columns)

    ax.set_ylabel("Mortality rate per 1.000")
    ax.set_xlabel("Year")
    ax.set_xticks(range(YEARS[0], YEARS[1] + 1))
    ax.legend(bbox_to_anchor=(0.5, -0.12), ncol=3)

    if not show:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    make_plot(show=False)
