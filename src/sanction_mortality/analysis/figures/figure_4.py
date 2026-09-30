"""Create Figure 4: Kaplan-Meier Survival Curves by Sanction.

Output
------
``PATHS.figures / "figure_4.csv"`` (data behind the figure).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import YEARS, setup_matplotlib
from sanction_mortality.io import require

setup_matplotlib()

__all__ = ["prepare_data", "make_plot"]

# ── Settings ────────────────────────────────────────────────────────────────

OUT_NAME: str = "figure_4"

# ── Helpers ─────────────────────────────────────────────────────────────────


def _km_surv(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate a Kaplan-Meier survival curve and apply disclosure control.

    Periods with 1-2 observed deaths are suppressed (set to missing).

    Parameters
    ----------
    df : DataFrame
        Columns ``time`` (years since observation start) and ``died`` (bool).

    Returns
    -------
    DataFrame
        Columns ``survival``, ``lower`` and ``upper`` (95% CI), indexed by time.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df["time"],
        event_observed=df["died"],
        timeline=range(0, YEARS[1] - YEARS[0] + 1),
    )

    # Statistical disclosure control
    count = df.groupby("time")["died"].sum()  # number of deaths observed at time t
    disc = count.between(1, 2)  # periods to suppress

    sf = kmf.survival_function_.mask(disc)
    ci = kmf.confidence_interval_.mask(disc)

    return pd.concat(
        [
            sf.rename(columns={sf.columns[0]: "survival"}),
            ci.rename(columns={ci.columns[0]: "lower", ci.columns[1]: "upper"}),
        ],
        axis=1,
    )


def _plot_km(df: pd.DataFrame, ax: Axes, **kwargs: Any) -> None:
    """Plot a Kaplan-Meier curve with confidence band on *ax*.

    The curve is broken at suppressed periods. All segments share one colour
    and linestyle, and only the first carries the legend label.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`_km_surv` for one group, with index level ``year``.
    ax : Axes
        Axes to draw on.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.step` (e.g. ``label``).
    """
    mask = df["survival"].notna()
    segments = (mask != mask.shift()).cumsum()

    for _, d in df[mask].groupby(segments[mask]):
        x = d.index.get_level_values("year")
        (line,) = ax.step(x, d["survival"], where="post", **kwargs)
        ax.fill_between(
            x, d["lower"], d["upper"], step="post", alpha=0.25, color=line.get_color()
        )
        # Reuse style for subsequent segments; label only the first.
        kwargs = {**kwargs, "color": line.get_color(), "linestyle": line.get_linestyle()}
        kwargs.pop("label", None)


# ── Functions ───────────────────────────────────────────────────────────────


def prepare_data(save: bool = True) -> pd.DataFrame:
    """Compute Kaplan-Meier survival curves by sanction group.

    Parameters
    ----------
    save : bool
        Write the data to ``PATHS.figures / "figure_4.csv"``.

    Returns
    -------
    DataFrame
        Columns ``survival``, ``lower`` and ``upper``, indexed by
        ``(group, year)`` where ``year`` is years since observation start.
    """
    require("population", directory=paths.temp)

    pop = pd.read_parquet(paths.temp / "population.parquet")
    pop = pop[["group", "death"]].copy()

    pop["death"] = pop["death"].dt.year
    pop["died"] = pop["death"].le(YEARS[1])
    pop["time"] = (pop["death"] - YEARS[0]).where(pop["died"], YEARS[1] - YEARS[0])

    out = {}
    for name, data in pop.groupby("group", observed=False):
        out[name] = _km_surv(data)

    out = pd.concat(out, names=["group", "year"])

    if save:
        out.to_csv(paths.figures / f"{OUT_NAME}.csv")

    return out


def make_plot(show: bool = True, save_data: bool = True) -> tuple[Figure, Axes]:
    """Create Figure 4.

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

    for name, data in df.groupby("group"):
        _plot_km(data, ax, label=name)

    ax.set_xlim(0, 10)
    ax.set_ylim(0.86, 1.01)
    ax.tick_params(axis="y", rotation=90)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))

    ax.set_xlabel("Observation year")
    ax.set_ylabel("Survival probability")

    ax.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    if not show:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    make_plot(show=False)
