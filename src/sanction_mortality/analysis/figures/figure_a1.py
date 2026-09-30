"""Create Figure A1: Checking the Proportional-Hazards Assumption.

Plots scaled Schoenfeld residuals against rank-transformed time for each
covariate of the Cox PH model, using "Arrested Only" as reference group.
Returns one figure per covariate.
"""

from __future__ import annotations

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import SEED, YEARS, setup_matplotlib
from sanction_mortality.io import require

setup_matplotlib()

__all__ = ["compute_residuals", "make_plot"]

# ── Settings ────────────────────────────────────────────────────────────────

OUT_NAME: str = "figure_a1"

# Numerical controls in model (incl. binary)
NUM_CONTROLS: list[str] = [
    "age",
    "age2",  # defined in compute_residuals
    "immigrant",
    "descendant",
    "lives_w_parents",
    "married",
    "cohabiting",
    "has_children",
]

# Categorical controls in model
CAT_CONTROLS: list[str] = ["education"]

REFERENCE: str = "Arrested Only"  # reference group

SUBSAMPLE_FRAC: float = 0.01  # share of the No CJ Contact group retained

# ── Helpers ─────────────────────────────────────────────────────────────────


def _residual_plot(y: pd.Series, ax: Axes) -> None:
    """Scatter residuals with a LOWESS smoother and a zero line on *ax*.

    Parameters
    ----------
    y : Series
        Scaled Schoenfeld residuals for one covariate.
    ax : Axes
        Axes to draw on.
    """
    x = np.arange(len(y))
    smooth = sm.nonparametric.lowess(y, x, frac=0.3)[:, 1]

    ax.axhline(0, color="red", ls="--")
    ax.scatter(x, y, alpha=0.3, s=10, color="0.5")
    ax.plot(x, smooth, linewidth=2, color="0")

    ax.set_ylabel("Scaled Schoenfeld residuals")
    ax.set_xlabel("Rank-transformed time")


# ── Functions ───────────────────────────────────────────────────────────────


def compute_residuals(check_assumptions: bool = True) -> pd.DataFrame:
    """Fit the adjusted Cox PH model and return scaled Schoenfeld residuals.

    For computational reasons, the No CJ Contact group is reduced to a
    random subsample (:data:`SUBSAMPLE_FRAC`, seed :data:`SEED`).

    Parameters
    ----------
    check_assumptions : bool
        Print the output of :meth:`lifelines.CoxPHFitter.check_assumptions`.

    Returns
    -------
    DataFrame
        Scaled Schoenfeld residuals, one column per covariate.
    """
    require("population", directory=paths.temp)

    pop = pd.read_parquet(paths.temp / "population.parquet")

    pop["death"] = pop["death"].dt.year
    pop["died"] = pop["death"].le(YEARS[1])
    pop["time"] = (pop["death"] - YEARS[0]).where(pop["died"], YEARS[1] - YEARS[0])

    pop["age2"] = pop["age"] ** 2

    need = ["group", "died", "time"] + NUM_CONTROLS + CAT_CONTROLS
    pop[NUM_CONTROLS] = pop[NUM_CONTROLS].apply(pd.to_numeric, errors="coerce")
    pop = pop[need].dropna()

    pop = pd.get_dummies(
        pop, columns=CAT_CONTROLS, drop_first=True, prefix="", prefix_sep=""
    )
    pop = pd.get_dummies(
        pop, columns=["group"], drop_first=False, prefix="", prefix_sep=""
    )

    # Reduce the No CJ Contact group to a random subsample for computation
    pop = pd.concat(
        [
            pop[pop["No CJ Contact"]].sample(frac=SUBSAMPLE_FRAC, random_state=SEED),
            pop[~pop["No CJ Contact"]],
        ],
        ignore_index=True,
    )

    # Estimate
    pop = pop.drop(columns=REFERENCE)
    cph = CoxPHFitter()
    res = cph.fit(pop, duration_col="time", event_col="died")

    if check_assumptions:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res.check_assumptions(pop)
        print(buf.getvalue())

    return res.compute_residuals(pop, "scaled_schoenfeld")


def make_plot(
    show: bool = True, check_assumptions: bool = True
) -> dict[str, tuple[Figure, Axes]]:
    """Create Figure A1: one residual plot per covariate.

    Parameters
    ----------
    show : bool
        Keep the figures open for display; if ``False`` they are closed.
    check_assumptions : bool
        Passed to :func:`compute_residuals`.

    Returns
    -------
    dict of str to (Figure, Axes)
        Keyed by covariate name.
    """
    resid = compute_residuals(check_assumptions)

    resid_plots = {}
    for c in resid.columns:
        fig, ax = plt.subplots()
        _residual_plot(resid[c], ax=ax)
        resid_plots[c] = (fig, ax)

    if not show:
        plt.close("all")

    return resid_plots


if __name__ == "__main__":
    make_plot(show=False)
