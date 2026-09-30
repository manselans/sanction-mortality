"""Create Figure 5: Cox Proportional-Hazards Estimates.

Output
------
``PATHS.figures / "figure_5.csv"`` (data behind the figure).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from matplotlib.figure import Figure

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import YEARS, scale_figsize, setup_matplotlib
from sanction_mortality.io import require

setup_matplotlib()

__all__ = ["prepare_data", "make_plot"]

# ── Settings ────────────────────────────────────────────────────────────────

OUT_NAME: str = "figure_5"

REFERENCES: list[str] = ["No CJ Contact", "Arrested Only"]  # one panel each

# Numerical controls in model (incl. binary)
NUM_CONTROLS: list[str] = [
    "age",
    "age2",  # defined in prepare_data
    "immigrant",
    "descendant",
    "lives_w_parents",
    "married",
    "cohabiting",
    "has_children",
]

# Categorical controls in model
CAT_CONTROLS: list[str] = ["education"]

# ── Helpers ─────────────────────────────────────────────────────────────────


def _cph_est(df: pd.DataFrame) -> CoxPHFitter:
    """Fit a Cox PH model with duration ``time`` and event ``died``.

    Parameters
    ----------
    df : DataFrame
        ``time``, ``died`` and all covariates to include.

    Returns
    -------
    CoxPHFitter
        Fitted model.
    """
    cph = CoxPHFitter()
    return cph.fit(df, duration_col="time", event_col="died")


def _cph_res(results: CoxPHFitter, variables: Sequence[str] | None = None) -> pd.DataFrame:
    """Extract hazard ratios, 95% CIs and p-values from a fitted Cox PH model.

    Parameters
    ----------
    results : CoxPHFitter
        Fitted model.
    variables : sequence of str, optional
        Covariates to keep. ``None`` → all.

    Returns
    -------
    DataFrame
        Columns ``HR``, ``lo``, ``hi`` and ``p``, indexed by covariate.
    """
    variables = variables if variables is not None else results.summary.index
    out = results.summary.loc[
        variables, ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ]
    return out.rename(
        columns={
            "exp(coef)": "HR",
            "exp(coef) lower 95%": "lo",
            "exp(coef) upper 95%": "hi",
        }
    )


# ── Functions ───────────────────────────────────────────────────────────────


def prepare_data(save: bool = True) -> pd.DataFrame:
    """Estimate sanction-group hazard ratios relative to each reference group.

    For each reference group in :data:`REFERENCES`, an unadjusted model
    (sanction dummies only) and an adjusted model (sanction dummies and
    controls) are estimated.

    Parameters
    ----------
    save : bool
        Write the data to ``PATHS.figures / "figure_5.csv"``.

    Returns
    -------
    DataFrame
        Columns ``HR``, ``lo``, ``hi`` and ``p``, indexed by
        ``(reference, type, coef)``.
    """
    require("population", directory=paths.temp)

    pop = pd.read_parquet(paths.temp / "population.parquet")
    groups = pop["group"].cat.categories

    pop["death"] = pop["death"].dt.year
    pop["died"] = pop["death"].le(YEARS[1])
    pop["time"] = (pop["death"] - YEARS[0]).where(pop["died"], YEARS[1] - YEARS[0])

    pop["age2"] = pop["age"] ** 2

    # Drop any rows with missing values
    need = ["group", "died", "time"] + NUM_CONTROLS + CAT_CONTROLS
    pop[NUM_CONTROLS] = pop[NUM_CONTROLS].apply(pd.to_numeric, errors="coerce")
    pop = pop[need].dropna()

    pop = pd.get_dummies(
        pop, columns=CAT_CONTROLS, drop_first=True, prefix="", prefix_sep=""
    )
    pop = pd.get_dummies(
        pop, columns=["group"], drop_first=False, prefix="", prefix_sep=""
    )

    # Estimate
    out = {}
    for ref in REFERENCES:
        for adjust in (False, True):
            gd = [g for g in groups if g != ref]
            include = (
                [c for c in pop.columns if c != ref] if adjust else ["time", "died"] + gd
            )
            aname = "Adjusted" if adjust else "Unadjusted"
            out[(ref, aname)] = _cph_res(_cph_est(pop[include]), variables=gd).reindex(
                groups
            )

    out = pd.concat(out, names=["reference", "type", "coef"])

    if save:
        out.to_csv(paths.figures / f"{OUT_NAME}.csv")

    return out


def make_plot(show: bool = True, save_data: bool = True) -> tuple[Figure, np.ndarray]:
    """Create Figure 5.

    Parameters
    ----------
    show : bool
        Keep the figure open for display; if ``False`` it is closed.
    save_data : bool
        Passed to :func:`prepare_data` as *save*.

    Returns
    -------
    tuple of (Figure, ndarray of Axes)
    """
    df = prepare_data(save=save_data)

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2), sharey=True)
    ax = ax.ravel()

    plotopts = {
        "capsize": 4,
        "linestyle": "none",
        "color": "black",
        "ecolor": "0.6",
    }

    for i, ref in enumerate(REFERENCES):
        ax[i].axvline(1, linestyle="--", color="red")
        for c, d in df.loc[ref].groupby("type"):
            y = d.index.get_level_values("coef")
            xerr = [d["HR"] - d["lo"], d["hi"] - d["HR"]]
            ax[i].errorbar(
                d["HR"],
                y,
                xerr=xerr,
                **plotopts,
                fmt="o" if c == "Adjusted" else "^",
                label=c if i else None,
            )
        ax[i].set_xlabel("Hazard ratio (HR)")
        ax[i].set_xlim(0, 5)
    fig.legend(bbox_to_anchor=(0.5, -0.1), ncol=2)

    if not show:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    make_plot(show=False)
