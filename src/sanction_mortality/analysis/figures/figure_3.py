"""Create Figure 3: Demographics-Adjusted Mortality Rates.

Output
------
``PATHS.figures / "figure_3.csv"`` (data behind the figure).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.figure import Figure

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import scale_figsize, setup_matplotlib
from sanction_mortality.io import require

setup_matplotlib()

__all__ = ["prepare_data", "make_plot"]

# ── Settings ────────────────────────────────────────────────────────────────

SCALE: int = 1000  # rates per SCALE persons
OUT_NAME: str = "figure_3"

# Numerical controls in model (incl. binary)
NUM_CONTROLS: list[str] = [
    "age",
    "immigrant",
    "descendant",
    "lives_w_parents",
    "married",
    "cohabiting",
    "has_children",
]

# Categorical controls in model
CAT_CONTROLS: list[str] = ["group", "education"]

# ── Functions ───────────────────────────────────────────────────────────────


def prepare_data(save: bool = True) -> pd.DataFrame:
    """Compute unadjusted and demographics-adjusted mortality by sanction group.

    A logit of death during the observation period on sanction group and
    controls is estimated (HC1 standard errors). The adjusted rate for a
    group is the mean predicted probability when every person in the
    estimation sample is assigned to that group.

    Parameters
    ----------
    save : bool
        Write the data to ``PATHS.figures / "figure_3.csv"``.

    Returns
    -------
    DataFrame
        Columns ``unadjusted`` and ``adjusted`` (per :data:`SCALE`), indexed
        by sanction group.
    """
    require("population", directory=paths.temp)

    pop = pd.read_parquet(paths.temp / "population.parquet")

    # Drop any rows with missing values
    need = NUM_CONTROLS + CAT_CONTROLS
    pop[NUM_CONTROLS] = pop[NUM_CONTROLS].astype(float)
    pop = pop.dropna(subset=need)

    # Design matrices
    y = pop["death"].notna().astype(float)
    X = pd.get_dummies(
        pop[NUM_CONTROLS + CAT_CONTROLS],
        columns=CAT_CONTROLS,
        drop_first=True,
        prefix="",
        prefix_sep="",
        dtype=float,
    )
    X = sm.add_constant(X)

    # Estimate model
    model = sm.Logit(y, X).fit(disp=False, cov_type="HC1")

    # Derive adjusted mortality
    out = []
    for name, data in pop.groupby("group", observed=False):
        # Unadjusted: within-group mean
        unadjusted = y.loc[data.index].mean() * SCALE

        # Adjusted: predicted mean if the whole sample had this group's sanction
        X_g = X.copy()
        for group in pop["group"].unique():
            if group in X_g.columns:
                X_g[group] = 1.0 if group == name else 0.0

        adjusted = model.predict(X_g).mean() * SCALE
        out.append((name, unadjusted, adjusted))

    out = pd.DataFrame(out, columns=["group", "unadjusted", "adjusted"])
    out = out.set_index("group")

    if save:
        out.to_csv(paths.figures / f"{OUT_NAME}.csv")

    return out


def make_plot(show: bool = True, save_data: bool = True) -> tuple[Figure, np.ndarray]:
    """Create Figure 3.

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

    df.index = df.index.str.wrap(12)

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2), sharey=True)
    ax = ax.ravel()

    ax[0].bar(df.index, df["unadjusted"])
    ax[1].bar(df.index, df["adjusted"])

    ax[0].set_ylabel("Mortality rate per 1.000")
    ax[0].set_title("Unadjusted")
    ax[1].set_title("Adjusted")

    if not show:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    make_plot(show=False)
