"""Gather the study population.

Scope
-----
All adult males (age >= 18) registered in the Danish Basic Register of
Population (BEF) at year-end 2010, enriched with sanction, education,
covariate, and death-censoring information.

Output
------
``PATHS.temp / "population.parquet"``
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from sanction_mortality.config import PATHS as paths
from sanction_mortality.defaults import YEARS
from sanction_mortality.io import fetch, load_lookup

__all__ = ["main", "OUTPUT", "SANCTIONS"]

# ── Settings ────────────────────────────────────────────────────────────────

MIN_AGE: int = 18  # minimum age at year-end
SEX: int = 1  # BEF coding: 1 = male, 2 = female

OUTPUT: Path = paths.temp / "population.parquet"

# ── Population records (BEF) ────────────────────────────────────────────────

BEF_COLS: list[str] = [
    "pnr",  # person ID
    "alder",  # age in years
    "antboernf",  # number of own children in household
    "e_faelle_id",  # cohabitant ID
    "aegte_id",  # spouse ID
    "ie_type",  # immigration status (1 = native, 2 = immigrant, 3 = descendant)
    "plads",  # role within family
]

RENAMING: dict[str, str] = {
    "alder": "age",
    "antboernf": "children",
    "e_faelle_id": "cohabitant",
    "aegte_id": "spouse",
    "plads": "role",
}

# ── Education records ───────────────────────────────────────────────────────

EDU_REGISTER: Path = paths.data.dst / f"udda{YEARS[0] - 1}.dta"

EDU_LEVELS: dict[str, str | float] = {
    "Early childhood education": "No HS diploma",
    "Primary": "No HS diploma",
    "Lower secondary": "No HS diploma",
    "Upper secondary": "High school",
    "Short cycle tertiary": "Post-high school, non-college",
    "Bachelor or equivalent": "College degree",
    "Master or equivalent": "College degree",
    "Doctoral or equivalent": "College degree",
    "Not elsewhere classified": pd.NA,
}

# ── Conviction records: sanction type codes ─────────────────────────────────

SANCTIONS: dict[str, str] = {  # ordered by severity (increasing)
    "nothing": "No CJ Contact",
    "arrest": "Arrested Only",
    "suspended": "Probation",
    "community": "Community Service",
    "e_monitor": "Electronic Monitoring",
    "prison": "Imprisonment",
}

_AFGTYP_SUSPENDED: list[int] = [117, 121, 122, 123, 124]
_AFGTYP_COMMUNITY: list[int] = [117, 122, 124]

# ── Pipeline steps ──────────────────────────────────────────────────────────


def _sanctions(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each person by the most severe sanction received in year 1.

    Severity order (descending): prison > e-monitoring > community service
    > probation > arrest > none.

    Parameters
    ----------
    df : DataFrame
        Population with column ``pnr``.

    Returns
    -------
    DataFrame
        Copy of *df* with the ordered categorical column ``group``.
    """
    df = df.copy()
    ids = set(df["pnr"])
    sanction: dict[str, set] = {}

    # ── Incarcerations / placements ────────────────────────────────────
    inc = fetch(
        paths.data.crime / "krin_placering.dta",
        columns=["pnr", "handelse", "stedkode"],
        filters={
            "pnr": ids,
            "fgsldto": lambda x: x.dt.year.eq(YEARS[0]),
            "handelse": [1, 2, 5],  # 1/2 = arrest, 5 = serving time
        },
    )

    sanction["prison"] = set(
        inc.loc[inc["handelse"].eq(5) & inc["stedkode"].isin([2, 3]), "pnr"]
    )
    sanction["e_monitor"] = set(
        inc.loc[inc["handelse"].eq(5) & inc["stedkode"].isin([6]), "pnr"]
    )
    sanction["arrest"] = set(inc.loc[inc["handelse"].isin([1, 2]), "pnr"])

    # ── Convictions ─────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UnicodeWarning)
        convictions = fetch(
            paths.data.crime / "incidentdata.dta",
            columns=["pnr", "safgdto", "afgtyp3"],
            filters={
                "safgdto": lambda x: x.dt.year.eq(YEARS[0]),
                "penallaw": 1,
                "afgtyp3": _AFGTYP_SUSPENDED,
            },
        )

    sanction["suspended"] = set(convictions["pnr"])
    sanction["community"] = set(
        convictions.loc[convictions["afgtyp3"].isin(_AFGTYP_COMMUNITY), "pnr"]
    )

    # Guard: every populated sanction type should have a label in SANCTIONS.
    unlabelled = set(sanction) - set(SANCTIONS)
    if unlabelled:
        warnings.warn(f"Sanction type without a stated name: {unlabelled}", stacklevel=2)

    # ── Assign the most severe sanction per person ─────────────────────
    df["group"] = pd.Categorical(
        values=pd.Series(pd.NA, index=df.index),
        categories=list(SANCTIONS.values()),
        ordered=True,
    )

    for key, label in SANCTIONS.items():
        # Iterating in increasing severity: harder sanctions overwrite milder ones.
        if key not in sanction:
            continue
        df.loc[df["pnr"].isin(sanction[key]), "group"] = label

    df["group"] = df["group"].fillna(SANCTIONS["nothing"])
    return df


def _define_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary covariates from raw BEF fields and drop the source columns.

    Parameters
    ----------
    df : DataFrame
        Population with columns ``children``, ``cohabitant``, ``spouse``,
        ``role`` and ``ie_type``.

    Returns
    -------
    DataFrame
        Copy of *df* with ``married``, ``cohabiting``, ``lives_w_parents``,
        ``has_children``, ``immigrant`` and ``descendant`` added, and
        ``spouse``, ``cohabitant``, ``role`` and ``ie_type`` dropped.

    Raises
    ------
    KeyError
        If a required column is missing.
    """
    df = df.copy()

    required = {"spouse", "cohabitant", "role", "children", "ie_type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Required columns missing from `df`: {sorted(missing)}")

    # Relationship / household structure
    df["married"] = df["spouse"].notna()
    df["cohabiting"] = ~df["married"] & df["cohabitant"].notna()
    df["lives_w_parents"] = df["role"].eq(3)

    df["children"] = df["children"].mask(df["lives_w_parents"], 0)
    df["has_children"] = df["children"].gt(0)

    # Origin
    df["immigrant"] = df["ie_type"].eq(2)
    df["descendant"] = df["ie_type"].eq(3)

    return df.drop(columns=["spouse", "cohabitant", "role", "ie_type"])


def _add_education(df: pd.DataFrame) -> pd.DataFrame:
    """Attach highest education at year-end 2010 as column ``education``.

    HFAUDD codes are mapped to ISCED level names via
    ``lookups/audd_levels.csv`` and then to the study categories in
    :data:`EDU_LEVELS`.

    Parameters
    ----------
    df : DataFrame
        Population with column ``pnr``.

    Returns
    -------
    DataFrame
        Copy of *df* with column ``education``.
    """
    df = df.copy()

    ids = set(df["pnr"])
    education = fetch(EDU_REGISTER, columns=["pnr", "hfaudd"], filters={"pnr": ids})

    # HFAUDD code → level name, then reclassify into study categories.
    levels = load_lookup("audd_levels.csv")

    missing = set(education["hfaudd"].dropna()) - set(levels["audd"])
    if missing:
        warnings.warn(f"HFAUDD codes not in lookup table: {missing}", stacklevel=2)

    education["level"] = education["hfaudd"].map(levels.set_index("audd")["level_name"])

    missing = set(levels["level_name"]) - set(EDU_LEVELS)
    if missing:
        warnings.warn(f"Education level names not in EDU_LEVELS: {missing}", stacklevel=2)

    education["level"] = education["level"].map(EDU_LEVELS)

    df["education"] = df["pnr"].map(education.set_index("pnr")["level"])
    return df


def _deaths(df: pd.DataFrame) -> pd.DataFrame:
    """Attach death dates and drop records inconsistent with the study scope.

    A person registered at year-end 2010 cannot have died before the start
    of observation (``YEARS[0]``); such records are treated as data errors
    and removed.

    Parameters
    ----------
    df : DataFrame
        Population with column ``pnr``.

    Returns
    -------
    DataFrame
        Copy of *df* with column ``death`` (date of death, ``NaT`` if alive).
    """
    df = df.copy()

    ids = set(df["pnr"])
    death = fetch(
        paths.data.dst / f"dod{YEARS[1]}.dta",
        columns=["pnr", "doddato"],
        filters={"pnr": ids},
    ).rename(columns={"doddato": "death"})

    df["death"] = df["pnr"].map(death.set_index("pnr")["death"])

    # Retain survivors and those who died since observation start.
    df = df.loc[df["death"].isna() | df["death"].dt.year.ge(YEARS[0])]
    return df


def _remove_migrated(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude persons who left Denmark without a recorded death.

    A person is retained if they have a recorded death or are registered in
    BEF at the end of observation (``YEARS[1]``).

    Parameters
    ----------
    df : DataFrame
        Population with columns ``pnr`` and ``death``.

    Returns
    -------
    DataFrame
        Filtered copy of *df*.
    """
    df = df.copy()
    path = paths.data.dst / f"bef12_{YEARS[1]}.dta"
    at_end = set(fetch(path, columns=["pnr"])["pnr"])

    return df.loc[df["death"].notna() | df["pnr"].isin(at_end)]


# ── Entry point ─────────────────────────────────────────────────────────────


def main(overwrite: bool = False) -> None:
    """Build the analysis population and write it to :data:`OUTPUT`.

    Parameters
    ----------
    overwrite : bool
        Recompute even if :data:`OUTPUT` already exists.
    """
    if OUTPUT.exists() and not overwrite:
        print(f"{OUTPUT} already exists; skipping.")
        return

    # 1. Load population (BEF year-end 2010)
    path = paths.data.dst / f"bef12_{YEARS[0] - 1}.dta"
    pop = fetch(path, columns=BEF_COLS, filters={"koen": SEX}).rename(columns=RENAMING)

    # Coerce to numeric
    pop = pop.apply(pd.to_numeric, errors="coerce")
    pop = pop.loc[pop["age"].ge(MIN_AGE)]

    # 2. Sanctions
    pop = _sanctions(pop)

    # 3. Covariates & education
    pop = _define_covariates(pop)
    pop = _add_education(pop)

    # 4. Deaths
    pop = _deaths(pop)

    # 5. Migrations
    pop = _remove_migrated(pop)

    # 6. Save
    pop.to_parquet(OUTPUT)


if __name__ == "__main__":
    main()
