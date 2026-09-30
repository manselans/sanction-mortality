"""Replicate all tables and figures.

Run from the repository root: ``python replicate.py``
"""

from sanction_mortality.analysis.figures import (
    figure_2,
    figure_3,
    figure_4,
    figure_5,
    figure_a1,
)
from sanction_mortality.analysis.tables import table_2
from sanction_mortality.config import PATHS
from sanction_mortality.etl import gather_population


def main() -> None:
    """Build the study population, then create all tables and figures."""
    gather_population.main()

    table_2.make_table(show=False)

    for module in (figure_2, figure_3, figure_4, figure_5):
        fig, _ = module.make_plot(show=False)
        fig.savefig(PATHS.figures / f"{module.OUT_NAME}.png")

    for name, (fig, _) in figure_a1.make_plot(show=False).items():
        fig.savefig(PATHS.figures / f"{figure_a1.OUT_NAME}_{name}.png")


if __name__ == "__main__":
    main()
