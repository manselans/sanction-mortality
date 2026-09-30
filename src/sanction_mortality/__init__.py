"""Replication package for "What Can We Learn About (Getting Around) Data
Limitations in the US from Denmark? A Case Study of Mortality Risk and
Criminal Justice System Involvement".

Subpackages
-----------
etl
    Construction of the study population from Danish register data.
analysis
    Tables and figures reported in the article.

Modules
-------
config
    Project and data paths (requires ``local_paths.py`` at the repo root).
defaults
    Project-wide parameters, variable labels and plotting setup.
io
    Helpers for loading Stata files and lookup tables.

Submodules are not imported eagerly: importing :mod:`sanction_mortality.config`
validates the data paths and creates output directories.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
