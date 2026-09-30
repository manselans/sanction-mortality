# Replication package: Sanctions and Mortality

Code for *What Can We Learn About (Getting Around) Data Limitations in the US from Denmark? A Case Study of Mortality Risk and Criminal Justice System Involvement*.

The analysis uses Danish administrative register data, available only on Statistics Denmark's research servers. The data are not included.

## Setup

1. Install the package (from the repository root):

   ```
   pip install -e .
   ```

2. Copy `local_paths.example.py` to `local_paths.py` and set the paths to the raw data on your server.

## Run

```
python replicate.py
```

## Structure

```
replicate.py                    run the full pipeline
local_paths.example.py          template for machine-specific data paths
src/sanction_mortality/
    config.py                   project and data paths
    defaults.py                 parameters, labels, plotting setup
    io.py                       data-loading helpers
    lookups/audd_levels.csv     HFAUDD → education level
    etl/gather_population.py    build the study population
    analysis/tables/table_2.py
    analysis/figures/figure_{2,3,4,5,a1}.py
```

## Outputs

| Output | File(s) |
|---|---|
| Population | `temp/population.parquet` |
| Table 2 | `output/tables/table_2.csv` |
| Figures 2–5 | `output/figures/figure_*.png` and `.csv` (data behind each figure) |
| Figure A1 | `output/figures/figure_a1_<covariate>.png` |

Table 1 and Figure 1 are not produced by this code. They are based on other published studies, not on register data.
