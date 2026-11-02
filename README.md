# Resource based football club rankings and tournament uncertainty

Code and aggregate results for "Reliability of resource based football club rankings under tournament uncertainty"
(A. A. Gorenkov, Plekhanov Russian University of Economics). The revised analysis contains 1,080 club-seasons,
57 intact league-season calendars and 19,411 matches in six top divisions, 2013/14–2023/24.

The resource measure uses realised minutes and valuations strictly preceding each appearance. This is a
retrospective exposure measure, not preseason expenditure. A Davidson outcome model supplies joint tournament
simulations; DEA is a descriptive benchmark, not an identified managerial effect. Simulation envelopes are
conditional model references. Next-season validation uses strictly lagged predictors.

## Revised analysis

```
python code/download.py --tables games competitions appearances player_valuations
python code/revision_panel.py
python code/revision_analysis.py
python code/revision_checks.py
```

Alternatively pass `--source /path/to/csv/tables` to `revision_panel.py`. Gzip CSVs are supported.
The checked historical derived `data/panel.csv` is needed for sample-quality diagnostics and comparisons with
the previous resource definition. The main resource and score construction are rerun from the four raw tables.
Raw tables are obtained from the data provider; no manuscript is included.

`results/revision/source_manifest.json` records the inputs used for the reported run. A different provider snapshot
can change the results. `code/revision_checks.py` checks monotonic expected points, fixture consistency, DEA
against linear programming, quality thresholds, and temporal validation. Full estimation and simulation settings
are stored in `results/revision/summary.json`. Vector figures are in `figures/revision/`.

## Previous analysis

Archived code and aggregate results for "Efficiency of football clubs in the Russian Premier League and Europe's
top five leagues: DEA and SFA estimates of turning squad value into points" (A. A. Gorenkov, Plekhanov Russian
University of Economics, 2026).

Output-oriented DEA with the Simar–Wilson bootstrap for each league-season and stochastic frontier models
(Battese–Coelli 1995 with inefficiency determinants, true fixed effects) on 1,248 club-seasons of the English, Spanish,
Italian, German, French and Russian top divisions, 2013/14–2023/24.

## Data
Transfermarkt dataset compiled by dcaribou (`transfermarkt-datasets`, CC0). `code/download.py` fetches the snapshot
into `data/raw/`; `data/panel.csv` is the derived club-season panel.

## Run
```
pip install -r code/requirements.txt
bash code/run_all.sh      # download, panel, validation, DEA, SFA, analysis, figures (~25 min on 4 CPU cores)
```
`code/03_validate.py` checks the DEA and SFA implementations against a textbook example and the R packages
Benchmarking and frontier (reference output in `code/validation/`).

## Licence
Code: MIT. Data: CC0 (Transfermarkt data compiled by dcaribou).
