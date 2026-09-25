# Football club efficiency: DEA and SFA

Code and aggregate results for the article "Efficiency of football clubs in the Russian Premier League and Europe's
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
