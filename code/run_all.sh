#!/usr/bin/env bash
# Full pipeline, one command: bash code/run_all.sh  (about 25 min on 4 CPU cores).
# R reference files in code/validation/ are committed; regenerate them with: Rscript code/validation/r_reference.R
set -euo pipefail
cd "$(dirname "$0")"
python3 download.py
python3 01_inspect.py > /dev/null
python3 02_panel.py
python3 03_validate.py > /dev/null
python3 04_dea.py
python3 05_sfa.py > ../results/sfa_log.txt
python3 06_analysis.py > ../results/analysis_log.txt
python3 07_figures.py
