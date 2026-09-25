"""DEA efficiency of clubs, one frontier per league-season, output orientation.

Sample: seasons 2013-2025 of GB1, ES1, IT1, L1, FR1, RU1 with complete squad data (data/panel.csv, complete == 1).
The main window is 2013-2023; 2024-2025 are kept for robustness.
Output: points per game (within a league-season this is proportional to points).
Models:
  M1  input: squad value                                  -> VRS and CRS; Simar-Wilson (1998) bootstrap, B = 2000
  M2  inputs: squad value, number of players used          -> VRS; bootstrap B = 1000
  M3  inputs: squad value, transfer spending (3-season mean) -> VRS, seasons >= 2017 only (fee coverage, see STATUS)
  R1  input: minutes-weighted value of the fielded XI      -> VRS (robustness)
  R2  input: squad value; outputs: ppg, goal share gf/(gf+ga) -> VRS (robustness)
Efficiency is reported as theta = 1 / delta in (0, 1]: the share of the frontier points per game attainable with the
club's inputs. Writes results/dea_scores.csv.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import dea

ROOT = Path(__file__).resolve().parents[1]
LEAGUE_ORDER = ["GB1", "ES1", "IT1", "L1", "FR1", "RU1"]


def one_league_season(args):
    key, d = args
    league, season = key
    seed = 10000 * season + LEAGUE_ORDER.index(league)
    out = d[["league", "season", "club_id", "club"]].copy()
    y = d[["ppg"]].values
    x1 = d[["squad_value"]].values / 1e6
    b1 = dea.sw98_bootstrap(x1, y, "vrs", B=2000, rng=seed)
    out["m1_vrs"] = 1 / b1["delta_hat"]
    out["m1_vrs_bc"] = 1 / b1["delta_bc"]
    out["m1_vrs_lo"] = 1 / b1["ci_high"]
    out["m1_vrs_hi"] = 1 / b1["ci_low"]
    out["m1_vrs_bias"] = b1["bias"]
    out["m1_vrs_bootsd"] = b1["boot_sd"]
    out["m1_h"] = b1["h"]
    out["m1_crs"] = 1 / dea.farrell_output_1d(x1[:, 0], y[:, 0], "crs")
    x2 = np.c_[x1, d.players_used.values]
    b2 = dea.sw98_bootstrap(x2, y, "vrs", B=1000, rng=seed + 1)
    out["m2_vrs"] = 1 / b2["delta_hat"]
    out["m2_vrs_bc"] = 1 / b2["delta_bc"]
    out["m2_vrs_lo"] = 1 / b2["ci_high"]
    out["m2_vrs_hi"] = 1 / b2["ci_low"]
    out["m2_crs"] = 1 / dea.farrell_output(x2, y, "crs")
    if season >= 2017:
        x3 = np.c_[x1, d.spend3.values / 1e6]
        out["m3_vrs"] = 1 / dea.farrell_output(x3, y, "vrs")
    else:
        out["m3_vrs"] = np.nan
    out["r1_vrs"] = 1 / dea.farrell_output_1d(d.value_xi.values / 1e6, y[:, 0], "vrs")
    y2 = np.c_[d.ppg.values, d.gf.values / (d.gf.values + d.ga.values)]
    out["r2_vrs"] = 1 / dea.farrell_output(x1, y2, "vrs")
    out["n_dmu"] = len(d)
    return out


def main():
    p = pd.read_csv(ROOT / "data" / "panel.csv")
    p = p[p.season.between(2013, 2025) & (p.complete == 1)]
    groups = list(p.groupby(["league", "season"]))
    with Pool(4) as pool:
        parts = pool.map(one_league_season, groups)
    res = pd.concat(parts).sort_values(["league", "season", "club_id"])
    res.to_csv(ROOT / "results" / "dea_scores.csv", index=False)
    cols = ["m1_vrs", "m1_vrs_bc", "m1_crs", "m2_vrs", "m2_vrs_bc", "m3_vrs", "r1_vrs", "r2_vrs"]
    print(res[res.season <= 2023].groupby("league")[cols].mean().round(3))
    print("share efficient (M1 VRS):", (res.m1_vrs > 0.9999).mean().round(3))


if __name__ == "__main__":
    main()
