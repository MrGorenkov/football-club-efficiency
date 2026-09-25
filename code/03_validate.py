"""Checks of the DEA and SFA code on known examples. Writes results/validation.json.

1. DEA, textbook example: 12 hospitals (2 inputs: doctors, nurses; 2 outputs: outpatients, inpatients) from
   Cooper, Seiford & Tone (2007, Hospital example of ch. 1 and 3); published CCR efficiencies are hard-coded below.
2. DEA vs R package Benchmarking (Bogetoft & Otto) on the Charnes-Cooper-Rhodes (1981) Program Follow Through data
   (70 schools, 5 inputs, 3 outputs), CRS and VRS, output orientation.
3. 1-input/1-output closed form vs LP on random data.
4. SFA vs R package frontier (Coelli & Henningsen): half-normal model on front41Data (the FRONTIER 4.1 example) and the
   Battese-Coelli (1995) model with inefficiency determinants on riceProdPhil.
5. Monte Carlo: recovery of SFA parameters and coverage of Simar-Wilson (1998) bootstrap intervals for a known DGP.
Reference CSVs from R are produced by code/validation/r_reference.R (Rscript) and are committed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import dea
import sfa

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "code" / "validation"


def main():
    res = {}
    X = np.array([[20, 151], [19, 131], [25, 160], [27, 168], [22, 158], [55, 255], [33, 235], [31, 206], [30, 244],
                  [50, 268], [53, 306], [38, 284]], float)
    Y = np.array([[100, 90], [150, 50], [160, 55], [180, 72], [94, 66], [230, 90], [220, 88], [152, 80], [190, 100],
                  [250, 100], [260, 147], [250, 120]], float)
    published = np.array([1, 1, 0.883, 1, 0.763, 0.835, 0.902, 0.796, 0.960, 0.871, 0.955, 0.958])
    ours = 1 / dea.farrell_output(X, Y, "crs")
    res["dea_hospital_ccr_max_abs_diff_3dp"] = float(np.abs(np.round(ours, 3) - published).max())
    res["dea_hospital_ccr_ours"] = np.round(ours, 4).tolist()

    ch = pd.read_csv(V / "charnes1981.csv")
    r = pd.read_csv(V / "r_dea_charnes1981.csv")
    Xc, Yc = ch[[f"x{i}" for i in range(1, 6)]].values, ch[["y1", "y2", "y3"]].values
    for rts in ("crs", "vrs"):
        res[f"dea_charnes1981_{rts}_max_abs_diff_vs_R"] = float(np.abs(dea.farrell_output(Xc, Yc, rts) - r[rts]).max())

    rng = np.random.default_rng(20260925)
    worst = 0.0
    for _ in range(200):
        x = rng.lognormal(3, 1, 20)
        y = np.sqrt(x) * rng.uniform(0.4, 1, 20)
        for rts in ("crs", "vrs"):
            worst = max(worst, np.abs(dea.farrell_output(x[:, None], y[:, None], rts) - dea.farrell_output_1d(x, y, rts)).max())
    res["dea_1d_closed_form_vs_lp_max_abs_diff"] = float(worst)

    f41 = pd.read_csv(V / "front41Data.csv")
    rr = pd.read_csv(V / "r_sfa_front41.csv")
    Xf = np.c_[np.ones(len(f41)), np.log(f41.capital), np.log(f41.labour)]
    m = sfa.fit(np.log(f41.output.values), Xf)
    te_r = pd.read_csv(V / "r_sfa_front41_te.csv").te.values
    res["sfa_front41"] = {"ours": np.round(m["est"], 5).tolist(), "R": np.round(rr.est.values, 5).tolist(),
                          "max_abs_diff_est": float(np.abs(m["est"] - rr.est.values).max()),
                          "max_abs_diff_se": float(np.abs(m["se"] - rr.se.values).max()),
                          "loglik_ours": m["loglik"], "loglik_R": float(rr.loglik[0]),
                          "max_abs_diff_te": float(np.abs(m["te"] - te_r).max())}
    rice = pd.read_csv(V / "riceProdPhil.csv")
    rr = pd.read_csv(V / "r_sfa_rice_bc95.csv")
    Xr = np.c_[np.ones(len(rice)), np.log(rice.AREA), np.log(rice.LABOR), np.log(rice.NPK)]
    Zr = np.c_[np.ones(len(rice)), rice.EDYRS, rice.BANRAT]
    m = sfa.fit(np.log(rice.PROD.values), Xr, Zr, "bc95")
    te_r = pd.read_csv(V / "r_sfa_rice_te.csv").te.values
    res["sfa_rice_bc95"] = {"ours": np.round(m["est"], 5).tolist(), "R": np.round(rr.est.values, 5).tolist(),
                            "max_abs_diff_frontier_coef": float(np.abs(m["est"][:4] - rr.est.values[:4]).max()),
                            "loglik_ours": m["loglik"], "loglik_R": float(rr.loglik[0]),
                            "max_abs_diff_te": float(np.abs(m["te"] - te_r).max())}

    # Monte Carlo: SFA recovery (half-normal), 100 replications, n = 1000
    est = []
    for _ in range(100):
        n = 1000
        x = rng.normal(size=n)
        y = 1 + 0.5 * x + rng.normal(0, 0.15, n) - np.abs(rng.normal(0, 0.3, n))
        est.append(sfa.fit(y, np.c_[np.ones(n), x])["est"])
    est = np.array(est)
    res["mc_sfa_halfnormal"] = {"true": [1, 0.5, 0.1125, 0.8], "mean": np.round(est.mean(0), 4).tolist(),
                                "sd": np.round(est.std(0), 4).tolist()}

    # Monte Carlo: coverage of SW98 95% intervals, 1 input / 1 output, VRS, n = 20 (league size), 200 replications
    cover, bias_raw, bias_bc = [], [], []
    for rep in range(200):
        n = 20
        x = rng.uniform(1, 10, n)
        eff = np.exp(-np.abs(rng.normal(0, 0.25, n)))
        y = np.sqrt(x) * eff
        b = dea.sw98_bootstrap(x[:, None], y[:, None], "vrs", B=500, rng=rep)
        true_delta = 1 / eff
        cover.append(np.mean((b["ci_low"] <= true_delta) & (true_delta <= b["ci_high"])))
        bias_raw.append(np.mean(b["delta_hat"] - true_delta))
        bias_bc.append(np.mean(b["delta_bc"] - true_delta))
    res["mc_sw98_vrs_n20"] = {"coverage_95": float(np.mean(cover)), "mean_error_delta_hat": float(np.mean(bias_raw)),
                              "mean_error_delta_bc": float(np.mean(bias_bc))}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
