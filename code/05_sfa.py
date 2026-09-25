"""Stochastic frontier models of points per game, pooled panel of club-seasons 2013-2023 (complete squad data).

Frontier: ln(ppg) = b0 + b1 rv + b2 rv^2 + league dummies + v - u, where rv = ln(squad value) minus the mean of
ln(squad value) in the league-season (relative squad strength; removes league scale and wage/price inflation).
Models:
  S1  half-normal u (Aigner, Lovell & Schmidt 1977)
  S2  Battese & Coelli (1995): u ~ N+(z'd, s_u^2), z = age, age^2 (centred at 27), ln players used (centred at ln 28),
      promoted, UEFA participation, foreign minute share, RPL x season >= 2022
  S3  true fixed effects (Greene 2005): club dummies + half-normal u, clubs with >= 8 seasons in the window
  S4  S2 with the minutes-weighted XI value instead of squad value (robustness)
  S5  S2 with ln players used moved from z into the frontier (robustness)
  S6  S2 on seasons 2013-2025 (robustness)
  H1  half-normal with league-specific slopes on rv (value-points gradient by league) and an RPL x post-2022 slope shift
LR tests against OLS use the mixed chi-square 0.5 chi2(q) + 0.5 chi2(q+1) (Kodde & Palm 1986).
Writes results/sfa_coefficients.csv, results/sfa_scores.csv, results/sfa_tests.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import sfa

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1", "RU1"]


def mixed_chi2_crit(q, alpha=0.05):
    """Critical value of 0.5 chi2(q) + 0.5 chi2(q+1) (one parameter on the boundary, q equality restrictions)."""
    def tail(c):
        a = stats.chi2.sf(c, q) if q > 0 else 0.0
        return 0.5 * a + 0.5 * stats.chi2.sf(c, q + 1) - alpha
    return optimize.brentq(tail, 1e-6, 100)


def prepare(p, value_col="squad_value"):
    p = p.copy()
    lv = np.log(p[value_col])
    p["rv"] = lv - lv.groupby([p.league, p.season]).transform("mean")
    p["y"] = np.log(p.ppg)
    p["age_c"] = p.age - 27
    p["lnpl_c"] = np.log(p.players_used) - np.log(28)
    p["ru_post"] = ((p.league == "RU1") & (p.season >= 2022)).astype(float)
    for lg in LEAGUES[1:]:
        p[f"d_{lg}"] = (p.league == lg).astype(float)
    return p


FRONT = ["const", "rv", "rv2"] + [f"d_{lg}" for lg in LEAGUES[1:]]
ZVARS = ["z_const", "age_c", "age_c2", "lnpl_c", "promoted", "europe", "foreign_share", "ru_post"]


def design(p, front=FRONT, zvars=ZVARS):
    X = pd.DataFrame({"const": 1.0, "rv": p.rv, "rv2": p.rv ** 2}, index=p.index)
    for c in front:
        if c not in X:
            X[c] = p[c]
    Z = pd.DataFrame({"z_const": 1.0, "age_c": p.age_c, "age_c2": p.age_c ** 2}, index=p.index)
    for c in zvars:
        if c not in Z:
            Z[c] = p[c]
    return X[front], Z[zvars]


def table(name, r, xnames, znames):
    names = xnames + znames + ["sigma2", "gamma"]
    return pd.DataFrame({"model": name, "param": names, "est": r["est"], "se": r["se"],
                         "z": r["est"] / r["se"], "p": 2 * stats.norm.sf(np.abs(r["est"] / r["se"]))})


def main():
    panel = pd.read_csv(ROOT / "data" / "panel.csv")
    base = panel[(panel.complete == 1) & panel.promoted.notna()]
    p = prepare(base[base.season.between(2013, 2023)])
    y = p.y.values
    coefs, tests = [], {}
    scores = p[["league", "season", "club_id", "club", "ppg", "rv"]].copy()

    X, Z = design(p)
    s1 = sfa.fit(y, X.values)
    coefs.append(table("S1", s1, list(X.columns), []))
    ll_ols = sfa.ols_loglik(y, X.values)
    tests["n_obs"] = int(len(p))
    tests["n_clubs"] = int(p.club_id.nunique())
    tests["ols_resid_skewness"] = float(sfa.resid_skewness(y, X.values))
    tests["S1_LR_vs_OLS"] = float(2 * (s1["loglik"] - ll_ols))
    tests["S1_LR_crit5"] = mixed_chi2_crit(0)
    scores["te_s1"] = s1["te"]

    s2 = sfa.fit(y, X.values, Z.values, "bc95")
    coefs.append(table("S2", s2, list(X.columns), list(Z.columns)))
    q = Z.shape[1]
    tests["S2_LR_vs_OLS"] = float(2 * (s2["loglik"] - ll_ols))
    tests["S2_LR_crit5"] = mixed_chi2_crit(q)
    tests["S2_LR_vs_S1"] = float(2 * (s2["loglik"] - s1["loglik"]))
    tests["S2_LR_vs_S1_df"] = q  # S1 is S2 with d = 0 (mu = 0): q equality restrictions
    tests["S2_LR_vs_S1_p"] = float(stats.chi2.sf(tests["S2_LR_vs_S1"], q))
    tests["loglik"] = {"OLS": ll_ols, "S1": s1["loglik"], "S2": s2["loglik"]}
    scores["te_s2"] = s2["te"]

    # true fixed effects: on the full sample (median T ~ 6) the ML estimate sits on the boundary gamma -> 1
    # (deterministic frontier, incidental-parameter problem); it is recorded and S3 uses clubs with >= 8 seasons
    clubs = pd.get_dummies(p.club_id, prefix="c", dtype=float)
    X3a = pd.concat([p[["rv"]], (p.rv ** 2).rename("rv2"), clubs], axis=1)
    s3a = sfa.fit(y, X3a.values)
    tests["S3_full_sample"] = {"gamma": float(s3a["est"][-1]), "b_rv": float(s3a["est"][0]),
                               "loglik": s3a["loglik"], "n_clubs": int(clubs.shape[1])}
    T = p.groupby("club_id").season.transform("size")
    p3 = p[T >= 8]
    clubs = pd.get_dummies(p3.club_id, prefix="c", dtype=float)
    X3 = pd.concat([p3[["rv"]], (p3.rv ** 2).rename("rv2"), clubs], axis=1)
    s3 = sfa.fit(p3.y.values, X3.values)
    t3 = table("S3", s3, list(X3.columns), [])
    coefs.append(t3[~t3.param.str.startswith("c_")])
    b_lsdv, *_ = np.linalg.lstsq(X3.values, p3.y.values, rcond=None)
    tests["S3"] = {"n_obs": int(len(p3)), "n_clubs": int(clubs.shape[1]), "loglik": s3["loglik"],
                   "ols_lsdv_loglik": sfa.ols_loglik(p3.y.values, X3.values),
                   "lsdv_b_rv": float(b_lsdv[0]), "lsdv_b_rv2": float(b_lsdv[1]),
                   "resid_skewness": float(sfa.resid_skewness(p3.y.values, X3.values))}
    tests["S3"]["LR_vs_LSDV"] = float(2 * (s3["loglik"] - tests["S3"]["ols_lsdv_loglik"]))
    scores = scores.join(pd.Series(s3["te"], index=p3.index, name="te_s3"))

    p4 = prepare(base[base.season.between(2013, 2023)], "value_xi")
    X4, Z4 = design(p4)
    s4 = sfa.fit(p4.y.values, X4.values, Z4.values, "bc95")
    coefs.append(table("S4", s4, list(X4.columns), list(Z4.columns)))
    scores["te_s4"] = s4["te"]

    X5, Z5 = design(p, FRONT + ["lnpl_c"], [z for z in ZVARS if z != "lnpl_c"])
    s5 = sfa.fit(y, X5.values, Z5.values, "bc95")
    coefs.append(table("S5", s5, list(X5.columns), list(Z5.columns)))
    scores["te_s5"] = s5["te"]

    p6 = prepare(base[base.season.between(2013, 2025)])
    X6, Z6 = design(p6)
    s6 = sfa.fit(p6.y.values, X6.values, Z6.values, "bc95")
    coefs.append(table("S6", s6, list(X6.columns), list(Z6.columns)))
    s6_scores = p6[["league", "season", "club_id"]].assign(te_s6=s6["te"])

    # value-points gradient by league, RPL slope shift after 2022
    H = pd.DataFrame({"const": 1.0}, index=p.index)
    for lg in LEAGUES:
        H[f"rv_{lg}"] = p.rv * (p.league == lg)
    H["rv_RU1_post"] = p.rv * p.ru_post
    H["rv2"] = p.rv ** 2
    for lg in LEAGUES[1:]:
        H[f"d_{lg}"] = p[f"d_{lg}"]
    H["ru_post"] = p.ru_post
    h1 = sfa.fit(y, H.values)
    coefs.append(table("H1", h1, list(H.columns), []))
    # Wald test: RPL slope equals the mean top-5 slope
    names = list(H.columns)
    R = np.zeros(len(h1["est"]))
    R[names.index("rv_RU1")] = 1
    for lg in LEAGUES[:-1]:
        R[names.index(f"rv_{lg}")] = -0.2
    diff = R @ h1["est"]
    se = np.sqrt(R @ h1["cov"] @ R)
    tests["H1_rpl_minus_top5_slope"] = {"diff": float(diff), "se": float(se), "p": float(2 * stats.norm.sf(abs(diff / se)))}

    coefs = pd.concat(coefs)
    coefs.to_csv(ROOT / "results" / "sfa_coefficients.csv", index=False)
    scores = scores.merge(s6_scores, on=["league", "season", "club_id"], how="outer")
    scores.to_csv(ROOT / "results" / "sfa_scores.csv", index=False)
    (ROOT / "results" / "sfa_tests.json").write_text(json.dumps(tests, indent=1))
    pd.set_option("display.width", 200)
    print(coefs.round(4).to_string())
    print(json.dumps(tests, indent=1))
    print(scores.groupby("league")[["te_s1", "te_s2", "te_s3", "te_s4", "te_s5"]].mean().round(3))


if __name__ == "__main__":
    main()
