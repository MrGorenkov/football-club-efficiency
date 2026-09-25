"""Comparisons built on the DEA and SFA scores: descriptives, league means, rank agreement, persistence, RPL
dynamics before/after 2022 (descriptive, no causal claim), Simar-Wilson (2007) second stage, robustness.

Confidence intervals for means are percentile intervals from a cluster bootstrap over clubs (B = 2000, seed 2026).
Writes results/*.csv and results/summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1", "RU1"]
TOP5 = LEAGUES[:-1]
B = 2000
MEASURES = {"m1_vrs": "DEA M1 VRS", "m1_vrs_bc": "DEA M1 VRS, bias-corrected", "m1_crs": "DEA M1 CRS",
            "m2_vrs": "DEA M2 VRS", "m2_vrs_bc": "DEA M2 VRS, bias-corrected", "m3_vrs": "DEA M3 VRS (2017+)",
            "r1_vrs": "DEA R1 (XI value)", "r2_vrs": "DEA R2 (ppg + goal share)",
            "te_s1": "SFA S1 half-normal", "te_s2": "SFA S2 BC95", "te_s3": "SFA S3 TFE (T>=8)",
            "te_s4": "SFA S4 BC95, XI value", "te_s5": "SFA S5 BC95, players in frontier", "te_s6": "SFA S6 BC95, 2013-2025"}


def load():
    panel = pd.read_csv(ROOT / "data" / "panel.csv")
    d = pd.read_csv(RES / "dea_scores.csv").drop(columns=["club"])
    s = pd.read_csv(RES / "sfa_scores.csv").drop(columns=["club", "ppg", "rv"])
    df = panel.merge(d, on=["league", "season", "club_id"], how="inner")
    df = df.merge(s, on=["league", "season", "club_id"], how="left")
    return df


def cluster_boot_mean(d, col, rng, b=B):
    """Mean of col with a percentile CI from resampling clubs."""
    d = d[d[col].notna()]
    g = d.groupby("club_id")[col].agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(g), size=(b, len(g)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return d[col].mean(), *np.quantile(boots, [0.025, 0.975]), boots


def diff_boot(a, b_, col, rng):
    ma, lo, hi, ba = cluster_boot_mean(a, col, rng)
    mb, _, _, bb = cluster_boot_mean(b_, col, rng)
    diff = ba - bb
    return ma - mb, *np.quantile(diff, [0.025, 0.975]), 2 * min((diff <= 0).mean(), (diff >= 0).mean())


def descriptives(df):
    rows = []
    for lg in LEAGUES:
        d = df[df.league == lg]
        rho = d.groupby("season").apply(lambda s: stats.spearmanr(s.squad_value, s.points)[0], include_groups=False)
        rows.append({"league": lg, "club_seasons": len(d), "clubs": d.club_id.nunique(),
                     "ppg_mean": d.ppg.mean(), "squad_value_median_meur": d.squad_value.median() / 1e6,
                     "squad_value_mean_meur": d.squad_value.mean() / 1e6,
                     "value_top_to_bottom_ratio": d.groupby("season").squad_value.apply(lambda s: s.max() / s.min()).median(),
                     "players_used_mean": d.players_used.mean(), "age_mean": d.age.mean(),
                     "foreign_share_mean": d.foreign_share.mean(), "spend_median_meur": d.spend.median() / 1e6,
                     "europe_share": d.europe.mean(), "spearman_value_points_mean": rho.mean(),
                     "spearman_value_points_min": rho.min(), "spearman_value_points_max": rho.max()})
    return pd.DataFrame(rows)


def league_means(df, rng):
    rows = []
    for col in MEASURES:
        for lg in LEAGUES:
            m, lo, hi, _ = cluster_boot_mean(df[df.league == lg], col, rng)
            rows.append({"measure": col, "league": lg, "mean": m, "lo": lo, "hi": hi})
        m, lo, hi, _ = cluster_boot_mean(df[df.league.isin(TOP5)], col, rng)
        rows.append({"measure": col, "league": "TOP5", "mean": m, "lo": lo, "hi": hi})
        dm, dlo, dhi, p = diff_boot(df[df.league == "RU1"], df[df.league.isin(TOP5)], col, rng)
        rows.append({"measure": col, "league": "RU1-TOP5", "mean": dm, "lo": dlo, "hi": dhi, "p_boot": p})
    return pd.DataFrame(rows)


def rank_agreement(df):
    cols = ["m1_vrs", "m1_vrs_bc", "m1_crs", "m2_vrs", "m2_vrs_bc", "r1_vrs", "r2_vrs", "te_s1", "te_s2", "te_s3", "te_s4"]
    pooled = df[cols].corr(method="spearman")
    # within league-season rank agreement, averaged over league-seasons
    within = {}
    for a in cols:
        for b in cols:
            vals = df.groupby(["league", "season"]).apply(
                lambda s: stats.spearmanr(s[a], s[b], nan_policy="omit")[0] if s[[a, b]].notna().all(1).sum() > 4
                else np.nan, include_groups=False)
            within[(a, b)] = vals.mean()
    within = pd.Series(within).unstack()[cols].loc[cols]
    return pooled, within


def persistence(df):
    rows = []
    d = df.sort_values("season")
    lag = d[["league", "club_id", "season"] + list(MEASURES)].copy()
    lag["season"] += 1
    m = d.merge(lag, on=["league", "club_id", "season"], suffixes=("", "_lag"))
    for col in ["m1_vrs", "m1_vrs_bc", "m2_vrs", "te_s1", "te_s2", "te_s3"]:
        for lg in LEAGUES + ["ALL"]:
            s = m if lg == "ALL" else m[m.league == lg]
            s = s[[col, col + "_lag"]].dropna()
            r, p = stats.spearmanr(s[col], s[col + "_lag"])
            rows.append({"measure": col, "league": lg, "n_pairs": len(s), "spearman_t_t1": r, "p": p})
    # quartile transition matrix for bias-corrected M1 (quartiles within league-season)
    m["q"] = m.groupby(["league", "season"]).m1_vrs_bc.transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False))
    q_lag = d[["league", "club_id", "season", "m1_vrs_bc"]].copy()
    q_lag["q_lag"] = q_lag.groupby(["league", "season"]).m1_vrs_bc.transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False))
    q_lag["season"] += 1
    mm = m.merge(q_lag[["league", "club_id", "season", "q_lag"]], on=["league", "club_id", "season"])
    trans = pd.crosstab(mm.q_lag + 1, mm.q + 1, normalize="index")
    return pd.DataFrame(rows), trans, m


def rpl_dynamics(df, rng):
    rows = []
    for lg in LEAGUES:
        for season, s in df[df.league == lg].groupby("season"):
            rows.append({"league": lg, "season": season, "n": len(s),
                         "m1_vrs_bc_mean": s.m1_vrs_bc.mean(), "m1_vrs_mean": s.m1_vrs.mean(),
                         "m1_vrs_sd": s.m1_vrs.std(), "share_efficient_m1": (s.m1_vrs > 0.9999).mean(),
                         "te_s6_mean": s.te_s6.mean(), "spearman_value_points": stats.spearmanr(s.squad_value, s.points)[0],
                         "value_top_to_bottom_ratio": s.squad_value.max() / s.squad_value.min(),
                         "squad_value_total_meur": s.squad_value.sum() / 1e6, "foreign_share": s.foreign_share.mean()})
    dyn = pd.DataFrame(rows)
    pp = []
    for col in ["m1_vrs", "m1_vrs_bc", "m2_vrs_bc", "te_s6"]:
        for lg in ["RU1", "TOP5"]:
            s = df[df.league == lg] if lg == "RU1" else df[df.league.isin(TOP5)]
            post, pre = s[s.season >= 2022], s[s.season <= 2021]
            d, lo, hi, p = diff_boot(post, pre, col, rng)
            pp.append({"measure": col, "group": lg, "pre_mean": pre[col].mean(), "post_mean": post[col].mean(),
                       "post_minus_pre": d, "lo": lo, "hi": hi, "p_boot": p, "n_pre": pre[col].notna().sum(),
                       "n_post": post[col].notna().sum()})
    # dispersion of the value-points ranking: Spearman by season, pre/post means
    sp = dyn.assign(group=np.where(dyn.league == "RU1", "RU1", "TOP5"), post=dyn.season >= 2022) \
        .groupby(["group", "post"]).spearman_value_points.agg(["mean", "size"]).reset_index()
    return dyn, pd.DataFrame(pp), sp


def truncreg_fit(y, Z):
    """ML truncated normal regression, left truncation at 1: y = Z b + e, e ~ N(0, s^2), y > 1."""
    def nll(t):
        b, s = t[:-1], np.exp(t[-1])
        mu = Z @ b
        return -(stats.norm.logpdf((y - mu) / s) - np.log(s) - stats.norm.logcdf((mu - 1) / s)).sum()
    b0, *_ = np.linalg.lstsq(Z, y, rcond=None)
    t0 = np.r_[b0, np.log((y - Z @ b0).std())]
    r = optimize.minimize(nll, t0, method="BFGS", options={"gtol": 1e-8, "maxiter": 5000})
    return r.x[:-1], np.exp(r.x[-1])


def sw2007(df, rng, L=2000):
    """Simar & Wilson (2007), Algorithm 1: truncated regression of the output Farrell measures delta > 1 (DEA M1 VRS)
    on environmental variables, parametric bootstrap for inference."""
    d = df[df.season.between(2013, 2023) & df.promoted.notna()].copy()
    d["delta"] = 1 / d.m1_vrs
    d = d[d.delta > 1 + 1e-9]
    zc = pd.DataFrame({"const": 1.0, "age_c": d.age - 27, "age_c2": (d.age - 27) ** 2,
                       "lnpl_c": np.log(d.players_used) - np.log(28), "promoted": d.promoted, "europe": d.europe,
                       "foreign_share": d.foreign_share}, index=d.index)
    for lg in LEAGUES[1:]:
        zc[f"d_{lg}"] = (d.league == lg).astype(float)
    zc["ru_post"] = ((d.league == "RU1") & (d.season >= 2022)).astype(float)
    Z, y = zc.values, d.delta.values
    b, s = truncreg_fit(y, Z)
    mu = Z @ b
    boots = np.empty((L, len(b) + 1))
    for l in range(L):
        a = (1 - mu) / s
        e = stats.truncnorm.rvs(a, np.inf, loc=0, scale=s, random_state=rng)
        bb, ss = truncreg_fit(mu + e, Z)
        boots[l] = np.r_[bb, ss]
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    names = list(zc.columns) + ["sigma"]
    est = np.r_[b, s]
    return pd.DataFrame({"param": names, "est": est, "boot_sd": boots.std(0), "lo": lo, "hi": hi}), len(y), len(df)


def dea_size_check(df, draws=200, seed=7):
    """RPL league-seasons have 16 clubs, the others 18-20; DEA efficiency rises mechanically with fewer units.
    Recompute M1 VRS for random 16-club subsets of each top-5 league-season and compare the means."""
    import dea
    rng = np.random.default_rng(seed)
    out = []
    for (lg, season), d in df[df.league.isin(TOP5)].groupby(["league", "season"]):
        x, y = d.squad_value.values / 1e6, d.ppg.values
        means = []
        for _ in range(draws):
            idx = rng.choice(len(d), 16, replace=False)
            means.append((1 / dea.farrell_output_1d(x[idx], y[idx], "vrs")).mean())
        out.append({"league": lg, "season": season, "n": len(d), "m1_vrs_full": d.m1_vrs.mean(),
                    "m1_vrs_sub16": float(np.mean(means))})
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(2026)
    all_seasons = load()
    df = all_seasons[all_seasons.season.between(2013, 2023)].copy()
    summary = {"n_club_seasons_main": int(len(df)), "n_clubs_main": int(df.club_id.nunique()),
               "n_league_seasons_main": int(df.groupby(["league", "season"]).ngroups),
               "n_club_seasons_2013_2025": int(len(all_seasons)),
               "excluded_incomplete": pd.read_csv(ROOT / "data" / "panel.csv").query("complete == 0 and season >= 2013")
               [["league", "season", "club"]].to_dict("records")}
    desc = descriptives(df)
    desc.to_csv(RES / "descriptives.csv", index=False)
    lm = league_means(df, rng)
    lm.to_csv(RES / "league_efficiency.csv", index=False)
    pooled, within = rank_agreement(df)
    pooled.to_csv(RES / "rank_agreement_pooled.csv")
    within.to_csv(RES / "rank_agreement_within.csv")
    pers, trans, _ = persistence(df)
    pers.to_csv(RES / "persistence.csv", index=False)
    trans.to_csv(RES / "transition_m1_bc.csv")
    dyn, pp, sp = rpl_dynamics(all_seasons, rng)
    dyn.to_csv(RES / "league_season_dynamics.csv", index=False)
    pp.to_csv(RES / "rpl_pre_post.csv", index=False)
    sp.to_csv(RES / "spearman_value_points_pre_post.csv", index=False)
    sw, n_trunc, n_all = sw2007(all_seasons, rng)
    sw.to_csv(RES / "sw2007_second_stage.csv", index=False)
    summary["sw2007_n_inefficient"] = n_trunc
    d = df
    summary["share_efficient_m1_vrs"] = float((d.m1_vrs > 0.9999).mean())
    summary["share_efficient_m2_vrs"] = float((d.m2_vrs > 0.9999).mean())
    summary["mean_bias_m1"] = float(d.m1_vrs_bias.mean())
    summary["median_ci_width_m1"] = float((d.m1_vrs_hi - d.m1_vrs_lo).median())
    summary["median_bandwidth_m1"] = float(d.m1_h.median())
    summary["corr_m1_vrs_bc_vs_te_s2_pooled"] = float(pooled.loc["m1_vrs_bc", "te_s2"])
    # clubs: most / least efficient on average (>= 8 seasons), bias-corrected M1
    g = df.groupby(["league", "club"]).agg(n=("m1_vrs_bc", "size"), m1_bc=("m1_vrs_bc", "mean"),
                                            te_s2=("te_s2", "mean"), ppg=("ppg", "mean"))
    g = g[g.n >= 8].sort_values("m1_bc")
    g.to_csv(RES / "club_average_efficiency.csv")
    # robustness: RPL minus top-5 across measures, and excluding the COVID seasons 2019-2020
    rob = lm[lm.league == "RU1-TOP5"][["measure", "mean", "lo", "hi", "p_boot"]].assign(sample="2013-2023")
    nc = df[~df.season.isin([2019, 2020])]
    r2 = []
    for col in ["m1_vrs", "m1_vrs_bc", "m2_vrs_bc", "te_s2"]:
        dm, lo, hi, p = diff_boot(nc[nc.league == "RU1"], nc[nc.league.isin(TOP5)], col, rng)
        r2.append({"measure": col, "mean": dm, "lo": lo, "hi": hi, "p_boot": p, "sample": "2013-2023 without 2019-2020"})
    pd.concat([rob, pd.DataFrame(r2)]).to_csv(RES / "robustness_rpl_vs_top5.csv", index=False)
    sz = dea_size_check(df)
    sz.to_csv(RES / "dea_sample_size_check.csv", index=False)
    summary["dea_m1_vrs_top5_full_mean"] = float(sz.m1_vrs_full.mean())
    summary["dea_m1_vrs_top5_sub16_mean"] = float(sz.m1_vrs_sub16.mean())
    summary["dea_m1_vrs_rpl_mean"] = float(df[df.league == "RU1"].m1_vrs.mean())
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
    pd.set_option("display.width", 220)
    for name, t in [("desc", desc), ("league means", lm[lm.measure.isin(["m1_vrs_bc", "m2_vrs_bc", "te_s2", "te_s3"])]),
                    ("within", within), ("pers", pers[pers.league == "ALL"]), ("trans", trans), ("pre/post", pp),
                    ("spearman pre/post", sp), ("sw2007", sw), ("clubs", pd.concat([g.head(5), g.tail(5)]))]:
        print("==", name)
        print(t.round(3).to_string())
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
