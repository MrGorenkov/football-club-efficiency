"""Figures for the manuscript: greyscale only, white background, Times New Roman-like serif 9 pt (journal rules).
Writes figures/fig1_frontier.{pdf,png}, fig2_league_means.{pdf,png}, fig3_rpl_dynamics.{pdf,png}.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
NAMES = {"GB1": "Англия", "ES1": "Испания", "IT1": "Италия", "L1": "Германия", "FR1": "Франция", "RU1": "РПЛ",
         "TOP5": "Топ-5"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": "0.3", "axes.linewidth": 0.6, "xtick.color": "0.3",
                     "ytick.color": "0.3", "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def save(fig, name, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1():
    s = pd.read_csv(RES / "sfa_scores.csv").dropna(subset=["rv"])
    c = pd.read_csv(RES / "sfa_coefficients.csv").query("model == 'S1'").set_index("param").est
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    top = s[s.league != "RU1"]
    ru = s[s.league == "RU1"]
    ax.scatter(top.rv, top.ppg, s=9, facecolor="0.75", edgecolor="none", label="Клубы топ-5 лиг")
    ax.scatter(ru.rv, ru.ppg, s=12, facecolor="white", edgecolor="black", linewidth=0.6, label="Клубы РПЛ")
    g = np.linspace(s.rv.min(), s.rv.max(), 200)
    ax.plot(g, np.exp(c["const"] + c["rv"] * g + c["rv2"] * g ** 2), color="black", lw=1.2,
            label="Граница S1 (Англия)")
    ax.set_xlabel("Относительная стоимость состава: ln(стоимость) минус среднее по лиге-сезону")
    ax.set_ylabel("Очки за матч")
    ax.set_ylim(0.3, 3.0)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig1_frontier")


def fig2():
    lm = pd.read_csv(RES / "league_efficiency.csv")
    order = ["GB1", "ES1", "IT1", "L1", "FR1", "TOP5", "RU1"]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    specs = [("m1_vrs_bc", "DEA M1, со снятым смещением", "o", "black", -0.15),
             ("te_s2", "SFA S2 (Battese–Coelli 1995)", "s", "0.55", 0.15)]
    for col, lab, mk, colr, off in specs:
        d = lm[lm.measure == col].set_index("league").loc[order]
        y = np.arange(len(order)) + off
        ax.errorbar(d["mean"], y, xerr=[d["mean"] - d.lo, d.hi - d["mean"]], fmt=mk, color=colr, ms=5,
                    elinewidth=0.9, capsize=0, label=lab)
    ax.set_yticks(range(len(order)), [NAMES[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("Средняя эффективность (95%-й кластерный бутстрап-интервал по клубам)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "fig2_league_means")


def fig3():
    d = pd.read_csv(RES / "league_season_dynamics.csv")
    d["grp"] = np.where(d.league == "RU1", "RU1", "TOP5")
    g = d.groupby(["grp", "season"])[["m1_vrs_bc_mean", "spearman_value_points"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), sharex=True)
    for ax, col, lab in [(axes[0], "m1_vrs_bc_mean", "Средняя эффективность DEA M1\n(со снятым смещением)"),
                         (axes[1], "spearman_value_points", "Ранговая корреляция\nстоимости состава и очков")]:
        for grp, style, mk in [("TOP5", dict(color="0.55", ls="--"), "s"), ("RU1", dict(color="black", ls="-"), "o")]:
            s = g[g.grp == grp]
            ax.plot(s.season, s[col], marker=mk, ms=4, lw=1.1, label=NAMES[grp], **style)
        ax.axvline(2021.5, color="0.6", lw=0.6, ls=":")
        ax.set_ylabel(lab)
        ax.set_xticks(range(2013, 2026, 2), [f"{y % 100:02d}/{(y + 1) % 100:02d}" for y in range(2013, 2026, 2)])
        ax.set_xlabel("Сезон")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2)
    save(fig, "fig3_rpl_dynamics", rect=(0, 0, 1, 0.92))


def main():
    FIG.mkdir(exist_ok=True)
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
