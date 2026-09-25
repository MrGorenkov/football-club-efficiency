"""Club-season panel for GB1, ES1, IT1, L1, FR1, RU1, seasons 2012-2025 (season = start year).

Definitions (fixed before any efficiency estimate):
  - output: league points computed from match results (3/1/0), points per game (ppg); point deductions are ignored
    on purpose, they are not sporting output;
  - squad = players with > 0 league minutes for the club in the season (appearances table);
  - player value = last Transfermarkt valuation in the 365 days up to 1 September of the season's start year (after the
    summer window); if none, the first valuation in the following 180 days; players without either are counted as
    missing (their minute share is reported as coverage);
  - squad value (main input) = sum of values of the squad; minutes-weighted value (robustness) = 11 x sum of values
    weighted by the player's share of the club's league minutes;
  - transfer spending = sum of positive fees paid by the club for incoming transfers dated 1 June (year s) - 31 March
    (year s+1); spend3 = mean over seasons s-2..s;
  - players used = squad size; age = minutes-weighted mean age on 1 September;
  - promoted = club did not play the same league in season s-1; europe = club played a UEFA club competition game in s;
  - foreign share = minute share of players whose citizenship differs from the league's country;
  - complete = the club has >= 11 players with minutes in >= 50% of its league games (club-seasons failing it are excluded
    from the efficiency analysis, because their squad value is understated).
Writes data/panel.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LEAGUES = {"GB1": "England", "ES1": "Spain", "IT1": "Italy", "L1": "Germany", "FR1": "France", "RU1": "Russia"}
SEASONS = range(2012, 2026)


def club_results(g):
    home = pd.DataFrame({"club_id": g.home_club_id, "gf": g.home_club_goals, "ga": g.away_club_goals})
    away = pd.DataFrame({"club_id": g.away_club_id, "gf": g.away_club_goals, "ga": g.home_club_goals})
    r = pd.concat([home.assign(league=g.competition_id.values, season=g.season.values),
                   away.assign(league=g.competition_id.values, season=g.season.values)])
    r["pts"] = np.select([r.gf > r.ga, r.gf == r.ga], [3, 1], 0)
    out = r.groupby(["league", "season", "club_id"]).agg(games=("pts", "size"), points=("pts", "sum"),
                                                          gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
    out["ppg"] = out.points / out.games
    out["rank"] = out.assign(gd=out.gf - out.ga).sort_values(["points", "gd", "gf"], ascending=False) \
        .groupby(["league", "season"]).cumcount().reindex(out.index) + 1
    return out


def main():
    g = pd.read_csv(RAW / "games.csv.gz")
    lg = g[g.competition_id.isin(LEAGUES) & g.season.isin(SEASONS) & g.home_club_goals.notna()]
    panel = club_results(lg)
    names = pd.read_csv(RAW / "clubs.csv.gz", usecols=["club_id", "name"]).set_index("club_id").name
    panel["club"] = panel.club_id.map(names)

    prev = set(zip(panel.league, panel.season + 1, panel.club_id))
    panel["promoted"] = [int((l, s, c) not in prev) for l, s, c in zip(panel.league, panel.season, panel.club_id)]
    panel.loc[panel.season == min(SEASONS), "promoted"] = np.nan
    eu = g[g.competition_type == "international_cup"]
    eu_set = set(zip(eu.season, eu.home_club_id)) | set(zip(eu.season, eu.away_club_id))
    panel["europe"] = [int((s, c) in eu_set) for s, c in zip(panel.season, panel.club_id)]

    app = pd.read_csv(RAW / "appearances.csv.gz", usecols=["game_id", "player_id", "player_club_id", "competition_id",
                                                           "minutes_played"])
    app = app.merge(lg[["game_id", "season"]], on="game_id")
    app = app[app.minutes_played > 0]
    per_game = app.groupby(["competition_id", "season", "player_club_id", "game_id"]).size()
    cov = (per_game >= 11).groupby(level=[0, 1, 2]).sum().rename("app_games")
    cov.index.names = ["league", "season", "club_id"]
    sq = app.groupby(["competition_id", "season", "player_club_id", "player_id"]).minutes_played.sum().reset_index()
    sq.columns = ["league", "season", "club_id", "player_id", "minutes"]
    sq["ref"] = pd.to_datetime(sq.season.astype(str) + "-09-01")

    val = pd.read_csv(RAW / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"],
                      parse_dates=["date"])
    val = val[val.market_value_in_eur > 0].sort_values("date")
    keys = sq[["player_id", "ref"]].drop_duplicates().sort_values("ref")
    back = pd.merge_asof(keys, val.rename(columns={"date": "vdate"}), left_on="ref", right_on="vdate", by="player_id",
                         direction="backward", tolerance=pd.Timedelta(days=365))
    fwd = pd.merge_asof(keys, val.rename(columns={"date": "vdate"}), left_on="ref", right_on="vdate", by="player_id",
                        direction="forward", tolerance=pd.Timedelta(days=180))
    back["value"] = back.market_value_in_eur.fillna(fwd.market_value_in_eur)
    back["value_source"] = np.where(back.market_value_in_eur.notna(), "backward",
                                    np.where(fwd.market_value_in_eur.notna(), "forward", "missing"))
    sq = sq.merge(back[["player_id", "ref", "value", "value_source"]], on=["player_id", "ref"], how="left")

    pl = pd.read_csv(RAW / "players.csv.gz", usecols=["player_id", "date_of_birth", "country_of_citizenship"],
                     parse_dates=["date_of_birth"])
    sq = sq.merge(pl, on="player_id", how="left")
    sq["age"] = (sq.ref - sq.date_of_birth).dt.days / 365.25
    sq["foreign"] = (sq.country_of_citizenship != sq.league.map(LEAGUES)).astype(float)
    sq.loc[sq.country_of_citizenship.isna(), "foreign"] = np.nan
    sq["tot_min"] = sq.groupby(["league", "season", "club_id"]).minutes.transform("sum")
    sq["w"] = sq.minutes / sq.tot_min
    sq["has_value"] = sq.value.notna()

    def agg(d):
        wv = d.w[d.has_value]
        return pd.Series({
            "players_used": len(d),
            "squad_value": d.value.sum(),
            "value_xi": 11 * (d.value[d.has_value] * wv).sum() / wv.sum(),
            "value_cov_minutes": wv.sum(),
            "value_cov_players": d.has_value.mean(),
            "value_forward_share": (d.value_source == "forward").mean(),
            "age": np.average(d.age.fillna(d.age.mean()), weights=d.w),
            "foreign_share": np.nansum(d.foreign * d.w) / d.w[d.foreign.notna()].sum(),
        })

    sqa = sq.groupby(["league", "season", "club_id"]).apply(agg, include_groups=False).reset_index()
    panel = panel.merge(sqa, on=["league", "season", "club_id"], how="left")
    panel = panel.join(cov, on=["league", "season", "club_id"])
    # squad data are usable only if the club has >= 11 players with minutes in >= 50% of its league games
    panel["app_cov"] = panel.app_games.fillna(0) / panel.games
    panel["complete"] = (panel.app_cov >= 0.5).astype(int)

    tr = pd.read_csv(RAW / "transfers.csv.gz", parse_dates=["transfer_date"])
    tr = tr[tr.transfer_fee > 0]
    y = tr.transfer_date.dt.year
    tr["season"] = np.where(tr.transfer_date.dt.month >= 6, y, y - 1)
    spend = tr.groupby(["season", "to_club_id"]).transfer_fee.sum().rename("spend")
    income = tr.groupby(["season", "from_club_id"]).transfer_fee.sum().rename("income")
    panel = panel.join(spend, on=["season", "club_id"]).join(income, on=["season", "club_id"])
    panel[["spend", "income"]] = panel[["spend", "income"]].fillna(0.0)
    s_all = spend.reset_index().rename(columns={"to_club_id": "club_id"})
    for k in (1, 2):
        lagk = s_all.assign(season=s_all.season + k).rename(columns={"spend": f"spend_l{k}"})
        panel = panel.merge(lagk, on=["season", "club_id"], how="left")
    panel[["spend_l1", "spend_l2"]] = panel[["spend_l1", "spend_l2"]].fillna(0.0)
    panel["spend3"] = panel[["spend", "spend_l1", "spend_l2"]].mean(axis=1)
    panel = panel.drop(columns=["spend_l1", "spend_l2"])
    panel["post2022"] = (panel.season >= 2022).astype(int)
    panel = panel.sort_values(["league", "season", "rank"])
    panel.to_csv(ROOT / "data" / "panel.csv", index=False)
    main_ = panel[panel.season.between(2013, 2023) & (panel.complete == 1)]
    cov_by_season = main_.groupby("season").value_cov_minutes.mean()
    tr_rows = pd.read_csv(RAW / "transfers.csv.gz", usecols=["transfer_season"]).transfer_season.value_counts()
    coverage = {"value_cov_minutes_season_mean_min": float(cov_by_season.min()),
                "value_cov_minutes_season_mean_max": float(cov_by_season.max()),
                "value_cov_minutes_clubseason_min": float(main_.value_cov_minutes.min()),
                "transfer_rows_by_season": {k: int(tr_rows[k]) for k in ["12/13", "13/14", "15/16", "17/18", "23/24"]},
                "valuation_rows_by_year": {str(y): int(n) for y, n in val.date.dt.year.value_counts().sort_index().loc[2021:2025].items()}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_coverage.json").write_text(json.dumps(coverage, indent=1))
    print(panel.groupby(["league"]).size())
    print("incomplete club-seasons:\n", panel[panel.complete == 0][["league", "season", "club", "app_cov"]])
    print(panel.groupby("season")[["value_cov_minutes", "value_cov_players", "value_forward_share"]].mean().round(3))
    print(panel.pivot_table(index="season", columns="league", values="spend", aggfunc="sum").div(1e6).round(0))
    print(panel.pivot_table(index="season", columns="league", values="squad_value", aggfunc="sum").div(1e6).round(0))
    print("zero spend share\n", panel.assign(z=panel.spend == 0).pivot_table(index="season", columns="league", values="z").round(2))
    print("zero spend3 share\n", panel.assign(z=panel.spend3 == 0).pivot_table(index="season", columns="league", values="z").round(2))


if __name__ == "__main__":
    main()
