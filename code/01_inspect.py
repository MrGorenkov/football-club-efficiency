"""Coverage checks for the six leagues (GB1, ES1, IT1, L1, FR1, RU1), seasons 2012-2025. Output: data/inspect_output.txt."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1", "RU1"]


def main():
    out = []
    g = pd.read_csv(RAW / "games.csv.gz")
    g = g[g.competition_id.isin(LEAGUES)]
    out.append("games per league x season\n" + g.pivot_table(index="season", columns="competition_id", values="game_id", aggfunc="count").to_string())
    g["has_score"] = g.home_club_goals.notna()
    out.append("share of games with score\n" + g.pivot_table(index="season", columns="competition_id", values="has_score", aggfunc="mean").round(3).to_string())
    clubs = pd.concat([g[["competition_id", "season", "home_club_id"]].rename(columns={"home_club_id": "c"}),
                       g[["competition_id", "season", "away_club_id"]].rename(columns={"away_club_id": "c"})]).drop_duplicates()
    out.append("clubs per league x season\n" + clubs.pivot_table(index="season", columns="competition_id", values="c", aggfunc="count").to_string())
    a = pd.read_csv(RAW / "appearances.csv.gz", usecols=["game_id", "player_id", "competition_id", "minutes_played"])
    a = a[a.competition_id.isin(LEAGUES)].merge(g[["game_id", "season"]], on="game_id")
    cov = a.groupby(["competition_id", "season"]).agg(games=("game_id", "nunique"), rows=("player_id", "size"))
    cov["rows_per_game"] = (cov.rows / cov.games).round(1)
    out.append("appearance coverage\n" + cov.unstack(0).to_string())
    v = pd.read_csv(RAW / "player_valuations.csv.gz", parse_dates=["date"])
    out.append("valuations: " + str(len(v)) + " rows, " + str(v.date.min()) + " .. " + str(v.date.max()))
    t = pd.read_csv(RAW / "transfers.csv.gz")
    t["fee_pos"] = t.transfer_fee > 0
    out.append("transfers: rows, fee>0 by transfer_season (last 16)\n" + t.groupby("transfer_season").agg(rows=("player_id", "size"), paid=("fee_pos", "sum")).tail(16).to_string())
    txt = "\n\n".join(out)
    (ROOT / "data" / "inspect_output.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
