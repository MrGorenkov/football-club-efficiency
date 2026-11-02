"""Match-date resources; no future valuations. Does not overwrite the old panel.

Local CSV snapshot can be supplied with --source; otherwise use data/raw.
Full hashes of input tables are recorded. Derived player records stay in data/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results/revision'
LEAGUES = ['GB1', 'ES1', 'IT1', 'L1', 'FR1', 'RU1']
UEFA = ['CL', 'CLQ', 'EL', 'ELQ', 'UCOL', 'ECLQ', 'USC']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'data/raw')
    args = parser.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    raw = args.source.resolve()
    def locate(n):
        for suffix in ['.csv', '.csv.gz']:
            p = raw / (n + suffix)
            if p.exists(): return p
        raise FileNotFoundError(n)
    manifest = {}
    for n in ['games', 'competitions', 'appearances', 'player_valuations']:
        f = locate(n)
        manifest[n] = {'file': f.name, 'bytes': f.stat().st_size,
                       'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
    (RES / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    g = pd.read_csv(locate('games'), parse_dates=['date'])
    comp = pd.read_csv(locate('competitions'))
    comp[comp.competition_id.isin(UEFA)][['competition_id', 'name', 'type']].to_csv(RES/'uefa_competitions.csv', index=False)
    matches = g[g.competition_id.isin(LEAGUES) & g.season.between(2013, 2023)
                & g.home_club_goals.notna() & g.away_club_goals.notna()].copy()
    assert not matches.game_id.duplicated().any()
    a = pd.read_csv(locate('appearances'), usecols=['game_id','player_id','player_club_id','minutes_played'])
    a = a[a.minutes_played.gt(0)].merge(matches[['game_id','date','season','competition_id']], on='game_id', validate='many_to_one')
    assert not a.duplicated(['game_id','player_id']).any()
    v = pd.read_csv(locate('player_valuations'), usecols=['player_id','date','market_value_in_eur'], parse_dates=['date'])
    v = v[v.market_value_in_eur.gt(0)].rename(columns={'date':'value_date'})
    # Duplicate same-day valuations must be identical, not arbitrarily selected.
    dup = v.groupby(['player_id','value_date']).market_value_in_eur.nunique()
    assert dup.max() == 1, 'conflicting duplicate valuations'
    v = v.drop_duplicates(['player_id','value_date']).sort_values('value_date')
    a = pd.merge_asof(a.sort_values('date'), v, left_on='date', right_on='value_date', by='player_id',
                      direction='backward', allow_exact_matches=False, tolerance=pd.Timedelta(days=365))
    assert (a.loc[a.value_date.notna(),'value_date'] < a.loc[a.value_date.notna(),'date']).all()
    a['valued_minutes'] = a.minutes_played.where(a.market_value_in_eur.notna(), 0)
    a['exposure'] = (a.minutes_played * a.market_value_in_eur).fillna(0)
    cm = a.groupby(['competition_id','season','game_id','player_club_id']).agg(
        recorded_minutes=('minutes_played','sum'), valued_minutes=('valued_minutes','sum'),
        exposure=('exposure','sum'), players=('player_id','size')).reset_index()
    cm['value_xi'] = 11 * cm.exposure / cm.valued_minutes
    cm['coverage'] = cm.valued_minutes / cm.recorded_minutes
    cm['usable'] = (cm.players >= 11) & (cm.coverage >= .95) & cm.value_xi.gt(0)
    keys = ['competition_id','season','game_id']
    h = cm.rename(columns={'player_club_id':'home_club_id'})
    aw = cm.rename(columns={'player_club_id':'away_club_id'})
    matches = matches.merge(h[keys+['home_club_id','value_xi','usable','coverage']], on=keys+['home_club_id'], how='left')
    matches = matches.rename(columns={'value_xi':'home_value','usable':'home_usable','coverage':'home_coverage'})
    matches = matches.merge(aw[keys+['away_club_id','value_xi','usable','coverage']], on=keys+['away_club_id'], how='left')
    matches = matches.rename(columns={'value_xi':'away_value','usable':'away_usable','coverage':'away_coverage'})
    matches['usable'] = matches.home_usable.eq(True) & matches.away_usable.eq(True)
    flow = matches.groupby(['competition_id','season']).agg(matches=('game_id','size'), usable=('usable','sum'))
    flow['complete_group'] = flow.matches.eq(flow.usable)
    flow.to_csv(RES/'match_coverage.csv')
    # An intact schedule is required for tournament simulation, never drop individual games.
    matches = matches.merge(flow.complete_group, on=['competition_id','season'])
    matches.to_csv(ROOT/'data/revision_matches.csv', index=False)
    season_resource = cm.groupby(['competition_id','season','player_club_id']).agg(
        exposure=('exposure','sum'), valued_minutes=('valued_minutes','sum'),
        recorded_minutes=('recorded_minutes','sum'), usable_games=('usable','sum'), recorded_games=('game_id','size'),
        appearance_complete_games=('players',lambda x:int((x>=11).sum()))).reset_index()
    season_resource['match_date_xi'] = 11 * season_resource.exposure / season_resource.valued_minutes
    season_resource['value_coverage'] = season_resource.valued_minutes / season_resource.recorded_minutes
    season_resource = season_resource.rename(columns={'competition_id':'league','player_club_id':'club_id'})
    season_resource.to_csv(ROOT/'data/revision_resources.csv', index=False)
    old = pd.read_csv(ROOT/'data/panel.csv')
    old = old[old.season.between(2013,2023) & old.complete.eq(1)].copy()
    eu = g[g.competition_id.isin(UEFA)]
    ids = set(zip(eu.season,eu.home_club_id)) | set(zip(eu.season,eu.away_club_id))
    old['europe_corrected'] = [int((s,c) in ids) for s,c in zip(old.season,old.club_id)]
    old[['league','season','club_id','europe','europe_corrected']].to_csv(ROOT/'data/revision_europe.csv', index=False)
    old['changed'] = old.europe.ne(old.europe_corrected)
    old.groupby('league').agg(n=('club_id','size'),old_share=('europe','mean'),corrected_share=('europe_corrected','mean'),
                            changed=('changed','sum')).to_csv(RES/'europe_correction.csv')
    summary = {'resource_definition':'11 * sum(pre-match value * played minutes) / valued minutes; strictly prior valuations, at most 365 days old',
               'resource_status':'realised retrospective exposure, not preseason endowment or causal spending',
               'all_matches':len(matches),'usable_matches':int(matches.usable.sum()),
               'all_groups':len(flow),'complete_groups':int(flow.complete_group.sum()),
               'complete_group_matches':int(matches.loc[matches.complete_group,'game_id'].size),
               'europe_changed_clubseasons':int(old.changed.sum()),
               'same_day_values_excluded':True,'minimum_coverage':.95,'minimum_players':11,
               'maximum_valuation_age_days':365}
    (RES/'panel_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
