"""Supplementary checks, prospective lag features, and corrected legacy S2.
Does not alter archived estimates. Run after revision_analysis.py.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize._numdiff import approx_derivative
import revision_analysis as ra
import dea

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/revision'


def main():
    p=pd.read_csv(ROOT/'data/panel.csv');e=pd.read_csv(ROOT/'data/revision_europe.csv')
    p=p.merge(e[['league','season','club_id','europe_corrected']],on=['league','season','club_id'])
    p['europe']=p.europe_corrected
    spec=importlib.util.spec_from_file_location('legacy_sfa',ROOT/'code/05_sfa.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    pp=m.prepare(p[(p.complete==1)&p.promoted.notna()]);X,Z=m.design(pp)
    result=m.sfa.fit(pp.y.values,X.values,Z.values,'bc95')
    new=m.table('S2_europe_corrected',result,list(X.columns),list(Z.columns))
    new.to_csv(RES/'legacy_s2_corrected.csv',index=False)
    old=pd.read_csv(ROOT/'results/sfa_coefficients.csv');old=old[old.model=='S2']
    old.merge(new,on='param',suffixes=('_old','_corrected')).to_csv(RES/'legacy_s2_comparison.csv',index=False)
    (RES/'legacy_s2_diagnostics.json').write_text(json.dumps({'loglik':float(result['loglik']),
        'legacy_convergence_flag':bool(result['converged']),'n':len(pp),
        'interpretation':'Diagnostic on archived resource; original iid likelihood, not validated managerial inference.'},indent=2))

    # Recompute each season's residual with parameters available by that season's end.
    # Full-sample coefficients would leak later outcomes into a lag predictor.
    games=pd.read_csv(ROOT/'data/revision_analysis_matches.csv')
    past=[]
    for year in sorted(games.season.unique()):
        if year < 2014: continue  # each league needs training observations
        tr=games[games.season<=year];te=games[games.season==year].reset_index(drop=True)
        tt,_,_=ra.fit(tr);li,xx=ra.design(te)
        past.append(ra.club_stats(te,ra.probabilities(tt,li,xx)))
    c=pd.concat(past).rename(columns={'competition_id':'league'})
    d=pd.read_csv(RES/'club_simulation_envelopes.csv')
    c=c.merge(d[['league','season','club_id','dea']],on=['league','season','club_id'])
    c['log_resource']=np.log(c.resource)
    c['relative_value']=c.log_resource-c.groupby(['league','season']).log_resource.transform('mean')
    lag=c[['league','season','club_id','ppg','relative_value','excess_ppg','dea']].copy()
    lag.season+=1
    lag=lag.rename(columns={v:'lag_'+v for v in ['ppg','relative_value','excess_ppg','dea']})
    x=c.merge(lag,on=['league','season','club_id'])
    base=['lag_ppg','lag_relative_value']
    predictions=[]
    # Fixed ridge penalty, no validation-set tuning, all transformations use training data.
    for year in sorted(x.season.unique())[3:]:
        train=x[x.season<year];test=x[x.season==year]
        for name,cols in [('lag_points',base),('plus_dea',base+['lag_dea']),('plus_excess',base+['lag_excess_ppg'])]:
            mu=train[cols].mean();sd=train[cols].std().replace(0,1)
            def matrix(d):
                lg=np.column_stack([d.league.eq(k).astype(float) for k in ra.LEAGUES[1:]])
                return np.column_stack([np.ones(len(d)),(d[cols]-mu)/sd,lg])
            a=matrix(train);b=matrix(test);pen=np.eye(a.shape[1]);pen[0,0]=0
            coef=np.linalg.solve(a.T@a+pen,a.T@train.ppg)
            pred=np.clip(b@coef,0,3)
            predictions.extend([{'season':int(year),'league':lg,'club_id':int(cid),'model':name,
                'actual':float(y),'prediction':float(yhat),'error':float(yhat-y)}
                for lg,cid,y,yhat in zip(test.league,test.club_id,test.ppg,pred)])
    pr=pd.DataFrame(predictions);pr.to_csv(ROOT/'data/revision_next_season_predictions.csv',index=False)
    rows=[]
    for name,d in pr.groupby('model'):
        rows.append({'model':name,'n':len(d),'rmse':np.sqrt(np.mean(d.error**2)),
                     'mae':np.mean(np.abs(d.error)),'first_season':d.season.min(),'last_season':d.season.max()})
    pd.DataFrame(rows).to_csv(RES/'next_season_validation.csv',index=False)
    pr.groupby(['season','model']).error.agg(rmse=lambda x:np.sqrt(np.mean(x**2)),n='size').to_csv(RES/'next_season_by_year.csv')

    # Coverage thresholds are sensitivity analyses, not outcome-driven selection rules.
    q=pd.read_csv(ROOT/'data/revision_quality.csv')
    all_g=pd.read_csv(ROOT/'data/revision_matches.csv')
    sensitivity=[]
    for appearance_cut in [.5,.8,.9,.95]:
        for value_cut in [.95,.99]:
            lookup=q.set_index(['league','season','club_id'])
            rows=[]
            for side in ['home','away']:
                keys=pd.MultiIndex.from_arrays([all_g.competition_id,all_g.season,all_g[side+'_club_id']])
                z=lookup.reindex(keys)
                rows.append((z.app_cov.ge(appearance_cut)&z.value_coverage.ge(value_cut)).to_numpy())
                all_g[side+'_resource']=z.match_date_xi.to_numpy()
            valid=pd.Series(rows[0]&rows[1]).groupby([all_g.competition_id,all_g.season]).transform('all')
            gg=all_g[valid.to_numpy()].copy().reset_index(drop=True)
            gg['outcome']=np.where(gg.home_club_goals>gg.away_club_goals,0,np.where(gg.home_club_goals==gg.away_club_goals,1,2))
            tt,ll,_=ra.fit(gg)
            sensitivity.append({'appearance_cut':appearance_cut,'value_cut':value_cut,'matches':len(gg),
                                'groups':gg.groupby(['competition_id','season']).ngroups,'beta':tt[-1],'logloss':ll/len(gg)})
    pd.DataFrame(sensitivity).to_csv(RES/'coverage_sensitivity.csv',index=False)
    # Validate 1D hull against LP, model monotonicity, chronology and fixture conservation.
    rng=np.random.default_rng(76);xx=np.exp(rng.normal(size=20));yy=rng.uniform(.3,2.5,size=20)
    fast=dea.farrell_output_1d(xx,yy);slow=dea.farrell_output(xx[:,None],yy[:,None])
    err=float(np.max(np.abs(fast-slow)));assert err<1e-8
    t=pd.read_csv(RES/'match_coefficients.csv').estimate.to_numpy()
    lr=np.linspace(-10,10,1001)
    h=ra.probabilities(t,np.zeros(len(lr),dtype=int),lr)
    expected=3*h[:,0]+h[:,1]
    assert np.min(np.diff(expected))>=-1e-12
    g=pd.read_csv(ROOT/'data/revision_analysis_matches.csv')
    assert not g.game_id.duplicated().any()
    assert np.isfinite(g[['home_resource','away_resource']]).all().all()
    assert g[['home_resource','away_resource']].gt(0).all().all()
    report={'dea_closed_form_max_error_vs_lp':err,'expected_ppg_monotonic':True,
        'unique_matches':len(g),'next_season_predictors':'strictly lagged, trained only on earlier target seasons',
        'selection_warning':'Next-season sample conditions on remaining in the top tier and resource coverage.',
        'no_causal_interpretation':True}
    (RES/'verification.json').write_text(json.dumps(report,indent=2))
    print(pd.read_csv(RES/'next_season_validation.csv').to_string(index=False));print(report)


if __name__=='__main__':main()
