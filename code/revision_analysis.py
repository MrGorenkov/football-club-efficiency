"""Tournament-aware retrospective benchmarking, replacing managerial-efficiency claims.

Main sample: intact league-seasons; every club has >=90% games with at least
11 recorded players and >=95% recorded minutes covered by prior valuations.
The thresholds use resource quality, never match results. All result files are new.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import spearmanr
from dea import farrell_output_1d

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/'results/revision'
LEAGUES = ['GB1','ES1','IT1','L1','FR1','RU1']


def design(g):
    li = g.competition_id.map(dict(zip(LEAGUES,range(6)))).to_numpy()
    x = np.log(g.home_resource.to_numpy()/g.away_resource.to_numpy())
    return li, x


def probabilities(t, li, x):
    z = t[li] + t[-1]*x
    return softmax(np.column_stack([z/2,t[6+li],-z/2]),axis=1)


def fit(g, outcomes=None):
    li,x=design(g)
    y=g.outcome.to_numpy() if outcomes is None else outcomes
    def objective(t):
        z=t[li]+t[-1]*x
        logits=np.column_stack([z/2,t[6+li],-z/2])
        p=softmax(logits,axis=1)
        nll=np.sum(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])
        p[np.arange(len(y)),y]-=1
        dz=(p[:,0]-p[:,2])/2
        grad=np.r_[np.bincount(li,weights=dz,minlength=6),
                   np.bincount(li,weights=p[:,1],minlength=6),np.dot(dz,x)]
        return nll,grad
    r=minimize(objective,np.r_[np.repeat(.5,6),np.repeat(-.4,6),.5],jac=True,
               bounds=[(None,None)]*12+[(0,None)],method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-6})
    if not r.success: raise RuntimeError(r.message)
    return r.x, float(r.fun),float(np.max(np.abs(r.jac)))


def club_stats(g, p):
    hp=3*p[:,0]+p[:,1];ap=3*p[:,2]+p[:,1]
    observed_h=np.choose(g.outcome,[3,1,0]);observed_a=np.choose(g.outcome,[0,1,3])
    base=g[['competition_id','season']].reset_index(drop=True)
    h=base.assign(club_id=g.home_club_id.to_numpy(),points=observed_h,expected=hp,resource=g.home_resource.to_numpy())
    a=base.assign(club_id=g.away_club_id.to_numpy(),points=observed_a,expected=ap,resource=g.away_resource.to_numpy())
    out=pd.concat([h,a]).groupby(['competition_id','season','club_id']).agg(
        points=('points','sum'),expected_points=('expected','sum'),games=('points','size'),resource=('resource','first')).reset_index()
    out['ppg']=out.points/out.games;out['expected_ppg']=out.expected_points/out.games
    out['excess_ppg']=out.ppg-out.expected_ppg
    assert out.expected_ppg.between(0,3).all()
    assert np.allclose(out.groupby(['competition_id','season']).expected_points.sum().to_numpy(),
                       pd.Series(3-p[:,1]).groupby([g.competition_id.to_numpy(),g.season.to_numpy()]).sum().to_numpy())
    return out


def simulate_group(d, probability, parameters, rng, reps):
    clubs=np.sort(np.unique(np.r_[d.home_club_id,d.away_club_id]));n=len(clubs)
    hi=np.searchsorted(clubs,d.home_club_id);ai=np.searchsorted(clubs,d.away_club_id)
    ng=np.bincount(np.r_[hi,ai],minlength=n)
    resource=np.array([d.loc[d.home_club_id.eq(c),'home_resource'].iloc[0] if d.home_club_id.eq(c).any()
                       else d.loc[d.away_club_id.eq(c),'away_resource'].iloc[0] for c in clubs])
    actual=np.bincount(hi,weights=np.choose(d.outcome,[3,1,0]),minlength=n)+np.bincount(ai,weights=np.choose(d.outcome,[0,1,3]),minlength=n)
    obs=1/farrell_output_1d(resource,actual/ng)
    vals=np.empty(reps);scores=np.empty((reps,n)); draw_share=np.empty(reps)
    li,x=design(d)
    for b in range(reps):
        pr=probability if parameters is None else probabilities(parameters[b%len(parameters)],li,x)
        u=rng.random(len(d)); y=np.where(u<pr[:,0],0,np.where(u<pr[:,:2].sum(axis=1),1,2))
        pts=np.bincount(hi,weights=np.choose(y,[3,1,0]),minlength=n)+np.bincount(ai,weights=np.choose(y,[0,1,3]),minlength=n)
        scores[b]=1/farrell_output_1d(resource,pts/ng)
        vals[b]=scores[b].mean();draw_share[b]=np.mean(y==1)
    assert np.isfinite(scores).all()
    row={'league':d.competition_id.iloc[0],'season':int(d.season.iloc[0]),'n_clubs':n,'matches':len(d),
         'observed_mean_dea':obs.mean(),'null_mean_dea':vals.mean(),'null_low':np.quantile(vals,.025),
         'null_high':np.quantile(vals,.975),'null_sd':vals.std(ddof=1),
         'observed_minus_null':obs.mean()-vals.mean(),'outside_envelope':int(obs.mean()<np.quantile(vals,.025) or obs.mean()>np.quantile(vals,.975))}
    # These are simulation quantiles under a fitted reference model, not CIs for latent efficiency.
    cr=pd.DataFrame({'league':row['league'],'season':row['season'],'club_id':clubs,'dea':obs,
                     'null_low':np.quantile(scores,.025,axis=0),'null_high':np.quantile(scores,.975,axis=0)})
    return row,cr,vals


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=199)
    parser.add_argument('--simulations',type=int,default=1000);args=parser.parse_args()
    rng=np.random.default_rng(20261102)
    g=pd.read_csv(ROOT/'data/revision_matches.csv')
    s=pd.read_csv(ROOT/'data/revision_resources.csv')
    old=pd.read_csv(ROOT/'data/panel.csv')
    schedule=pd.concat([g[['competition_id','season','home_club_id']].rename(columns={'home_club_id':'club_id'}),
                        g[['competition_id','season','away_club_id']].rename(columns={'away_club_id':'club_id'})])
    schedule=schedule.groupby(['competition_id','season','club_id']).size().rename('schedule_games').reset_index()
    schedule=schedule.rename(columns={'competition_id':'league'})
    quality=schedule.merge(s,on=['league','season','club_id'],how='left',validate='one_to_one')
    quality['app_cov']=quality.appearance_complete_games.fillna(0)/quality.schedule_games
    quality['good']=(quality.app_cov>=.9)&(quality.value_coverage>=.95)&quality.match_date_xi.gt(0)
    for side in ['home','away']:
        lookup=quality[['league','season','club_id','match_date_xi','good']].rename(columns={
            'league':'competition_id','club_id':side+'_club_id','match_date_xi':side+'_resource','good':side+'_good'})
        g=g.merge(lookup,on=['competition_id','season',side+'_club_id'],how='left',validate='many_to_one')
    g['good']=g.home_good.eq(True)&g.away_good.eq(True)
    keep=g.groupby(['competition_id','season']).good.all().rename('keep')
    g=g.merge(keep,on=['competition_id','season'])
    keep.to_csv(RES/'season_sample.csv');quality[quality.season.between(2013,2023)].to_csv(ROOT/'data/revision_quality.csv',index=False)
    all_g=g.copy();g=g[g.keep].reset_index(drop=True)
    g['outcome']=np.where(g.home_club_goals>g.away_club_goals,0,np.where(g.home_club_goals==g.away_club_goals,1,2))
    t,nll,grad=fit(g); li,x=design(g);pr=probabilities(t,li,x)
    names=['home_'+l for l in LEAGUES]+['draw_'+l for l in LEAGUES]+['log_resource_ratio']
    params=pd.DataFrame({'parameter':names,'estimate':t});params.to_csv(RES/'match_coefficients.csv',index=False)
    scores=club_stats(g,pr);scores.to_csv(RES/'club_benchmarks.csv',index=False)
    # Parameter uncertainty: parametric bootstrap of entire fixture list and full ML re-fit.
    boot=[]
    for b in range(args.bootstrap):
        u=rng.random(len(g));y=np.where(u<pr[:,0],0,np.where(u<pr[:,:2].sum(axis=1),1,2))
        tb,_,_=fit(g,y);boot.append(tb)
        if (b+1)%25==0:print('bootstrap',b+1,flush=True)
    boot=np.array(boot);pd.DataFrame(boot,columns=names).to_csv(RES/'match_parameter_bootstrap.csv',index=False)
    rows=[];cr=[];nulls=[]
    for key,d in g.groupby(['competition_id','season'],sort=True):
        row,c,vals=simulate_group(d,pr[d.index],boot,rng,args.simulations)
        rows.append(row);cr.append(c);nulls.append(vals)
    sim=pd.DataFrame(rows);sim.to_csv(RES/'tournament_null.csv',index=False)
    pd.concat(cr).to_csv(RES/'club_simulation_envelopes.csv',index=False)
    # Equal-weight season summaries of the retrospective gap, no tests of managerial equality.
    sim.groupby('league').agg(seasons=('season','size'),observed=('observed_mean_dea','mean'),
        null=('null_mean_dea','mean'),gap=('observed_minus_null','mean'),outside=('outside_envelope','sum')).to_csv(RES/'league_summary.csv')
    corr=[]
    dea_all=pd.concat(cr)[['league','season','club_id','dea']].rename(columns={'league':'competition_id'})
    sc=scores.merge(dea_all,on=['competition_id','season','club_id'])
    for key,d in sc.groupby(['competition_id','season']):
        corr.append({'league':key[0],'season':key[1],'rho_dea_points':spearmanr(d.dea,d.ppg).statistic,
                     'rho_dea_excess':spearmanr(d.dea,d.excess_ppg).statistic})
    pd.DataFrame(corr).to_csv(RES/'rank_information.csv',index=False)
    # Temporal validation: fitted coefficients see only earlier seasons. Resource is realised,
    # so these are conditional held-out reconstructions, not pre-season forecasts.
    validation=[];pred=[]
    for year in sorted(g.season.unique())[3:]:
        train=g[g.season<year];test=g[g.season==year]
        tt,_,_=fit(train);tl,tx=design(test);p=probabilities(tt,tl,tx)
        tb,_,_=fit(train.assign(home_resource=1.,away_resource=1.))
        base=probabilities(tb,tl,np.zeros(len(test)))
        ll=-np.log(p[np.arange(len(test)),test.outcome]).mean()
        bl=-np.log(base[np.arange(len(test)),test.outcome]).mean()
        validation.append({'season':int(year),'matches':len(test),'model_logloss':ll,'base_logloss':bl,'improvement':bl-ll})
        cs=club_stats(test.reset_index(drop=True),p);pred.append(cs)
    pd.DataFrame(validation).to_csv(RES/'temporal_validation.csv',index=False)
    # No causal or persistence claims from simple lag correlations.
    cp=pd.concat(pred);lag=cp[['competition_id','season','club_id','excess_ppg','ppg']].copy();lag.season+=1
    lag=lag.rename(columns={'excess_ppg':'lag_excess','ppg':'lag_ppg'})
    cp=cp.merge(lag,on=['competition_id','season','club_id'])
    pd.DataFrame([{'n_pairs':len(cp),'rho_excess_lag':spearmanr(cp.excess_ppg,cp.lag_excess).statistic,
                   'rho_ppg_lag':spearmanr(cp.ppg,cp.lag_ppg).statistic}]).to_csv(RES/'lag_descriptive.csv',index=False)
    # Sensitivity to original input on exactly the same schedule and clubs.
    sens=[]
    for key,d in sc.groupby(['competition_id','season']):
        op=old[old.league.eq(key[0])&old.season.eq(key[1])].set_index('club_id')
        theta=1/farrell_output_1d(op.loc[d.club_id,'squad_value'].to_numpy(),d.ppg.to_numpy())
        sens.append({'league':key[0],'season':key[1],'mean_old_resource_dea':theta.mean(),
                     'mean_new_resource_dea':d.dea.mean(),'rank_agreement':spearmanr(theta,d.dea).statistic})
    pd.DataFrame(sens).to_csv(RES/'resource_sensitivity.csv',index=False)
    summary={'seed':20261102,'main_matches':len(g),'main_clubseasons':len(scores),'main_groups':len(sim),
             'excluded_groups':int((~keep).sum()),'beta':float(t[-1]),'negative_loglik':nll,'max_abs_gradient':grad,
             'bootstrap_refits':len(boot),'simulations_per_group':args.simulations,
             'model':'Davidson, league-specific home and draw intercepts, common nonnegative log-resource slope',
             'expected_ppg_min':float(scores.expected_ppg.min()),'expected_ppg_max':float(scores.expected_ppg.max()),
             'outside_null_envelopes':int(sim.outside_envelope.sum()),'mean_observed_dea':float(sim.observed_mean_dea.mean()),
             'mean_null_dea':float(sim.null_mean_dea.mean()),
             'conditional_validation_logloss_gain':float(np.average(pd.DataFrame(validation).improvement,weights=pd.DataFrame(validation).matches)),
             'mean_within_rho_dea_points':float(pd.DataFrame(corr).rho_dea_points.mean()),
             'warning':'Simulation envelopes are conditional model reference distributions, not confidence intervals for managerial efficiency. Realised minute weights and value feedback preclude causal/ex-ante claims.'}
    (RES/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    g[['game_id','competition_id','season','home_club_id','away_club_id','outcome','home_resource','away_resource']].to_csv(ROOT/'data/revision_analysis_matches.csv',index=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
