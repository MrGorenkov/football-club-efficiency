"""Reproduce the revised paper's figures from aggregate results."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/revision'
FIG=ROOT/'figures/revision'
LEAGUE={'GB1':'Англия','ES1':'Испания','IT1':'Италия','L1':'Германия','FR1':'Франция','RU1':'РПЛ'}
ORDER=list(LEAGUE)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    s=pd.read_csv(RES/'tournament_null.csv')
    fig,axs=plt.subplots(2,3,figsize=(10,5.8),sharey=True)
    for ax,lg in zip(axs.flat,ORDER):
        d=s[s.league.eq(lg)].sort_values('season')
        ax.fill_between(d.season,d.null_low,d.null_high,color='.87',label='Модельный диапазон')
        ax.plot(d.season,d.null_mean_dea,color='.4',linestyle='--',label='Среднее модели')
        ax.plot(d.season,d.observed_mean_dea,'o-',color='black',markersize=3,label='Наблюдаемое')
        ax.set_title(LEAGUE[lg]);ax.set_ylim(.65,.98);ax.set_xticks([2013,2018,2023]);ax.set_xlabel('Год начала сезона')
    axs[0,0].set_ylabel('Средняя θ');axs[1,0].set_ylabel('Средняя θ')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.05,1,1))
    for ext in ['png','pdf']:fig.savefig(FIG/('fig1_tournament.'+ext),dpi=300,bbox_inches='tight', **({'metadata': {'CreationDate': None, 'ModDate': None}} if ext=='pdf' else {}))
    plt.close(fig)
    c=pd.read_csv(RES/'club_benchmarks.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.7))
    ax.scatter(c.expected_ppg,c.ppg,s=10,c='.35',alpha=.35,edgecolors='none')
    ax.plot([0,3],[0,3],'k--',linewidth=1)
    ax.set(xlabel='Ожидаемые очки за матч',ylabel='Фактические очки за матч',xlim=(0,3),ylim=(0,3))
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(FIG/('fig2_expected.'+ext),dpi=300,bbox_inches='tight', **({'metadata': {'CreationDate': None, 'ModDate': None}} if ext=='pdf' else {}))
    plt.close(fig)


if __name__=='__main__': main()
