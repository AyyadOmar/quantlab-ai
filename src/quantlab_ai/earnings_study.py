"""Explicitly exploratory public-snapshot earnings study, with a separate schedule test."""
from __future__ import annotations
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .config import Settings
from .data.loader import MarketDataLoader
from .data.earnings import load_earnings_snapshots
from .features.builder import FeatureBuilder
from .features.earnings import add_surprises, add_schedules
from .features.profiles import feature_columns
from .feature_study import CANDIDATES, Candidate, choose_candidate, evaluate_fold, paired_brier_interval
from .models.classical import ClassicalModelTrainer
from .models.base import combine_fold_predictions, aggregate_classification_metrics, classification_baselines
from .backtesting.engine import BacktestEngine


def run_earnings_study(settings: Settings, start: str, end: str, *, allow_retrospective_snapshot: bool = False) -> dict:
    if not allow_retrospective_snapshot:
        raise ValueError('This public-source study requires --allow-retrospective-snapshot; it is not point-in-time-verified evidence.')
    settings = replace(settings, protocol='earnings_study_v5', use_cached_data=True)
    settings.ensure_directories()
    snapshots, manifest = load_earnings_snapshots(settings.project_root)
    folder = settings.project_root/'data'/'earnings_v5'
    schedules = pd.read_csv(folder/'nvda_schedules.csv')
    for path, checksum in schedules[['source_file','source_sha256']].drop_duplicates().itertuples(index=False, name=None):
        if sha256((settings.project_root/path).read_bytes()).hexdigest() != checksum:
            raise ValueError('Schedule evidence changed since extraction.')
    plan = {'protocol': settings.protocol, 'start': start, 'end_exclusive': end,
            'quality': 'Exploratory: current vendor EPS snapshots are not historically vintage-verified.',
            'sources': manifest, 'schedule_sha256': sha256((folder/'nvda_schedules.csv').read_bytes()).hexdigest(),
            'schedule_scope': 'NVDA only; missing announcements are unknown, not backfilled.',
            'selection': 'Earlier validation Brier then log loss; original six price candidates always available.',
            'surprise_features': 'Vendor percentage surprise clipped to [-1,1], 20-calendar-day decay, capped recency, known flag; next-day use only.',
            'settings': {k:str(v) if isinstance(v,Path) else v for k,v in asdict(settings).items()},
            'packages': {name:version(name) for name in ['pandas','numpy','scikit-learn','xgboost','yfinance','lxml']},
            'defaults_changed': False}
    (settings.backtests_dir/'study_plan.json').write_text(json.dumps(plan,indent=2))
    loader, builder, engine = MarketDataLoader(settings), FeatureBuilder(settings), BacktestEngine(settings)
    market = loader.download('SPY',start,end)
    rows, provenance, choices, coverage, differences = [], {}, {}, {}, {}
    for ticker in ['AAPL','MSFT','NVDA']:
        raw=loader.download(ticker,start,end)
        data=builder.build(raw,ticker,market)
        data=add_surprises(data,snapshots[ticker],allow_retrospective_snapshot=True)
        base_names={candidate.name for candidate in CANDIDATES}
        surprise_candidates=tuple(Candidate(f'{base}_surprise','regularized') for base in ['relative','compact'])
        candidates=CANDIDATES+surprise_candidates
        scopes={'price_only':base_names,'surprise_snapshot':base_names|{c.name for c in surprise_candidates}}
        if ticker=='NVDA':
            data=add_schedules(data,schedules)
            schedule_candidates=tuple(Candidate(f'{base}_schedule','regularized') for base in ['relative','compact'])
            combined_candidates=tuple(Candidate(f'{base}_events','regularized') for base in ['relative','compact'])
            candidates+=schedule_candidates+combined_candidates
            scopes['dated_schedule']=base_names|{c.name for c in schedule_candidates}
            scopes['snapshot_and_schedule']={c.name for c in candidates}
        columns=sorted({col for c in candidates for col in feature_columns(c.feature_set)})
        if not np.isfinite(data[columns].to_numpy()).all():
            raise ValueError('Incomplete features would invalidate comparison dates.')
        data.to_csv(settings.processed_data_dir/f'{ticker.lower()}_features.csv',index=False)
        provenance[ticker]={'price_sha256':sha256(raw.to_csv(index=False).encode()).hexdigest(),
                            'market_sha256':sha256(market.to_csv(index=False).encode()).hexdigest(),
                            'candidates':[asdict(c) for c in candidates], 'scopes':{k:sorted(v) for k,v in scopes.items()}}
        period=snapshots[ticker].loc[(snapshots[ticker].report_date>=pd.Timestamp('2017-01-01'))&(snapshots[ticker].report_date<pd.Timestamp(end))]
        coverage[ticker]={'complete_earnings_reports':len(period),'surprise_known_fraction':float(data.earnings_surprise_known.mean()),
                          'schedule_available':ticker=='NVDA'}
        if ticker=='NVDA':
            coverage[ticker]['schedule_records']=len(schedules)
            coverage[ticker]['schedule_known_fraction']=float(data.earnings_schedule_known.mean())
            coverage[ticker]['missing_schedule_report_dates']=[str(d.date()) for d in period.report_date if str(d.date()) not in set(schedules.scheduled_date)]
        for model in ['logistic_regression','xgboost']:
            trainer=ClassicalModelTrainer(settings,model)
            groups={name:[] for name in scopes}
            folds=[]
            for fold,(train,val,test) in enumerate(trainer.walk_forward_splits(data),1):
                predictions,report,artifact=evaluate_fold(settings,data,train,val,test,model,candidates)
                report['fold']=fold
                report['scopes']={}
                for name,allowed in scopes.items():
                    selected=choose_candidate([score for score in report['validation_scores'] if score['candidate'] in allowed])
                    report['scopes'][name]=selected
                    pred=predictions[selected].copy()
                    pred['fold']=fold
                    groups[name].append(pred)
                folds.append(report)
            artifact['data_quality']='Exploratory current vendor snapshot; not a production-approved model.'
            path=trainer.registry.save_joblib(f'{model}_{ticker.lower()}_exploratory_events',artifact)
            choices[f'{ticker}_{model}']={'folds':folds,'artifact':path}
            combined={name:combine_fold_predictions(frames) for name,frames in groups.items()}
            for scope,pred in combined.items():
                metrics=aggregate_classification_metrics(pred)
                baselines=classification_baselines(pred)
                result=engine.run_with_threshold(pred,model,ticker,None,False)
                row={'ticker':ticker,'model':model,'scope':scope,**metrics,
                     'always_up_accuracy':baselines['always_up']['accuracy'],
                     'prior_brier':baselines['training_prevalence']['brier_score'],
                     'net_return':result.metrics['total_return'],'max_drawdown':result.metrics['max_drawdown'],
                     'trades':result.metrics['trade_count'],'execution_start':str(pred.execution_date.iloc[0]),
                     'execution_end':str(pred.execution_date.iloc[-1])}
                rows.append(row)
                pred.to_csv(settings.backtests_dir/f'{ticker.lower()}_{model}_{scope}_predictions.csv',index=False)
                if scope!='price_only':
                    base=combined['price_only']
                    pair=pred[['execution_date']].copy()
                    pair['loss_difference']=(pred.prob_up-pred.target)**2-(base.prob_up-base.target)**2
                    differences.setdefault(f'{model}/{scope}',[]).append(pair)
            print(f'Completed earnings study {ticker}/{model}',flush=True)
    report={'plan':plan,'data':provenance,'coverage':coverage,'results':rows,'selection':choices,
            'uncertainty':{key:paired_brier_interval(values,settings.random_state) for key,values in differences.items()}}
    (settings.backtests_dir/'earnings_study.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(rows).to_csv(settings.backtests_dir/'earnings_study.csv',index=False)
    write_report(settings,report)
    return report


def write_report(settings: Settings,report: dict) -> None:
    data=pd.DataFrame(report['results'])
    lines=['# Public-source earnings study','',
           '**Exploratory results, not historically vintage-verified evidence.** EPS estimates, actuals and surprise percentages were downloaded as current Yahoo snapshots. They may contain revisions or adjustments. No defaults or demo results were changed.','',
           'Surprises are available no earlier than the calendar day after the reported date. Upcoming-call features are derived only from dated NVIDIA newsroom announcements, available from the day after publication. A reported earnings date is never used to reconstruct advance notice.','',
           '## Surprise snapshots: all three companies','',
           '| Model | Inputs | Mean accuracy | Mean AUC | Mean Brier ↓ | Mean ticker net return |',
           '|---|---|---:|---:|---:|---:|']
    for (model,scope),group in data.loc[data.scope.isin(['price_only','surprise_snapshot'])].groupby(['model','scope'],sort=False):
        lines.append(f'| {model} | {scope} | {group.accuracy.mean():.2%} | {group.roc_auc.mean():.3f} | {group.brier_score.mean():.4f} | {group.net_return.mean():.2%} |')
    lines+=['','Mean ticker returns are cumulative and are not a portfolio simulation. Same 5 bps fee and 2 bps slippage per side as earlier studies.','',
            '## NVIDIA: dated schedule and combined experiments','',
            '| Model | Inputs | Accuracy | Brier ↓ | Net return | Trades |','|---|---|---:|---:|---:|---:|']
    for row in data.loc[data.ticker.eq('NVDA')].itertuples():
        lines.append(f'| {row.model} | {row.scope} | {row.accuracy:.2%} | {row.brier_score:.4f} | {row.net_return:.2%} | {row.trades} |')
    lines+=['','Schedule-only selection is separate from the snapshot experiments and has no surprise columns. Schedule dates are earnings-call dates, not exact earnings-release timestamps. Public archive coverage is incomplete. Apple/Microsoft schedule models were not evaluated.','',
            '## Individual surprise results','',
            '| Ticker | Model | Price accuracy | Surprise accuracy | Always up | Price Brier | Surprise Brier | Training-prior Brier |','|---|---|---:|---:|---:|---:|---:|---:|']
    for (ticker,model),group in data.groupby(['ticker','model'],sort=False):
        base=group.loc[group.scope.eq('price_only')].iloc[0]
        new=group.loc[group.scope.eq('surprise_snapshot')].iloc[0]
        lines.append(f'| {ticker} | {model} | {base.accuracy:.2%} | {new.accuracy:.2%} | {new.always_up_accuracy:.2%} | {base.brier_score:.4f} | {new.brier_score:.4f} | {new.prior_brier:.4f} |')
    lines+=['','## Coverage','']
    for ticker,coverage in report['coverage'].items():
        lines.append(f"- {ticker}: {coverage['complete_earnings_reports']} complete estimate/actual/surprise rows from 2017 through the study cutoff; surprise history available on {coverage['surprise_known_fraction']:.1%} of feature dates.")
        if coverage['schedule_available']:
            lines.append(f"- NVIDIA: {coverage['schedule_records']} advance notices; scheduled future call known on {coverage['schedule_known_fraction']:.1%} of feature dates. Missing matches: {', '.join(coverage['missing_schedule_report_dates'])}.")
    lines+=['','## Interpretation limits','',
            'Candidates and thresholds were selected using earlier validation only. Price-only controls reproduce the prior feature study on the same dates. This does not eliminate repeated-research bias, current-snapshot revisions, schedule-archive gaps, or the small number of independent quarterly events. Daily rows repeatedly reuse quarterly values.','',
            'Any improvement in this report needs confirmation on fresh observations recorded when received. The code rejects unverified surprise snapshots by default; this run explicitly opted into retrospective exploration. No paid data was used.','',
            'Sources: [Yahoo public earnings tables](https://finance.yahoo.com/calendar/earnings/), [NVIDIA newsroom archive](https://nvidianews.nvidia.com/news). Per-event URLs and source hashes are saved with the downloaded data. Full validation scores and timing assumptions are in the JSON report.']
    (settings.backtests_dir/'earnings_study.md').write_text('\n'.join(lines)+'\n')
