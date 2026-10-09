from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from .common import STOCKS, digest, load_json, read_table, root, smoke
from .models import check_frozen
from .portfolio import INITIAL_CAPITAL, run_portfolios


def plot_eda():
    prices = read_table('history_prices')
    prices = prices[prices.date < '2024-01-01']
    close = prices[prices.ticker.isin(STOCKS)].pivot(index='date', columns='ticker', values='close')
    returns = close.pct_change(fill_method=None)
    coverage = prices.groupby('ticker').agg(first=('date', 'min'), last=('date', 'max'), sessions=('date', 'size'))
    coverage['missing_close'] = prices.groupby('ticker').close.apply(lambda s: s.isna().sum())
    display(coverage)
    display(returns.describe().T)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    if smoke():
        fig.suptitle('SYNTHETIC SMOKE TEST — NOT FINANCIAL RESULTS', fontweight='bold')
    close.plot(ax=axes[0, 0], title='Development-period adjusted closing prices')
    axes[0, 0].set_ylabel('Adjusted price (USD)')
    for ticker in STOCKS:
        axes[0, 1].hist(returns[ticker].dropna(), bins=80, alpha=.35, density=True, label=ticker)
    axes[0, 1].set(title='Distribution of daily returns', xlabel='Daily return', ylabel='Density')
    axes[0, 1].legend(fontsize=8)
    sns.heatmap(returns.corr(), annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title('Daily-return correlations')
    (returns.std(ddof=1)*np.sqrt(252)).sort_values().plot.bar(ax=axes[1, 1], title='Annualized daily-return volatility')
    axes[1, 1].set_ylabel('Annualized volatility')
    fig.tight_layout()
    plt.show()


def plot_importance():
    importance = pd.read_csv(root() / 'reports' / 'feature_importance.csv')
    pooled = importance[importance.method == 'validation_permutation']
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    if smoke():
        fig.suptitle('SYNTHETIC SMOKE TEST — NOT FINANCIAL RESULTS', fontweight='bold')
    for i, algorithm in enumerate(['rf', 'xgb']):
        for j, task in enumerate(['regression', 'classification']):
            data = pooled[(pooled.model == f'{algorithm}_pooled') & (pooled.task == task)]
            top = data.nlargest(10, 'importance').sort_values('importance')
            axes[i, j].barh(top.feature, top.importance)
            axes[i, j].set(title=f'{algorithm.upper()} — {task}',
                           xlabel='Increase in RMSE' if task == 'regression' else 'Decrease in ROC-AUC')
    fig.tight_layout()
    plt.show()
    individual = importance[importance.ticker != 'all']
    tops = individual.sort_values('importance', ascending=False).groupby(['model', 'ticker', 'task']).head(3)
    display(tops.sort_values(['task', 'model', 'ticker']))
    return importance


def feature_discussion():
    importance = pd.read_csv(root() / 'reports' / 'feature_importance.csv')
    ablation = pd.read_csv(root() / 'reports' / 'feature_ablation.csv')
    lines = ['### Validation evidence: features and model complexity']
    pooled = importance[importance.method == 'validation_permutation']
    for task in ['regression', 'classification']:
        tops = {}
        for algorithm in ['rf', 'xgb']:
            rows = pooled[(pooled.model == f'{algorithm}_pooled') & (pooled.task == task)]
            ranked = rows.sort_values('importance', ascending=False)
            tops[algorithm] = set(ranked.head(10).feature)
            positive = ranked[ranked.importance > 0].head(3).feature.tolist()
            lines.append(f'- **{algorithm.upper()} {task}:** leading positive validation importances: '
                         + (', '.join(positive) if positive else 'none') + '.')
        lines.append(f'- The two {task} models share **{len(tops["rf"] & tops["xgb"])} of their top ten** features. '
                     'Shared rank does not establish causality; weak or tied importances are unstable.')
    for (algorithm, task), rows in ablation.groupby(['algorithm', 'task']):
        values = rows.set_index('features').score
        full, momentum = values['full'], values['momentum_only']
        improved = full < momentum if task == 'regression' else full > momentum
        lines.append(f'- **{algorithm.upper()} {task} ablation:** full {rows.metric.iloc[0]} = {full:.5f}; '
                     f'momentum-only = {momentum:.5f}. Adding features {"improved" if improved else "did not improve"} '
                     'this validation metric at the fixed settings.')
    lines.append('\nThe following table counts how many stock-specific models place each feature in their top three. '
                 'A count of eight indicates broad agreement across this universe; lower counts indicate variation. '
                 'These are training impurity/gain rankings and may favor some predictors.')
    display(Markdown('\n\n'.join(lines)))
    individual = importance[importance.ticker != 'all']
    top = individual.sort_values('importance', ascending=False).groupby(['model', 'ticker', 'task']).head(3)
    counts = top.groupby(['model', 'task', 'feature']).ticker.nunique().rename('stocks_in_top_three').reset_index()
    display(counts.sort_values(['task', 'model', 'stocks_in_top_three'], ascending=[True, True, False]))


def report_portfolios(experiment):
    frozen = check_frozen()
    predictions = pd.read_parquet(root() / 'predictions' / f'{experiment}.parquet')
    bundle = load_json(root() / 'models' / f'{experiment}.json')
    if bundle['freeze_id'] != frozen['freeze_id'] or set(predictions.bundle_id) != {bundle['bundle_id']}:
        raise ValueError('Predictions do not match frozen model bundle')
    prices = read_table('history_prices')
    if experiment == 'competition':
        metadata = load_json(root() / 'data' / 'metadata.json')
        if digest(root() / 'data' / 'holdout_prices.parquet') != metadata['files']['holdout_prices.parquet']:
            raise ValueError('Holdout execution prices changed')
        prices = pd.concat([prices, read_table('holdout_prices')], ignore_index=True)
        start, end = '2026-01-01', str(prices.date.max().date())
        title = load_json(root() / 'competition_evaluation.json')['label']
    elif experiment == 'assignment':
        start, end, title = '2024-01-01', '2025-12-31', 'Assignment test: 2024–2025'
    else:
        raise ValueError('Expected assignment or competition')
    if smoke():
        title = 'SYNTHETIC TEST — ' + title
    metrics, curves, trades = run_portfolios(prices, predictions, start, end)
    metrics['competition_entry'] = metrics.strategy == frozen['selected_entry']['strategy']
    folder = root() / 'reports'
    metrics.to_csv(folder / f'{experiment}_portfolios.csv', index=False)
    curves.to_parquet(folder / f'{experiment}_daily.parquet', index=False)
    trades.to_parquet(folder / f'{experiment}_trades.parquet', index=False)
    display(Markdown(f'### {title}\nFrozen competition entry: **{frozen["selected_entry"]["strategy"]}**.'))
    display(metrics.set_index('strategy'))
    fig, ax = plt.subplots(figsize=(14, 7))
    for strategy, group in curves.groupby('strategy'):
        highlighted = strategy in ['SPY', 'equal_weight', frozen['selected_entry']['strategy']]
        # Show initial capital explicitly, so initial execution costs are visible.
        dates = pd.concat([pd.Series(group.date.min() - pd.Timedelta(days=1)), group.date]).to_numpy()
        values = np.r_[INITIAL_CAPITAL, group.value.to_numpy()]
        ax.plot(dates, values, label=strategy, linewidth=2.5 if highlighted else 1, alpha=1 if highlighted else .55)
    ax.set(title=f'{title}: growth of $100,000 after costs', xlabel='Date', ylabel='Portfolio value (USD)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=.2)
    fig.tight_layout()
    plt.show()
    display(trades.head(12))
    return metrics, curves, trades


def evidence_discussion(experiment):
    """Render result-grounded observations without claiming statistical significance."""
    predictive = pd.read_csv(root() / 'reports' / f'{experiment}_metrics.csv')
    overall = predictive[predictive.ticker == 'ALL'].set_index('model')
    portfolios = pd.read_csv(root() / 'reports' / f'{experiment}_portfolios.csv').set_index('strategy')
    frozen = check_frozen()
    chosen = frozen['selected_entry']['strategy']
    baseline = overall.loc['baseline']
    learned = overall.drop('baseline')
    best_return = learned.rmse.idxmin()
    best_risk = learned.roc_auc.idxmax()
    beats_return = learned.rmse.min() < baseline.rmse
    beats_risk = learned.roc_auc.max() > baseline.roc_auc
    delta = portfolios.loc[chosen, 'total_return'] - portfolios.loc['SPY', 'total_return']
    rows = []
    for algorithm in ['rf', 'xgb']:
        pooled, individual = overall.loc[f'{algorithm}_pooled'], overall.loc[f'{algorithm}_individual']
        rows.append(f'- {algorithm.upper()}: pooled/individual RMSE = {pooled.rmse:.5f}/{individual.rmse:.5f}; '
                    f'ROC-AUC = {pooled.roc_auc:.3f}/{individual.roc_auc:.3f}.')
    display(Markdown(
        f'### Evidence and discussion — {experiment}\n\n'
        f'**Which model performed better?** Lowest return RMSE: **{best_return}** '
        f'({learned.rmse.min():.5f}, versus zero-return baseline {baseline.rmse:.5f}). '
        f'Highest risk ROC-AUC: **{best_risk}** ({learned.roc_auc.max():.3f}). '
        'There may be different winners for returns and risk. Bagging and boosting have different '
        'bias/variance behavior, but these results alone cannot establish the cause of a difference.\n\n'
        f'**Returns versus risk:** at least one return model beats baseline RMSE: **{beats_return}**; '
        f'at least one risk model beats baseline ROC-AUC: **{beats_risk}**. '
        'Compare improvement over each task’s baseline, not the numerical size of RMSE against AUC.\n\n'
        '**Did pooling help?**\n\n' + '\n'.join(rows) + '\n\n'
        f'**Did the frozen entry beat SPY?** Its total return was {portfolios.loc[chosen, "total_return"]:.2%}, '
        f'versus {portfolios.loc["SPY", "total_return"]:.2%} for SPY; the difference was **{delta:+.2%}**. '
        'The entry was chosen on validation, not by selecting this period’s best portfolio.\n\n'
        '**Did better predictions guarantee better investments?** Use the joined table below to compare '
        'each model’s RMSE and ranking correlation with its ranked portfolio return. '
        'RMSE weights all stocks/dates; the strategy trades only three stocks on scheduled dates. '
        'Ranking, execution gaps, concentration, cash, and turnover therefore affect realized performance.\n\n'
        '**What supports generalization?** Predictions come from chronological refits with purged label boundaries, '
        'training-only risk thresholds, and a frozen configuration. These are validity checks, not proof of '
        'persistent outperformance. Overlapping five-day labels are dependent; no significance claim is made.\n\n'
        '**Limitations and improvements:** the retrospectively chosen surviving universe is technology-heavy; '
        'adjusted Yahoo prices are not a point-in-time institutional database; daily opens simplify fills; '
        'costs and cash yield are fixed; the test windows are short. Pre-register future walk-forward experiments, '
        'use point-in-time universes and more realistic execution, and estimate uncertainty with date-block resampling. '
        'These are future experiments, not changes to the frozen 2026 entry.'
    ))
    ranked = portfolios.loc[[f'{name}__ranked' for name in learned.index], ['total_return', 'sharpe', 'turnover']].copy()
    ranked.index = ranked.index.str.replace('__ranked', '', regex=False)
    display(learned[['rmse', 'daily_cross_sectional_spearman']].join(ranked))
