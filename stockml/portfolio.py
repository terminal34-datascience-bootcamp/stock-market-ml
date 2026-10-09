from __future__ import annotations

import numpy as np
import pandas as pd

from .common import STOCKS

ASSETS = STOCKS + ['SPY']
INITIAL_CAPITAL = 100_000.0
COST_RATE = 0.001
RULES = {
    'initial_capital': INITIAL_CAPITAL, 'cost_rate': COST_RATE, 'rebalance_sessions': 5,
    'risk_cutoff': 0.5, 'slots': 3, 'cash_interest': 0, 'risk_free_rate': 0,
    'execution': 'next session adjusted open', 'tie_break': 'ticker ascending',
    'terminal_value': 'mark to final adjusted close, no liquidation',
    'selection': 'validation net Sharpe descending, turnover ascending, strategy name ascending',
}


def weights_for(signals, strategy):
    weights = pd.Series(0.0, index=ASSETS)
    if strategy == 'equal_weight':
        weights.loc[STOCKS] = 1 / len(STOCKS)
    elif strategy == 'spy':
        weights.loc['SPY'] = 1.0
    else:
        if signals is None or signals.ticker.duplicated().any() or set(signals.ticker) != set(STOCKS):
            raise ValueError('Each signal date must contain exactly the eight selected stocks')
        if not np.isfinite(signals[['predicted_return', 'high_risk_probability']]).all().all():
            raise ValueError('Invalid portfolio signals')
        if not signals.high_risk_probability.between(0, 1).all():
            raise ValueError('Risk probability outside [0,1]')
        eligible = signals if strategy == 'ranked' else signals[signals.high_risk_probability < .5]
        if strategy not in ['ranked', 'risk_aware']:
            raise ValueError(f'Unknown strategy: {strategy}')
        picks = eligible.sort_values(['predicted_return', 'ticker'], ascending=[False, True]).head(3)
        weights.loc[picks.ticker] = 1 / 3
    return weights


def rebalance(holdings, cash, opening, weights, cost_rate):
    """Solve post-cost NAV + costs = pre-trade NAV, then buy affordable target units."""
    if not 0 <= cost_rate < 1 or (weights < 0).any() or weights.sum() > 1 + 1e-12:
        raise ValueError('Invalid costs or long-only weights')
    before = holdings * opening
    nav = float(cash + before.sum())
    low, high = 0.0, nav
    for _ in range(70):
        after_nav = (low + high) / 2
        equation = after_nav + cost_rate * (weights * after_nav - before).abs().sum() - nav
        if equation > 0:
            high = after_nav
        else:
            low = after_nav
    target = weights * ((low + high) / 2)
    change = target - before
    fees = change.abs() * cost_rate
    new_cash = float(cash - change.sum() - fees.sum())
    if new_cash < -1e-6:
        raise AssertionError('Rebalance borrowed cash')
    return target / opening, max(0.0, new_cash), change, fees, nav


def simulate(prices, signals, strategy, start, end, cost_rate=COST_RATE):
    """Signals are keyed by close date; actual positions are valued daily."""
    market = prices[prices.ticker.isin(ASSETS)].copy()
    if market.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate execution prices')
    opens = market.pivot(index='date', columns='ticker', values='open').reindex(columns=ASSETS).sort_index()
    closes = market.pivot(index='date', columns='ticker', values='close').reindex(columns=ASSETS).sort_index()
    if opens.isna().any().any() or closes.isna().any().any() or (opens <= 0).any().any() or (closes <= 0).any().any():
        raise ValueError('Incomplete or invalid execution calendar')
    dates = opens.index[(opens.index >= pd.Timestamp(start)) & (opens.index <= pd.Timestamp(end))]
    if len(dates) < 2 or opens.index.get_loc(dates[0]) == 0:
        raise ValueError('Need an evaluation window and its preceding signal session')
    grouped = {} if signals is None else {date: group for date, group in signals.groupby('date')}
    holdings = pd.Series(0.0, index=ASSETS)
    cash = INITIAL_CAPITAL
    daily, ledger = [], []
    for i, date in enumerate(dates):
        fee, turnover = 0.0, 0.0
        if i == 0 or (strategy != 'spy' and i % 5 == 0):
            signal_date = opens.index[opens.index.get_loc(date) - 1]
            assert signal_date < date
            weights = weights_for(grouped.get(signal_date), strategy)
            old = holdings.copy()
            holdings, cash, change, fees, nav = rebalance(holdings, cash, opens.loc[date], weights, cost_rate)
            fee = float(fees.sum())
            turnover = float(change.abs().sum() / nav)
            for ticker in ASSETS:
                if abs(change[ticker]) > 1e-8:
                    ledger.append({
                        'signal_date': signal_date, 'execution_date': date, 'ticker': ticker,
                        'units_before': old[ticker], 'units_after': holdings[ticker],
                        'units_traded': holdings[ticker] - old[ticker],
                        'execution_price': opens.loc[date, ticker], 'trade_value': change[ticker],
                        'transaction_cost': fees[ticker], 'nav_before_trade': nav,
                        'nav_after_trade': nav - fee, 'cash_after_trade': cash,
                    })
        value = float(cash + (holdings * closes.loc[date]).sum())
        daily.append({'date': date, 'value': value, 'cash': cash, 'cash_weight': cash/value,
                      'transaction_cost': fee, 'turnover': turnover,
                      **{f'units_{ticker}': holdings[ticker] for ticker in ASSETS}})
    return pd.DataFrame(daily), pd.DataFrame(ledger)


def portfolio_metrics(daily):
    values = daily.value.to_numpy()
    returns = values / np.r_[INITIAL_CAPITAL, values[:-1]] - 1
    std = returns.std(ddof=1)
    peak = np.maximum.accumulate(np.r_[INITIAL_CAPITAL, values])[1:]
    return {
        'total_return': values[-1]/INITIAL_CAPITAL - 1,
        'annualized_return': (values[-1]/INITIAL_CAPITAL)**(252/len(values)) - 1,
        'annualized_volatility': std*np.sqrt(252),
        'sharpe': returns.mean()/std*np.sqrt(252) if std > 1e-15 else np.nan,
        'maximum_drawdown': float(np.min(values/peak - 1)),
        'turnover': daily.turnover.sum(), 'transaction_costs': daily.transaction_cost.sum(),
        'average_cash_exposure': daily.cash_weight.mean(), 'sessions': len(daily),
        'start': str(daily.date.min().date()), 'end': str(daily.date.max().date()),
    }


def run_portfolios(prices, predictions, start, end):
    results, curves, ledgers = [], [], []
    candidates = [('equal_weight', None, 'equal_weight'), ('SPY', None, 'spy')]
    for model, group in predictions.groupby('model'):
        for rule in ['ranked', 'risk_aware']:
            candidates.append((f'{model}__{rule}', group, rule))
    for name, signals, rule in candidates:
        daily, ledger = simulate(prices, signals, rule, start, end)
        results.append({'strategy': name, **portfolio_metrics(daily)})
        curves.append(daily.assign(strategy=name))
        ledgers.append(ledger.assign(strategy=name))
    metrics = pd.DataFrame(results)
    if metrics[['start', 'end', 'sessions']].drop_duplicates().shape[0] != 1:
        raise AssertionError('Strategies use different evaluation periods')
    return metrics, pd.concat(curves, ignore_index=True), pd.concat(ledgers, ignore_index=True)


def select_entry(metrics):
    candidates = metrics[metrics.strategy.str.contains('__')].copy()
    candidates = candidates[np.isfinite(candidates.sharpe)]
    if candidates.empty:
        raise ValueError('No valid validation Sharpe for competition selection')
    row = candidates.sort_values(['sharpe', 'turnover', 'strategy'], ascending=[False, True, True]).iloc[0]
    return {'strategy': row.strategy, 'validation_sharpe': float(row.sharpe),
            'validation_turnover': float(row.turnover)}
