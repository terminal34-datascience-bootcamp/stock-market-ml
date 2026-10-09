import numpy as np
import pandas as pd
import pytest

from stockml.common import FEATURES, STOCKS, feature_matrix
from stockml.data import align_market_context, build_features, prepare_data, synthetic_prices, validate_prices
from stockml.models import check_frozen, fit_bundle, infer, labeled_window, risk_thresholds, save_bundle
from stockml.portfolio import (ASSETS, INITIAL_CAPITAL, portfolio_metrics, rebalance,
                               run_portfolios, select_entry, simulate, weights_for)


@pytest.fixture(scope='module')
def prices():
    return synthetic_prices('2015-08-31')


def test_five_day_targets_match_hand_calculation(prices):
    table = build_features(prices)
    stock = prices[prices.ticker == 'AAPL'].set_index('date').sort_index()
    row = table[table.ticker == 'AAPL'].iloc[15]
    i = stock.index.get_loc(row.date)
    expected = stock.close.iloc[i+5] / stock.close.iloc[i] - 1
    five_returns = stock.close.iloc[i+1:i+6].to_numpy() / stock.close.iloc[i:i+5].to_numpy() - 1
    assert row.future_return == pytest.approx(expected)
    assert row.future_volatility == pytest.approx(np.std(five_returns, ddof=1))
    assert row.target_end == stock.index[i+5]
    final = table[table.ticker == 'AAPL'].tail(5)
    assert final.future_return.isna().all()
    assert final.future_volatility.isna().all()
    assert final[FEATURES].notna().all().all()


def test_future_prices_cannot_change_past_features(prices):
    cutoff = pd.Timestamp('2015-05-01')
    altered = prices.copy()
    altered.loc[altered.date > cutoff, ['open', 'high', 'low', 'close']] *= 3
    altered.loc[altered.date > cutoff, 'volume'] *= 2
    before, after = build_features(prices), build_features(altered)
    pd.testing.assert_frame_equal(before.loc[before.date <= cutoff, ['date', 'ticker'] + FEATURES],
                                  after.loc[after.date <= cutoff, ['date', 'ticker'] + FEATURES])


def test_calendar_and_duplicates_fail_explicitly(prices):
    with pytest.raises(ValueError, match='Duplicate'):
        validate_prices(pd.concat([prices, prices.iloc[:1]]))
    with pytest.raises(ValueError, match='missing/extra'):
        validate_prices(prices.drop(prices.index[0]))


def test_vix_only_non_equity_dates_can_be_excluded_without_losing_sessions(prices):
    extra = prices[prices.ticker == '^VIX'].iloc[:1].copy()
    extra['date'] = pd.Timestamp('2015-01-03')  # Weekend in the synthetic equity calendar.
    aligned, excluded = align_market_context(pd.concat([prices, extra], ignore_index=True))
    assert excluded == ['2015-01-03']
    pd.testing.assert_frame_equal(aligned, validate_prices(prices))
    missing_vix = prices.drop(prices[prices.ticker == '^VIX'].index[0])
    with pytest.raises(ValueError, match='missing/extra'):
        align_market_context(missing_vix)


def test_purged_targets_do_not_cross_boundary(prices):
    table = build_features(prices)
    boundary = pd.Timestamp('2015-06-01')
    selected = labeled_window(table, '2015-01-01', boundary)
    assert (selected.target_end < boundary).all()
    assert selected.groupby('date').ticker.nunique().eq(8).all()
    assert len(table[(table.date < boundary) & (table.target_end >= boundary)]) == 5*8


def test_feature_order_and_missing_values_rejected(prices):
    features = build_features(prices)[['date', 'ticker'] + FEATURES]
    assert feature_matrix(features).shape[1] == 25
    with pytest.raises(ValueError, match='schema/order'):
        feature_matrix(features.drop(columns=FEATURES[0]))
    with pytest.raises(ValueError, match='schema/order'):
        feature_matrix(features[['date', 'ticker'] + FEATURES[::-1]])
    with pytest.raises(ValueError, match='Non-finite'):
        feature_matrix(features.assign(return_1=np.nan))


def test_thresholds_depend_only_on_training(prices):
    table = build_features(prices)
    train = labeled_window(table, '2015-01-01', '2015-06-01')
    first = risk_thresholds(train)
    table.loc[table.date >= '2015-06-01', 'future_volatility'] = 1000
    assert risk_thresholds(labeled_window(table, '2015-01-01', '2015-06-01')) == first


def test_2026_training_rejected_before_fit():
    table = pd.DataFrame({'date': pd.to_datetime(['2026-01-02']), 'target_end': pd.to_datetime(['2026-01-09'])})
    with pytest.raises(ValueError, match='cutoff'):
        fit_bundle(table, {}, 'competition', '2026-01-01')


def test_frozen_selection_required(tmp_path, monkeypatch):
    monkeypatch.setenv('STOCK_ML_ARTIFACTS', str(tmp_path))
    with pytest.raises(RuntimeError, match='validation-only'):
        check_frozen()


def constant_prices():
    dates = pd.bdate_range('2025-12-31', periods=13)
    prices = pd.DataFrame([{'date': d, 'ticker': ticker, 'open': 100., 'close': 100.}
                           for d in dates for ticker in ASSETS])
    return prices, dates


def test_initial_cost_matches_hand_calculation_and_trade_timing():
    prices, dates = constant_prices()
    daily, ledger = simulate(prices, None, 'spy', dates[1], dates[-1])
    expected = INITIAL_CAPITAL / 1.001
    np.testing.assert_allclose(daily.value, expected)
    assert ledger.transaction_cost.sum() == pytest.approx(expected * .001)
    assert ledger.iloc[0].units_after == pytest.approx(expected / 100)
    assert len(ledger) == 1  # Buy-and-hold never periodically rebalances.
    assert (ledger.signal_date < ledger.execution_date).all()
    assert ledger.iloc[0].signal_date == dates[0]
    free, _ = simulate(prices, None, 'spy', dates[1], dates[-1], cost_rate=0)
    np.testing.assert_allclose(free.value, INITIAL_CAPITAL)
    metrics = portfolio_metrics(daily)
    assert metrics['maximum_drawdown'] == pytest.approx(expected / INITIAL_CAPITAL - 1)


@pytest.mark.parametrize('eligible', [0, 1, 2, 3])
def test_risk_slots_and_cash(eligible):
    signals = pd.DataFrame({'ticker': sorted(STOCKS), 'predicted_return': 0.,
                            'high_risk_probability': [.1]*eligible + [.9]*(8-eligible)})
    weights = weights_for(signals, 'risk_aware')
    assert weights.sum() == pytest.approx(eligible / 3)
    assert (weights > 0).sum() == eligible
    holdings, cash, _, _, _ = rebalance(pd.Series(0., index=ASSETS), INITIAL_CAPITAL,
                                        pd.Series(100., index=ASSETS), weights, .001)
    expected_nav = INITIAL_CAPITAL / (1 + .001 * eligible/3)
    assert cash == pytest.approx(expected_nav * (1-eligible/3), abs=1e-6)
    assert (holdings >= 0).all()


def test_tie_break_and_missing_stock_signal():
    signals = pd.DataFrame({'ticker': STOCKS, 'predicted_return': 0., 'high_risk_probability': .2})
    picks = weights_for(signals, 'ranked')
    assert set(picks[picks > 0].index) == set(sorted(STOCKS)[:3])
    with pytest.raises(ValueError, match='eight selected'):
        weights_for(signals.iloc[:-1], 'ranked')


def test_turnover_accounts_for_drift_and_sells():
    holdings = pd.Series(0., index=ASSETS)
    holdings['AAPL'] = 100
    opening = pd.Series(100., index=ASSETS)
    opening['AAPL'] = 120  # Position has drifted to $12,000.
    weights = pd.Series(0., index=ASSETS)
    weights['MSFT'] = 1
    after, cash, changes, fees, nav = rebalance(holdings, 0, opening, weights, .001)
    # Sell 12,000; pay 12 selling cost; buy with the cash remaining after buy cost.
    bought = (12000 - 12) / 1.001
    assert nav == 12000
    assert changes['AAPL'] == -12000
    assert changes['MSFT'] == pytest.approx(bought)
    assert fees.sum() == pytest.approx(12 + .001*bought)
    assert cash == pytest.approx(0, abs=1e-6)
    assert after['AAPL'] == 0


def test_common_calendar_and_validation_selection():
    prices, dates = constant_prices()
    signals = pd.DataFrame([{'date': d, 'ticker': t, 'predicted_return': .01,
                             'high_risk_probability': .2, 'model': 'rf_pooled'}
                            for d in dates for t in STOCKS])
    metrics, daily, _ = run_portfolios(prices, signals, dates[1], dates[-1])
    assert metrics[['start', 'end', 'sessions']].drop_duplicates().shape[0] == 1
    assert daily.groupby('strategy').date.nunique().nunique() == 1
    candidates = pd.DataFrame({'strategy': ['SPY', 'b__ranked', 'a__ranked'],
                               'sharpe': [100., 1., 1.], 'turnover': [0., 2., 1.]})
    assert select_entry(candidates)['strategy'] == 'a__ranked'


def test_historical_export_never_contains_2026_outcomes(tmp_path, monkeypatch):
    monkeypatch.setenv('STOCK_ML_ARTIFACTS', str(tmp_path))
    monkeypatch.setenv('STOCK_ML_SMOKE', '1')
    prepare_data()
    outcomes = pd.read_parquet(tmp_path / 'data' / 'history_outcomes.parquet')
    assert outcomes.target_end.dropna().max() < pd.Timestamp('2026-01-01')
    assert outcomes.groupby('ticker').tail(5).future_return.isna().all()
    features = pd.read_parquet(tmp_path / 'data' / 'holdout_features.parquet')
    assert features.date.min() == pd.Timestamp('2026-01-01')
    assert features[FEATURES].notna().all().all()
    assert not set(['future_return', 'future_volatility', 'target_end']).intersection(features.columns)


def test_bundle_roundtrip_and_inference_without_outcomes(tmp_path, monkeypatch, prices):
    import joblib
    monkeypatch.setenv('STOCK_ML_ARTIFACTS', str(tmp_path))
    monkeypatch.setenv('STOCK_ML_SMOKE', '1')
    prepare_data()
    table = build_features(prices)
    train = labeled_window(table, '2015-01-01', '2015-07-01')
    chosen = {'rf': {task: {'max_depth': 3, 'min_samples_leaf': 10} for task in ['regression', 'classification']},
              'xgb': {task: {'max_depth': 2, 'learning_rate': .1} for task in ['regression', 'classification']}}
    bundle = fit_bundle(train, chosen, 'validation', '2022-01-01')
    sample = table.tail(16)[['date', 'ticker'] + FEATURES]
    original = infer(bundle, sample)
    path = save_bundle(bundle, sample)
    pd.testing.assert_frame_equal(original, infer(joblib.load(path), sample), check_exact=True)
    assert len(original) == 4 * len(sample)
    assert original[['predicted_return', 'high_risk_probability']].notna().all().all()
    with pytest.raises(ValueError, match='realized outcomes'):
        infer(bundle, sample.assign(future_return=0))
