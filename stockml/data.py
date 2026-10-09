from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .common import FEATURES, SEED, STOCKS, SYMBOLS, TARGETS, digest, load_json, root, save_json, smoke, versions


def validate_prices(prices):
    expected = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume'}
    if not expected.issubset(prices.columns):
        raise ValueError(f'Missing price columns: {expected - set(prices.columns)}')
    if prices.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate date/ticker prices')
    if set(prices.ticker) != set(SYMBOLS):
        raise ValueError('Missing or unexpected ticker')
    numeric = prices[['open', 'high', 'low', 'close', 'volume']]
    if not np.isfinite(numeric.to_numpy()).all() or (numeric.iloc[:, :4] <= 0).any().any():
        raise ValueError('Missing, non-finite, or non-positive prices; investigate rather than fill')
    if (prices.volume < 0).any() or (prices.loc[prices.ticker.isin(STOCKS), 'volume'] <= 0).any():
        raise ValueError('Invalid stock volume')
    calendar = pd.DatetimeIndex(prices.loc[prices.ticker == 'SPY', 'date']).sort_values()
    if not calendar.is_unique or len(calendar) < 30:
        raise ValueError('Insufficient SPY calendar')
    for ticker, group in prices.groupby('ticker'):
        dates = pd.DatetimeIndex(group.date).sort_values()
        if not dates.equals(calendar):
            raise ValueError(f'{ticker} has missing/extra sessions relative to SPY; repair source data')
    return prices.sort_values(['date', 'ticker']).reset_index(drop=True)


def align_market_context(prices):
    """Discard VIX-only dates; never discard an equity session or fill a missing VIX value."""
    sessions = prices.loc[prices.ticker == 'SPY', 'date']
    excluded = (prices.ticker == '^VIX') & ~prices.date.isin(sessions)
    dates = prices.loc[excluded, 'date'].sort_values().dt.strftime('%Y-%m-%d').tolist()
    return validate_prices(prices.loc[~excluded].copy()), dates


def build_features(prices):
    """Pure transformation. No fitting, thresholds, scaling, or future feature inputs."""
    prices = validate_prices(prices)
    market = prices[prices.ticker == 'SPY'].set_index('date').sort_index()
    vix = prices[prices.ticker == '^VIX'].set_index('date').sort_index()
    context = pd.DataFrame({
        'spy_return_1': market.close.pct_change(fill_method=None),
        'spy_return_5': market.close.pct_change(5, fill_method=None),
        'vix_level': vix.close,
        'vix_change': vix.close.pct_change(fill_method=None),
    })
    frames = []
    for ticker in STOCKS:
        g = prices[prices.ticker == ticker].set_index('date').sort_index()
        p, volume = g.close, g.volume
        out = pd.DataFrame(index=g.index)
        out['ticker'] = ticker
        for lag in [1, 5, 10, 20]:
            out[f'return_{lag}'] = p.pct_change(lag, fill_method=None)
        for window in [5, 20]:
            out[f'price_sma_{window}'] = p / p.rolling(window).mean() - 1
            out[f'volatility_{window}'] = out.return_1.rolling(window).std(ddof=1)
        out['volume_change'] = volume.pct_change(fill_method=None)
        out['volume_relative_20'] = volume / volume.rolling(20).mean()
        change = p.diff()
        # Wilder-style exponential smoothing with a 14-observation warm-up.
        gain = change.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
        out['rsi_14'] = rsi
        out['macd_relative'] = (p.ewm(span=12, adjust=False, min_periods=12).mean()
                                - p.ewm(span=26, adjust=False, min_periods=26).mean()) / p
        out = out.join(context)
        out['relative_return_5'] = out.return_5 - out.spy_return_5
        future = pd.concat([out.return_1.shift(-i) for i in range(1, 6)], axis=1)
        out['future_return'] = p.shift(-5) / p - 1
        out['future_volatility'] = future.std(axis=1, ddof=1).where(future.notna().all(axis=1))
        out['target_end'] = pd.Series(g.index, index=g.index).shift(-5)
        frames.append(out.reset_index())
    table = pd.concat(frames, ignore_index=True)
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return table[['date', 'ticker'] + FEATURES + ['future_return', 'future_volatility', 'target_end']].sort_values(
        ['date', 'ticker']).reset_index(drop=True)


def synthetic_prices(end='2026-03-31'):
    """Deterministic offline fixture; never presented as financial evidence."""
    dates = pd.bdate_range('2015-01-01', end)
    rng = np.random.default_rng(SEED)
    common = rng.normal(0.0002, 0.009, len(dates))
    rows = []
    for i, ticker in enumerate(SYMBOLS):
        returns = common + rng.normal(0, 0.004 + i * .0005, len(dates))
        close = (50 + 10*i) * np.exp(np.cumsum(returns))
        if ticker == '^VIX':
            close = 18 + 8 * np.abs(np.sin(np.arange(len(dates)) / 100)) + rng.uniform(0, 3, len(dates))
        opening = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, .002, len(dates)))
        rows.append(pd.DataFrame({'date': dates, 'ticker': ticker, 'open': opening,
                                 'high': np.maximum(opening, close)*1.002,
                                 'low': np.minimum(opening, close)*.998, 'close': close,
                                 'volume': rng.integers(100000, 10000000, len(dates))}))
    return pd.concat(rows, ignore_index=True)


def download_prices(as_of=None):
    import yfinance as yf
    yf.set_tz_cache_location(str(root() / 'yfinance-cache'))
    # Conservatively exclude the current New York calendar day, even after market close.
    yesterday = datetime.now(ZoneInfo('America/New_York')).date() - timedelta(days=1)
    cutoff = min(yesterday, pd.Timestamp('2026-12-31').date())
    if as_of is not None:
        cutoff = min(cutoff, pd.Timestamp(as_of).date())
    if cutoff < pd.Timestamp('2026-01-01').date():
        raise ValueError('Competition workflow needs at least one completed 2026 session')
    rows = []
    for ticker in SYMBOLS:
        raw = yf.download(ticker, start='2015-01-01', end=str(cutoff + timedelta(days=1)),
                          auto_adjust=True, actions=False, progress=False, threads=False,
                          multi_level_index=False)
        if raw.empty:
            raise RuntimeError(f'Download failed for {ticker}; retry later. No synthetic fallback is automatic.')
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        raw = raw.rename_axis('date').reset_index()
        raw.columns = [str(c).lower() for c in raw.columns]
        raw['ticker'] = ticker
        rows.append(raw[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']])
    aligned, excluded = align_market_context(pd.concat(rows, ignore_index=True))
    return aligned, str(cutoff), excluded


def prepare_data(as_of=None):
    folder = root() / 'data'
    metadata_path = folder / 'metadata.json'
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata['synthetic'] != smoke():
            raise ValueError('Synthetic/real artifact collision: use separate STOCK_ML_ARTIFACTS folders')
        if as_of is not None and metadata['requested_as_of'] != str(as_of):
            raise ValueError('Cached cutoff differs; choose a new artifact directory for a new snapshot')
        for name, fingerprint in metadata['files'].items():
            if digest(folder / name) != fingerprint:
                raise ValueError(f'Cached data changed: {name}')
        # Older snapshots were strict-calendar validated and excluded no extra dates.
        metadata.setdefault('excluded_vix_non_equity_dates', [])
        return metadata
    folder.mkdir(parents=True, exist_ok=True)
    if smoke():
        prices, cutoff = validate_prices(synthetic_prices()), '2026-03-31'
        excluded = []
    else:
        prices, cutoff, excluded = download_prices(as_of)
    table = build_features(prices)
    for period, mask in [('history', table.date < '2026-01-01'), ('holdout', table.date >= '2026-01-01')]:
        part = table.loc[mask].copy()
        outcomes = part[['date', 'ticker', 'future_return', 'future_volatility', 'target_end']].copy()
        if period == 'history':
            # Crucial: historical artifact must not contain targets realized in 2026.
            crosses = outcomes.target_end >= pd.Timestamp('2026-01-01')
            outcomes.loc[crosses, ['future_return', 'future_volatility']] = np.nan
            outcomes.loc[crosses, 'target_end'] = pd.NaT
        part[['date', 'ticker'] + FEATURES].to_parquet(folder / f'{period}_features.parquet', index=False)
        outcomes.to_parquet(folder / f'{period}_outcomes.parquet', index=False)
        price_mask = prices.date < '2026-01-01' if period == 'history' else prices.date >= '2026-01-01'
        prices.loc[price_mask].to_parquet(folder / f'{period}_prices.parquet', index=False)
    metadata = {
        'downloaded_at_utc': datetime.now(ZoneInfo('UTC')).isoformat(),
        'requested_as_of': str(as_of) if as_of is not None else None,
        'maximum_calendar_cutoff': cutoff, 'last_completed_session': str(prices.date.max().date()),
        'synthetic': smoke(), 'features': FEATURES, 'targets': TARGETS,
        'adjustment': 'yfinance auto_adjust=True for all OHLC; adjusted fractional-unit simulation; no separate dividends',
        'calendar': 'SPY sessions; VIX-only non-equity dates excluded and recorded; all equity sessions required; no filling',
        'excluded_vix_non_equity_dates': excluded,
        'rsi': '14-period Wilder-style EWM alpha=1/14, adjust=False; min_periods=14',
        'volume_change': 'V[t]/V[t-1]-1', 'vix_change': 'VIX[t]/VIX[t-1]-1',
        'versions': versions(),
        'files': {p.name: digest(p) for p in sorted(folder.glob('*.parquet'))},
    }
    save_json(metadata_path, metadata)
    return metadata
