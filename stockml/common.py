from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STOCKS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'JPM', 'XOM', 'JNJ']
SYMBOLS = STOCKS + ['SPY', '^VIX']
FEATURES = [
    'return_1', 'return_5', 'return_10', 'return_20',
    'price_sma_5', 'price_sma_20', 'volatility_5', 'volatility_20',
    'volume_change', 'volume_relative_20', 'rsi_14', 'macd_relative',
    'spy_return_1', 'spy_return_5', 'vix_level', 'vix_change', 'relative_return_5',
]
SEED = 42
TARGETS = {
    'return': 'adjusted_close[t+5] / adjusted_close[t] - 1',
    'volatility': 'sample std (ddof=1) of daily returns t+1,...,t+5',
    'risk': 'future_volatility > per-stock training 75th percentile',
}


def root() -> Path:
    path = Path(os.environ.get('STOCK_ML_ARTIFACTS', 'artifacts')).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def smoke() -> bool:
    return os.environ.get('STOCK_ML_SMOKE', '0') == '1'


def versions():
    names = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'yfinance',
             'pyarrow', 'joblib', 'matplotlib']
    return {name: importlib.metadata.version(name) for name in names}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identifier(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n')
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text())


def feature_matrix(frame: pd.DataFrame, names=FEATURES, pooled=True) -> pd.DataFrame:
    # Check the relative order, not merely membership: silent column swaps are unsafe.
    observed = [name for name in frame.columns if name in names]
    if observed != list(names):
        raise ValueError(f'Feature schema/order mismatch: expected {list(names)}, got {observed}')
    if not frame.ticker.isin(STOCKS).all():
        raise ValueError('Unknown ticker in inference data')
    matrix = frame[list(names)].astype(float).copy()
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError('Non-finite model input')
    if pooled:
        for ticker in STOCKS:
            matrix[f'ticker_{ticker}'] = (frame.ticker == ticker).astype(float)
    return matrix


def read_table(name):
    return pd.read_parquet(root() / 'data' / f'{name}.parquet')
