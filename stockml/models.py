from __future__ import annotations

from itertools import product

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

from .common import (FEATURES, SEED, STOCKS, TARGETS, digest, feature_matrix, identifier,
                     load_json, read_table, root, save_json, smoke, versions)
from .portfolio import RULES, run_portfolios, select_entry


def labeled_window(table, start, boundary):
    """Both signal and complete target must be strictly before the next period."""
    selected = table[(table.date >= pd.Timestamp(start)) & (table.date < pd.Timestamp(boundary))
                     & (table.target_end < pd.Timestamp(boundary))].copy()
    selected = selected.dropna(subset=['future_return', 'future_volatility', 'target_end'])
    if selected.empty:
        raise ValueError(f'No complete observations between {start} and {boundary}')
    assert selected.target_end.max() < pd.Timestamp(boundary)
    return selected


def risk_thresholds(train):
    thresholds = train.groupby('ticker').future_volatility.quantile(.75).to_dict()
    if set(thresholds) != set(STOCKS):
        raise ValueError('Training must include all stocks')
    return thresholds


def labels(frame, thresholds):
    return (frame.future_volatility > frame.ticker.map(thresholds)).astype(int)


def make_model(algorithm, task, params):
    # Serial tree reductions ensure bit-identical predictions across saves/reloads.
    common = {'n_estimators': 8 if smoke() else 300, 'random_state': SEED, 'n_jobs': 1}
    if algorithm == 'rf':
        cls = RandomForestRegressor if task == 'regression' else RandomForestClassifier
        return cls(**common, **params)
    cls = XGBRegressor if task == 'regression' else XGBClassifier
    return cls(**common, **params, tree_method='hist',
               objective='reg:squarederror' if task == 'regression' else 'binary:logistic',
               eval_metric='rmse' if task == 'regression' else 'logloss')


def parameter_grid(algorithm):
    if smoke():
        return [{'max_depth': 3, 'min_samples_leaf': 10}] if algorithm == 'rf' else [{'max_depth': 2, 'learning_rate': .1}]
    if algorithm == 'rf':
        return [dict(max_depth=d, min_samples_leaf=m) for d, m in product([4, 8, None], [10, 30])]
    return [dict(max_depth=d, learning_rate=r) for d, r in product([2, 3, 5], [.03, .1])]


def positive_probability(model, matrix):
    # RF can fit a single-class subset; make its constant probability explicit.
    if len(model.classes_) == 1:
        return np.full(len(matrix), float(model.classes_[0]))
    return model.predict_proba(matrix)[:, list(model.classes_).index(1)]


def tune(train, validation):
    thresholds = risk_thresholds(train)
    x_train, x_val = feature_matrix(train), feature_matrix(validation)
    chosen, records = {}, []
    for algorithm in ['rf', 'xgb']:
        chosen[algorithm] = {}
        for task in ['regression', 'classification']:
            y_train = train.future_return if task == 'regression' else labels(train, thresholds)
            y_val = validation.future_return if task == 'regression' else labels(validation, thresholds)
            if task == 'classification' and (y_train.nunique() < 2 or y_val.nunique() < 2):
                raise ValueError('Both classes are required for ROC-AUC tuning')
            candidates = []
            for params in parameter_grid(algorithm):
                model = make_model(algorithm, task, params).fit(x_train, y_train)
                pred = model.predict(x_val) if task == 'regression' else positive_probability(model, x_val)
                score = np.sqrt(mean_squared_error(y_val, pred)) if task == 'regression' else roc_auc_score(y_val, pred)
                candidates.append((score if task == 'regression' else -score, params))
                records.append({'algorithm': algorithm, 'task': task, 'params': str(params),
                                'metric': 'rmse' if task == 'regression' else 'roc_auc', 'score': score})
            chosen[algorithm][task] = min(candidates, key=lambda x: x[0])[1]
    return chosen, pd.DataFrame(records)


def fit_bundle(train, chosen, experiment, boundary, feature_names=FEATURES):
    if (train.date >= pd.Timestamp(boundary)).any() or (train.target_end >= pd.Timestamp(boundary)).any():
        raise ValueError('Training signals or outcomes cross the fit cutoff')
    if experiment in ['validation', 'assignment', 'competition'] and (train.target_end >= pd.Timestamp('2026-01-01')).any():
        raise ValueError('2026 may not enter model fitting')
    metadata = load_json(root() / 'data' / 'metadata.json')
    thresholds = risk_thresholds(train)
    y_risk = labels(train, thresholds)
    signature = {
        'experiment': experiment, 'training_boundary_exclusive': boundary,
        'features': list(feature_names), 'ticker_encoding': STOCKS, 'thresholds': thresholds,
        'chosen': chosen, 'seed': SEED, 'versions': versions(), 'targets': TARGETS,
        'history_fingerprints': {k: v for k, v in metadata['files'].items() if k.startswith('history_')},
        'synthetic': smoke(), 'trees': 8 if smoke() else 300,
    }
    bundle = {**signature, 'bundle_id': f'{experiment}-{identifier(signature)}', 'models': {},
              'majority': int(y_risk.mean() > .5), 'majority_probability': float(y_risk.mean()),
              'last_training_signal': str(train.date.max().date()),
              'last_training_target': str(train.target_end.max().date())}
    for algorithm in ['rf', 'xgb']:
        for layout in ['pooled', 'individual']:
            name = f'{algorithm}_{layout}'
            bundle['models'][name] = {}
            groups = [('all', train)] if layout == 'pooled' else list(train.groupby('ticker', sort=True))
            for key, group in groups:
                x = feature_matrix(group, feature_names, pooled=layout == 'pooled')
                risk = labels(group, thresholds)
                if risk.nunique() < 2:
                    raise ValueError(f'Insufficient risk classes for {name}/{key}')
                bundle['models'][name][key] = {
                    'regression': make_model(algorithm, 'regression', chosen[algorithm]['regression']).fit(x, group.future_return),
                    'classification': make_model(algorithm, 'classification', chosen[algorithm]['classification']).fit(x, risk),
                }
    return bundle


def infer(bundle, features):
    # Select only identifiers and predictors. Realized targets are neither accepted nor needed.
    feature_matrix(features, bundle['features'])
    if any(name in features for name in ['future_return', 'future_volatility', 'target_end']):
        raise ValueError('Inference input must not contain realized outcomes')
    if features.duplicated(['date', 'ticker']).any():
        raise ValueError('Duplicate inference rows')
    outputs = []
    for name, estimators in bundle['models'].items():
        groups = [('all', features)] if name.endswith('_pooled') else list(features.groupby('ticker', sort=True))
        for key, group in groups:
            x = feature_matrix(group, bundle['features'], pooled=key == 'all')
            pair = estimators[key]
            out = group[['date', 'ticker']].copy()
            out['predicted_return'] = pair['regression'].predict(x)
            out['high_risk_probability'] = positive_probability(pair['classification'], x)
            out['model'], out['bundle_id'], out['experiment'] = name, bundle['bundle_id'], bundle['experiment']
            outputs.append(out)
    return pd.concat(outputs, ignore_index=True).sort_values(['model', 'date', 'ticker']).reset_index(drop=True)


def save_bundle(bundle, sample):
    folder = root() / 'models'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{bundle['experiment']}.joblib"
    joblib.dump(bundle, path)
    loaded = joblib.load(path)  # Only load this workflow's own trusted local artifacts.
    pd.testing.assert_frame_equal(infer(bundle, sample), infer(loaded, sample), check_exact=True)
    save_json(path.with_suffix('.json'), {key: value for key, value in bundle.items() if key != 'models'})
    return path


def prediction_window(features, start, end):
    prior = features.loc[features.date < pd.Timestamp(start), 'date'].max()
    if pd.isna(prior):
        raise ValueError('Missing prior session needed for first executable prediction')
    return features[(features.date >= prior) & (features.date <= pd.Timestamp(end))].copy()


def safe_spearman(a, b):
    return float(spearmanr(a, b).statistic) if len(a) > 1 and pd.Series(a).nunique() > 1 and pd.Series(b).nunique() > 1 else np.nan


def evaluate(predictions, outcomes, bundle, start, boundary):
    actual = labeled_window(outcomes, start, boundary)
    actual['risk'] = labels(actual, bundle['thresholds'])
    joined = predictions.merge(actual, on=['date', 'ticker'], validate='many_to_one')
    records = []
    groups = [(name, group) for name, group in joined.groupby('model')]
    base = actual.copy()
    base['predicted_return'] = 0.0
    base['high_risk_probability'] = bundle['majority_probability']
    groups.append(('baseline', base))
    for name, group in groups:
        for ticker, part in [('ALL', group), *list(group.groupby('ticker', sort=True))]:
            risk_pred = ((part.high_risk_probability >= .5).astype(int) if name != 'baseline'
                         else np.full(len(part), bundle['majority']))
            daily_ic = [safe_spearman(g.future_return, g.predicted_return) for _, g in part.groupby('date')]
            valid_ic = [value for value in daily_ic if np.isfinite(value)]
            records.append({
                'experiment': bundle['experiment'], 'model': name, 'ticker': ticker, 'n': len(part),
                'mae': mean_absolute_error(part.future_return, part.predicted_return),
                'rmse': np.sqrt(mean_squared_error(part.future_return, part.predicted_return)),
                # Zero predictions count as correct only when realized return is exactly zero.
                'directional_accuracy': float((np.sign(part.predicted_return) == np.sign(part.future_return)).mean()),
                'spearman': safe_spearman(part.future_return, part.predicted_return),
                'daily_cross_sectional_spearman': np.mean(valid_ic) if valid_ic else np.nan,
                'accuracy': accuracy_score(part.risk, risk_pred),
                'precision': precision_score(part.risk, risk_pred, zero_division=0),
                'recall': recall_score(part.risk, risk_pred, zero_division=0),
                'f1': f1_score(part.risk, risk_pred, zero_division=0),
                'roc_auc': roc_auc_score(part.risk, part.high_risk_probability) if part.risk.nunique() > 1 else np.nan,
            })
    metrics = pd.DataFrame(records)
    for _, group in metrics.groupby('ticker'):
        if group.n.nunique() != 1:
            raise AssertionError('Models/baselines evaluated on different observations')
    return metrics


def interpretation(bundle, validation, train, chosen):
    records = []
    # Bounded deterministic sample; interpretation is validation-only.
    sample = validation.sample(min(1000 if not smoke() else 100, len(validation)), random_state=SEED)
    for name, estimators in bundle['models'].items():
        for key, pair in estimators.items():
            subset = sample if key == 'all' else sample[sample.ticker == key]
            x = feature_matrix(subset, bundle['features'], pooled=key == 'all')
            for task, model in pair.items():
                target = subset.future_return if task == 'regression' else labels(subset, bundle['thresholds'])
                if task == 'classification' and target.nunique() < 2:
                    continue
                if key == 'all':
                    result = permutation_importance(model, x, target, random_state=SEED,
                                                    n_repeats=2 if smoke() else 5, n_jobs=1,
                                                    scoring='neg_root_mean_squared_error' if task == 'regression' else 'roc_auc')
                    scores, method = result.importances_mean, 'validation_permutation'
                else:
                    scores, method = model.feature_importances_, 'training_impurity_or_gain'
                records.extend({'model': name, 'ticker': key, 'task': task, 'feature': feature,
                                'importance': float(value), 'method': method}
                               for feature, value in zip(x.columns, scores))
    # Controlled pooled ablation: same rows and hyperparameters; momentum plus ticker only.
    ablations = []
    for algorithm in ['rf', 'xgb']:
        for task in ['regression', 'classification']:
            for label, names in [('momentum_only', FEATURES[:4]), ('full', FEATURES)]:
                x = feature_matrix(train, names)
                y = train.future_return if task == 'regression' else labels(train, bundle['thresholds'])
                model = make_model(algorithm, task, chosen[algorithm][task]).fit(x, y)
                x_val = feature_matrix(validation, names)
                value = (np.sqrt(mean_squared_error(validation.future_return, model.predict(x_val)))
                         if task == 'regression' else roc_auc_score(labels(validation, bundle['thresholds']), positive_probability(model, x_val)))
                ablations.append({'algorithm': algorithm, 'task': task, 'features': label,
                                  'metric': 'rmse' if task == 'regression' else 'roc_auc', 'score': value})
    return pd.DataFrame(records), pd.DataFrame(ablations)


def historical_inputs():
    metadata = load_json(root() / 'data' / 'metadata.json')
    if metadata['synthetic'] != smoke():
        raise ValueError('Synthetic/real mode mismatch')
    for name, fingerprint in metadata['files'].items():
        if name.startswith('history_') and digest(root() / 'data' / name) != fingerprint:
            raise ValueError(f'Historical dataset changed: {name}')
    features, outcomes = read_table('history_features'), read_table('history_outcomes')
    return features, outcomes, features.merge(outcomes, on=['date', 'ticker'], validate='one_to_one')


def develop_and_freeze():
    """Only pre-2024 rows are used in this phase. Freeze before either final experiment."""
    if (root() / 'frozen_experiment.json').exists():
        # A normal rerun must not turn completed test results into a new selection round.
        return check_frozen()
    features, outcomes, table = historical_inputs()
    train = labeled_window(table, '2015-01-01', '2022-01-01')
    validation = labeled_window(table, '2022-01-01', '2024-01-01')
    folder = root() / 'reports'
    folder.mkdir(parents=True, exist_ok=True)
    chosen, tuning = tune(train, validation)
    tuning.to_csv(folder / 'tuning.csv', index=False)
    bundle = fit_bundle(train, chosen, 'validation', '2022-01-01')
    signal_features = prediction_window(features, '2022-01-01', '2023-12-31')
    predictions = infer(bundle, signal_features)
    save_bundle(bundle, signal_features.head(16))
    (root() / 'predictions').mkdir(exist_ok=True)
    predictions.to_parquet(root() / 'predictions' / 'validation.parquet', index=False)
    evaluate(predictions, outcomes, bundle, '2022-01-01', '2024-01-01').to_csv(folder / 'validation_metrics.csv', index=False)
    importance, ablation = interpretation(bundle, validation, train, chosen)
    importance.to_csv(folder / 'feature_importance.csv', index=False)
    ablation.to_csv(folder / 'feature_ablation.csv', index=False)
    prices = read_table('history_prices')
    prices = prices[prices.date < '2024-01-01']
    performance, daily, ledger = run_portfolios(prices, predictions, '2022-01-01', '2023-12-31')
    performance.to_csv(folder / 'validation_portfolios.csv', index=False)
    daily.to_parquet(folder / 'validation_daily.parquet', index=False)
    ledger.to_parquet(folder / 'validation_trades.parquet', index=False)
    frozen = {
        'selected_entry': select_entry(performance), 'hyperparameters': chosen, 'rules': RULES,
        'validation_bundle_id': bundle['bundle_id'], 'history_fingerprints': bundle['history_fingerprints'],
        'synthetic': smoke(), 'selection_period': ['2022-01-01', '2023-12-31'],
        'features': FEATURES, 'seed': SEED,
    }
    frozen['freeze_id'] = identifier(frozen)
    path = root() / 'frozen_experiment.json'
    if path.exists() and load_json(path) != frozen:
        raise ValueError('A different experiment is already frozen. Do not overwrite it after seeing test results.')
    save_json(path, frozen)
    return frozen


def check_frozen():
    path = root() / 'frozen_experiment.json'
    if not path.exists():
        raise RuntimeError('Run validation-only development and strategy selection before final fitting')
    frozen = load_json(path)
    content = {k: v for k, v in frozen.items() if k != 'freeze_id'}
    if identifier(content) != frozen['freeze_id'] or frozen['rules'] != RULES or frozen['features'] != FEATURES:
        raise ValueError('Frozen experiment or strategy rules have changed')
    metadata = load_json(root() / 'data' / 'metadata.json')
    if frozen['synthetic'] != smoke():
        raise ValueError('Synthetic/real mode mismatch')
    for name, fingerprint in frozen['history_fingerprints'].items():
        if metadata['files'][name] != fingerprint or digest(root() / 'data' / name) != fingerprint:
            raise ValueError('Frozen historical snapshot has changed')
    return frozen


def fit_final(experiment):
    frozen = check_frozen()
    if experiment not in ['assignment', 'competition']:
        raise ValueError('Unknown final experiment')
    existing = root() / 'models' / f'{experiment}.joblib'
    if existing.exists():
        bundle = joblib.load(existing)
        if bundle.get('freeze_id') != frozen['freeze_id']:
            raise ValueError('Existing final model belongs to a different frozen experiment')
    else:
        boundary = '2024-01-01' if experiment == 'assignment' else '2026-01-01'
        features, outcomes, table = historical_inputs()
        train = labeled_window(table, '2015-01-01', boundary)
        bundle = fit_bundle(train, frozen['hyperparameters'], experiment, boundary)
        bundle['freeze_id'] = frozen['freeze_id']
        save_bundle(bundle, features.tail(16))
    if experiment == 'assignment':
        # Recreate reports even if a previous run stopped after writing its model.
        features, outcomes, _ = historical_inputs()
        predictions = infer(bundle, prediction_window(features, '2024-01-01', '2025-12-31'))
        predictions.to_parquet(root() / 'predictions' / 'assignment.parquet', index=False)
        evaluate(predictions, outcomes, bundle, '2024-01-01', '2026-01-01').to_csv(root() / 'reports' / 'assignment_metrics.csv', index=False)
    return bundle


def competition_inference(bundle):
    frozen = check_frozen()
    if bundle['experiment'] != 'competition' or bundle['freeze_id'] != frozen['freeze_id']:
        raise ValueError('Competition requires its own frozen refitted bundle')
    metadata = load_json(root() / 'data' / 'metadata.json')
    for name in ['holdout_features.parquet', 'holdout_outcomes.parquet']:
        if digest(root() / 'data' / name) != metadata['files'][name]:
            raise ValueError(f'Holdout snapshot changed: {name}')
    # First access to 2026 predictor values; fitting is already finished.
    holdout = read_table('holdout_features')
    history = read_table('history_features')
    prior = history[history.date == history.date.max()]
    signals = pd.concat([prior, holdout], ignore_index=True)
    predictions = infer(bundle, signals)
    predictions.to_parquet(root() / 'predictions' / 'competition.parquet', index=False)
    # Outcomes loaded only after prediction generation; never passed to infer().
    outcomes = read_table('holdout_outcomes')
    metrics = evaluate(predictions, outcomes, bundle, '2026-01-01', '2027-01-01')
    metrics.to_csv(root() / 'reports' / 'competition_metrics.csv', index=False)
    save_json(root() / 'competition_evaluation.json', {
        'freeze_id': frozen['freeze_id'], 'bundle_id': bundle['bundle_id'],
        'label': '2026 year-to-date' if metadata['maximum_calendar_cutoff'] < '2026-12-31' else '2026 full year',
        'last_completed_session': metadata['last_completed_session'],
        'holdout_fingerprints': {k: v for k, v in metadata['files'].items() if k.startswith('holdout_')},
        'warning': 'Do not change decisions in response to these holdout results.',
    })
    return predictions, metrics
