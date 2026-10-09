"""Regenerate the initial notebook templates. Do not run over hand-edited notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


BOOTSTRAP = (ROOT / 'scripts' / 'colab_setup.py').read_text()


def setup(number):
    return ('''# Use the SAME values in all three Colab notebooks.
EXPERIMENT_NAME = 'competition-v1'
COLAB_SMOKE_TEST = False  # True = synthetic data and small models, stored separately.

''' + BOOTSTRAP + f'\nPROJECT = setup_notebook({number}, EXPERIMENT_NAME, COLAB_SMOKE_TEST)\n' + '''
import pandas as pd
from IPython.display import display, Markdown
from stockml.common import root, smoke, load_json

pd.set_option('display.max_columns', 20)
print('Artifacts:', root())
print('SYNTHETIC SMOKE TEST — NOT FINANCIAL RESULTS' if smoke() else 'REAL-DATA EXPERIMENT')
''')


COLAB_GUIDE = '''## Run locally or in Google Colab

In Colab, select a **Python 3 CPU runtime**, then run the setup cell below and authorize its Google Drive mount. It loads the shared code from GitHub and installs the runtime dependencies. Use the same `EXPERIMENT_NAME` and `COLAB_SMOKE_TEST` values in notebooks **1 → 2 → 3**; each can run in a separate session. Your datasets, models, predictions, and frozen selection are saved directly under `My Drive/stock-market-ml/<experiment>/real/` (or `smoke/`), so they survive runtime resets. Save a notebook copy to Drive if you want to keep your notes and cell outputs too.

**Private GitHub repository:** add a `GITHUB_TOKEN` in Colab's Secrets panel (key icon), using a fine-grained token with **Contents: Read-only** access to this repository, and enable notebook access in each notebook. Organization approval may be required. Do not paste the token into a code cell. Setup passes it only to Git subprocesses and does not save it in artifacts or Git remote URLs. If Colab cannot open the private GitHub link, download the `.ipynb` from GitHub and upload it to Colab; the same secret enables the code download.

The first notebook records a Git commit and exact package versions; later notebooks reuse them. If setup asks for a runtime restart after installing packages, restart the session and rerun setup. Run one notebook at a time for a given experiment. A missing-handoff error means you need to finish the preceding notebook with the same settings. On a local machine, setup preserves your environment and artifact path; install `requirements.txt` first.

Changing the experiment folder after seeing 2026 results does not restore holdout independence. To try this Colab port without revisiting the real holdout, set `COLAB_SMOKE_TEST=True` in all three notebooks. Your existing local artifacts are not uploaded automatically.
'''

notebooks = {
    '01_data_preprocessing.ipynb': [
        md('''# 1. Data acquisition, exploration, and preprocessing

**Research question:** Do Random Forest and XGBoost improve on simple prediction and portfolio baselines?

Use 2015–2021 for training and 2022–2023 for validation. Preserve 2024–2025 for the assignment test and **2026 as a separate competition holdout**. No 2026 performance is explored here.

Run the three notebooks in numerical order, each in a fresh kernel. Install `requirements.txt` and select that environment. Shared, tested implementations are in `stockml/`; retain that folder with the notebooks. Normal runs download real data. The automated smoke runner explicitly uses deterministic synthetic data and reduced models.
'''),
        md(COLAB_GUIDE),
        code(setup(1)),
        md('''## Acquire and lock a reproducible snapshot

`AS_OF=None` uses the latest available session through yesterday in New York, capped at December 2026. The current day is conservatively excluded. A cached snapshot is verified and reused, not silently refreshed. To intentionally acquire another snapshot, use a different `STOCK_ML_ARTIFACTS` directory; changing snapshots after inspecting results does not restore holdout independence.

All OHLC fields use `auto_adjust=True`. This supports a consistent corporate-action-adjusted fractional-unit simulation, not literal historical share counts or a point-in-time execution database. VIX volume may be zero. Stocks must have positive volume. VIX-only dates outside SPY's equity calendar are excluded and recorded in metadata. Missing equity-session observations or invalid prices stop the workflow rather than being filled.
'''),
        code('''from stockml.data import prepare_data
AS_OF = None  # Optional explicit completed calendar date, e.g. '2026-10-07'.
metadata = prepare_data(as_of=AS_OF)
display(pd.Series({key: metadata[key] for key in ['downloaded_at_utc', 'last_completed_session', 'synthetic', 'adjustment', 'excluded_vix_non_equity_dates']}))
display(pd.Series(metadata['versions'], name='installed_version'))'''),
        md('''## Development-only exploratory analysis

Prices and returns can have changing volatility, heavy tails, and strong cross-stock dependence. Predictable volatility does not imply predictable return direction. The eight retrospectively chosen stocks also introduce survivorship and selection bias. Interpret the plots as descriptive evidence, not proof that a profitable pattern exists.
'''),
        code('''from stockml.reporting import plot_eda
plot_eda()'''),
        md(r'''## Feature and target definitions

All features use observations available at the signal close. Returns are fractions (0.01 means 1%). Rolling standard deviations use `ddof=1`. RSI uses Wilder-style exponential smoothing with `alpha=1/14`, `adjust=False`, and a 14-observation warm-up. MACD is the 12-span minus 26-span exponential moving average, divided by the current close. No scaling or imputation is fitted here.

| Category | Predictors |
|---|---|
| Momentum | 1-, 5-, 10-, 20-session returns |
| Trend | Close / 5- and 20-session mean close minus one |
| Volatility | 5- and 20-session standard deviation of daily returns |
| Volume | Volume percentage change; volume / 20-session mean volume |
| Technical | RSI-14; normalized MACD |
| Market context | SPY 1- and 5-session returns; VIX level and percentage change |
| Relative performance | Stock 5-session return minus SPY 5-session return |

The return target is $P_{t+5}/P_t-1$. Future risk is the sample standard deviation of returns on sessions $t+1,\ldots,t+5$. Both require all five future observations. Keep a `target_end` date for leakage checks. High-risk thresholds are fitted in Notebook 2, never here.

Rolling features are calculated continuously across year boundaries. Historical outcomes extending into 2026 are blanked before export. Inference rows with unknown future outcomes remain available.
'''),
        code('''from stockml.common import FEATURES, read_table
features = read_table('history_features')
outcomes = read_table('history_outcomes')
assert len(FEATURES) == 17
assert not features.duplicated(['date', 'ticker']).any()
assert outcomes.target_end.dropna().max() < pd.Timestamp('2026-01-01')
display(features[features.date < '2024-01-01'].head())
display(outcomes[outcomes.date < '2024-01-01'].head())
print('Saved historical and holdout features, outcomes, and prices separately; SHA-256 fingerprints recorded.')'''),
        md('''## Handoff

`data/history_{features,outcomes,prices}.parquet` contains the historical experiment inputs. `data/holdout_{features,outcomes,prices}.parquet` contains 2026 inputs. `data/metadata.json` records the snapshot, definitions, versions, and fingerprints.

Do not inspect holdout outcomes to modify features. Continue with Notebook 2.
'''),
    ],
    '02_training_inference.ipynb': [
        md('''# 2. Model development, frozen selection, and inference

This notebook owns all model fitting and prediction generation. It first completes **validation-only portfolio selection** using the shared simulator, then freezes the entry before assignment evaluation or competition fitting. This ordering permits all three notebooks to run sequentially while preserving independence.

The assignment refit uses complete outcomes before January 2024. The separate competition refit uses complete outcomes before January 2026. No 2026 row enters fitting, tuning, risk thresholds, feature importance, or strategy selection.
'''),
        md(COLAB_GUIDE),
        code(setup(2)),
        md('''## Development protocol and bounded search

Split all stocks by calendar date, not stacked row number. Drop any training row whose target reaches the next period. Validation scoring excludes targets realized in 2024. Return models minimize validation RMSE; risk models maximize ROC-AUC.

- Random Forest: 300 trees; `max_depth` in `[4, 8, None]`, `min_samples_leaf` in `[10, 30]`.
- XGBoost: 300 trees; `max_depth` in `[2, 3, 5]`, `learning_rate` in `[0.03, 0.1]`.
- Both tasks use seed 42. Classification uses probability cutoff 0.5.
- Pooled models add a fixed one-hot ticker encoding. Individual models reuse each algorithm/task's selected pooled settings to bound compute; this is not a fully tuned individual-model comparison.
- High risk means future volatility strictly exceeds that stock's training 75th percentile. Each refit recomputes thresholds; compare models within the same experiment.

Baselines predict zero return and the training majority risk class. Zero-return directional accuracy counts a prediction correct only when the realized return is exactly zero; its rank correlation is undefined (N/A). Majority baseline AUC uses a constant training prevalence probability. Undefined single-class AUC is N/A; undefined precision/recall/F1 are reported as zero.
'''),
        code('''from stockml.models import develop_and_freeze
frozen = develop_and_freeze()
display(pd.read_csv(root() / 'reports' / 'tuning.csv'))
validation_metrics = pd.read_csv(root() / 'reports' / 'validation_metrics.csv')
display(validation_metrics[validation_metrics.ticker == 'ALL'])
display(pd.read_csv(root() / 'reports' / 'validation_portfolios.csv'))
display(Markdown(f"Frozen entry: **{frozen['selected_entry']['strategy']}**. Freeze ID: `{frozen['freeze_id']}`."))'''),
        md('''## Interpretation using validation data only

The four pooled charts show the ten largest permutation importances. Positive importance means shuffling the feature harmed validation performance; negative values can reflect noise or overfitting. Correlated indicators can share importance, so importance is not a causal claim.

The stock-specific table reports each model's three leading training impurity/gain importances. These are useful for comparing feature identities across stocks but are not numerically comparable to permutation scores. The ablation compares momentum plus ticker encoding with all features using identical rows and settings. More features help only if the validation result improves.
'''),
        code('''from stockml.reporting import plot_importance
importance = plot_importance()
ablation = pd.read_csv(root() / 'reports' / 'feature_ablation.csv')
display(ablation)
from stockml.reporting import feature_discussion
feature_discussion()
display(validation_metrics[validation_metrics.ticker != 'ALL'])'''),
        md('''## Assignment experiment: 2024–2025

Settings and portfolio rules are now fixed. Refit through 2023 using only targets completed before 2024. Predictions include the preceding close so the portfolio can trade at the first 2024 session's open. Predictive scoring itself uses only signal dates within the test period and complete targets ending within it.

Model artifacts include feature order, ticker encoding, thresholds, cutoffs, parameters, library versions, and dataset fingerprints. Every save is checked by reloading the trusted local artifact and comparing predictions exactly.
'''),
        code('''from stockml.models import fit_final
assignment_bundle = fit_final('assignment')
assignment_metrics = pd.read_csv(root() / 'reports' / 'assignment_metrics.csv')
display(assignment_metrics[assignment_metrics.ticker == 'ALL'])
display(assignment_metrics[assignment_metrics.ticker != 'ALL'])'''),
        md('''## Competition refit: through 2025, with no 2026 data access

Refit the already chosen configurations on complete outcomes through 2025. This creates a separate model bundle and fresh training-only volatility thresholds. It does not select a new strategy based on the assignment results.
'''),
        code('''competition_bundle = fit_final('competition')
display(pd.Series({key: competition_bundle[key] for key in ['bundle_id', 'last_training_signal', 'last_training_target', 'freeze_id']}))'''),
        md('''## Final holdout inference: 2026

This is the first model-stage access to 2026 features. Predictions are saved before realized outcomes are loaded for scoring. Recent rows without five subsequent sessions still receive predictions but do not enter predictive metrics.

Running this cell exposes the holdout results. Later changes to features, models, or strategy selection would make 2026 development data rather than a fully independent test. A repeated frozen evaluation is not a new independent experiment.
'''),
        code('''from stockml.models import competition_inference
competition_predictions, competition_metrics = competition_inference(competition_bundle)
display(competition_metrics[competition_metrics.ticker == 'ALL'])
display(competition_metrics[competition_metrics.ticker != 'ALL'])
display(pd.Series(load_json(root() / 'competition_evaluation.json')))'''),
        md('''## Handoff

`models/` holds the trusted local model bundles and readable metadata. `predictions/` holds separate validation, assignment, and competition predictions. `frozen_experiment.json` identifies the entry and rules selected on validation. Predictive metrics, tuning, importance, and ablation tables are under `reports/`.

No inference code in Notebook 3 refits these models. Continue there for portfolio evaluation and evidence-based conclusions.
'''),
    ],
    '03_portfolio_evaluation.ipynb': [
        md('''# 3. Portfolio evaluation and scientific discussion

Load the saved predictions and frozen selection. This notebook does not train models or choose a new competition entry. Report the assignment and competition experiments separately.
'''),
        md(COLAB_GUIDE),
        code(setup(3)),
        code('''from stockml.models import check_frozen
frozen = check_frozen()
display(pd.Series(frozen['selected_entry']))
display(pd.read_csv(root() / 'reports' / 'validation_portfolios.csv'))'''),
        md('''## Allocation, timing, costs, and accounting

| Strategy | Rule |
|---|---|
| Equal weight | One-eighth in each stock; rebalance every five sessions |
| Return-ranked ML | One-third in each of the three highest predicted-return stocks |
| Risk-aware ML | Exclude risk probability ≥0.5, rank the rest, and fill up to three one-third slots |
| SPY | Buy at the first execution open and hold |

Missing risk-aware slots remain in cash. Ties use alphabetical ticker order. Both algorithms and both pooled/individual layouts are reported. Only the entry selected by validation net Sharpe (then lower turnover, then strategy name) is the competition entry.

Each strategy starts with $100,000, uses fractional adjusted units, no leverage, no cash interest, and 10 basis points per dollar bought or sold. Signals from a completed close execute at the next session's open. Rebalancing uses drifted holdings and solves for post-cost wealth before allocating, so costs cannot create borrowing.

Track close-to-close portfolio values daily, including the initial allocation cost and first open-to-close return. Annualize with 252 sessions; use a zero risk-free rate for Sharpe. Drawdown includes the initial $100,000 peak. Terminal wealth is marked to market without liquidation. Turnover is total absolute dollars traded divided by pre-trade wealth at each rebalance; it is a two-sided, cumulative measure.

The required five-day return target is close-to-close; executable portfolio returns start at the following open. They are deliberately not treated as identical. Final portfolio valuation may include sessions whose five-day predictive targets are still unknown.
'''),
        code('''from stockml.reporting import report_portfolios, evidence_discussion
assignment_performance, assignment_daily, assignment_trades = report_portfolios('assignment')'''),
        code("evidence_discussion('assignment')"),
        md('''## Independent competition results

The following chart and table report the frozen entry alongside all predeclared comparators. Do not replace the entry with whichever comparator wins on 2026. A partial-year result is labeled year-to-date and includes its final completed session.
'''),
        code("competition_performance, competition_daily, competition_trades = report_portfolios('competition')"),
        code("evidence_discussion('competition')"),
        md('''## Interpretation checklist and limitations

Use the generated evidence above and Notebook 2's importance/ablation tables to defend these conclusions:

1. Which algorithm wins on return RMSE and risk ROC-AUC? Are the winners consistent across stocks?
2. Which task shows more reliable improvement over its own baseline? Do not compare RMSE units with AUC units.
3. Does pooling help each algorithm, given that individual models reuse pooled tuning settings?
4. Does the **frozen** entry outperform SPY after costs, and what happens to drawdown and volatility?
5. Do better predictions correspond to better ranked portfolios? Use the joined prediction/investment table.
6. Does the evidence support generalization across both periods? A short holdout and overlapping labels do not establish statistical significance.
7. Which financial features matter on validation? Compare top-feature identities and the feature ablation before asserting that additional features helped.

The model universe was selected retrospectively and is concentrated in large surviving companies. Regime shifts, Yahoo data revisions, simplified fills, fixed costs, and zero cash yield constrain interpretation. A longer preregistered forward test, point-in-time universe, block-bootstrap uncertainty intervals, and richer execution assumptions would strengthen a future experiment. Do not tune these changes on 2026 after seeing its results.

A successful project can conclude that the models did not beat simple baselines.
'''),
        md('''## AI assistance and independent verification

AI assistance was used to structure the experiment, draft the shared Python implementation, and create these notebooks and test cases. AI-generated code and conclusions require independent review.

The verification suite checks future-price perturbations, five-day targets against hand calculations, split purging, feature-order rejection, saved-model prediction parity, trade timing, and portfolio accounting against hand-computed constant-price examples. In particular, the transaction-cost implementation uses post-cost wealth rather than allocating all pre-cost cash and implicitly borrowing the fees.

Before submission, run `python -m pytest -q`, inspect these calculations yourself, and add a sentence describing a suggestion or implementation **you personally verified or corrected**. Do not claim personal verification solely because automated tests pass.
'''),
    ],
}

for name, cells in notebooks.items():
    cells[0].source += ('\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]'
                        f'(https://colab.research.google.com/github/terminal34-datascience-bootcamp/stock-market-ml/blob/main/{name})')
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'colab': {'name': name, 'provenance': []}}
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / name)
    print(name)
