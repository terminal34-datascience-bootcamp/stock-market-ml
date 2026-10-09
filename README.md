# Can Machine Learning Beat the Stock Market?

Three notebooks implement chronological Random Forest/XGBoost experiments, a 2024–2025 assignment test, and a separately frozen 2026 competition holdout. The repository includes shared, tested Python code; keep `stockml/` beside the notebooks.

## Run in Google Colab

Open these notebooks in order; choose a **Python 3 CPU runtime** (no GPU required):

| Step | Notebook |
|---|---|
| 1. Download and preprocess | [Open in Colab](https://colab.research.google.com/github/terminal34-datascience-bootcamp/stock-market-ml/blob/main/01_data_preprocessing.ipynb) |
| 2. Train and predict | [Open in Colab](https://colab.research.google.com/github/terminal34-datascience-bootcamp/stock-market-ml/blob/main/02_training_inference.ipynb) |
| 3. Evaluate portfolios | [Open in Colab](https://colab.research.google.com/github/terminal34-datascience-bootcamp/stock-market-ml/blob/main/03_portfolio_evaluation.ipynb) |

In each notebook, run the setup cell, authorize Google Drive, then run the remaining cells. Keep `EXPERIMENT_NAME = 'competition-v1'` and `COLAB_SMOKE_TEST = False` identical across all three. Set the smoke flag to `True` in all three for a fast synthetic trial instead.

**This GitHub repository is private.** In Colab's Secrets panel (key icon), add `GITHUB_TOKEN` and enable notebook access in each notebook. Use a fine-grained GitHub token with **Contents: Read-only** permission for this repository; the organization may need to approve it. The token is passed through a Git child-process environment, never stored in notebook cells, artifacts, or remote URLs. Without access, setup stops with instructions. If Colab cannot open the private GitHub notebook link, download the notebook from GitHub and use Colab's **Upload** tab instead; the secret is still needed to download the shared code. [GitHub fine-grained token instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

Setup automatically downloads the shared Python code and installs `requirements-colab.txt`. The first notebook saves its Git revision, Python version, and resolved package versions; subsequent sessions reuse that source revision and install those exact package versions. If pip changed an already-imported package, setup explicitly asks you to restart the session and rerun the setup cell. Do not install the macOS-generated `requirements-lock.txt` in Colab.

Artifacts are written directly to `My Drive/stock-market-ml/competition-v1/real/` (or `smoke/`). This folder is the handoff between independent notebook sessions. Finish each notebook before starting the next, and do not run two notebooks against the same experiment folder concurrently. Notebook 2 may take several minutes or longer. Completed frozen model bundles can be reused after a reset; an interrupted, unfinished fitting stage may need to run again. Temporary model/metadata files are promoted only after successful writes.

Use **File → Save a copy in Drive** to retain notebook edits and displayed outputs; artifact persistence does not save the notebook itself. Colab runtimes are temporary, which is why the workflow mounts Drive for persistent files. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

Local artifacts are not uploaded automatically. The existing 2026 results have already been evaluated locally: rerunning on Colab is a reproduction, not another independent holdout. Keep the experimental choices fixed. Avoid changing the experiment name to try alternatives against 2026.

## Local setup and run

Python 3.11–3.13 is recommended. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_notebooks.py --smoke
```

The smoke run uses **synthetic prices**, eight-tree models, and one configuration per algorithm. It tests execution and accounting, not profitability. Executed smoke notebooks are saved under `artifacts-smoke/executed/`.

Run the actual experiment, including the final 2026 evaluation:

```sh
python scripts/run_notebooks.py
```

This downloads Yahoo Finance data and runs the full six-configuration searches plus pooled and individual models. It may take several minutes or longer. Real executed notebooks are saved under `artifacts/executed/`; the source notebooks remain editable and unexecuted. For interactive use, open each notebook in VS Code or Jupyter with `.venv` selected and run **1 → 2 → 3**, restarting the kernel between notebooks. Set `AS_OF` in Notebook 1 for an explicit historical cutoff. The optional runner `--artifacts PATH` chooses a separate snapshot directory.

On macOS, if XGBoost explicitly reports that `libomp` is missing, install the OpenMP runtime using your package manager and retry. No system installation is needed when the packaged runtime loads successfully.

## Notebook responsibilities

1. **01_data_preprocessing.ipynb** downloads and validates eight stocks, SPY, and VIX; creates 17 causal features and continuous targets; explores only development data; saves separate historical and 2026 datasets.
2. **02_training_inference.ipynb** tunes on 2022–2023, compares pooled/individual models, computes validation importance/ablation, and selects a strategy using validation portfolios. It freezes that choice **before** either final experiment, then fits assignment and competition bundles and exports predictions.
3. **03_portfolio_evaluation.ipynb** loads predictions, simulates next-open portfolios after transaction costs, and renders separate assignment/competition charts, tables, trade ledgers, and result-grounded discussion. It never retrains or selects a new winner.

The validation portfolio calculation in Notebook 2 resolves the dependency between strategy selection and competition training. Notebook 3 reuses the simulator and frozen selection.

## Experiment and artifact contract

| Phase | Fitting observations/outcomes | Evaluation |
|---|---|---|
| Development | 2015–2021, targets completed before 2022 | 2022–2023, targets completed before 2024 |
| Assignment | Through 2023, targets completed before 2024 | 2024–2025 |
| Competition | Through 2025, targets completed before 2026 | 2026, through the recorded completed session |

All splits group stocks by date. Unknown final targets do not prevent inference. Signals include the prior year's last close for next-open execution on the first test session. All strategies share an execution calendar and terminal valuation date. A partial 2026 year is explicitly labeled year-to-date.

Within each artifact directory:

- `data/`: separate historical/holdout features, outcomes, adjusted OHLCV prices, and SHA-256 snapshot metadata.
- `models/`: joblib bundles plus JSON metadata, ordered features, encoding, training thresholds, cutoffs, parameters, seeds, versions, and dataset fingerprints. Only load trusted local joblib files.
- `predictions/`: date, ticker, predicted return, high-risk probability, model, experiment, and bundle ID. No realized outcomes.
- `frozen_experiment.json`: validation-selected entry, rules, hyperparameters, and historical fingerprints.
- `reports/`: tuning, metrics, importance, ablation, daily portfolios, and trade ledgers.
- `competition_evaluation.json`: final completed session and holdout provenance.

The default snapshot is immutable on rerun: files are checked against their fingerprints. No automatic synthetic fallback is used if downloading fails. VIX-only dates outside SPY's equity calendar are excluded and recorded in metadata; no stock trading session is removed. Missing equity-session observations raise an explicit error; investigate the source rather than silently filling or compressing trading horizons.

The full environment versions are recorded in the artifacts. `requirements.txt` specifies supported major-version ranges; `requirements-lock.txt`, when present, records the exact tested environment.

## Interpretation and integrity

The entry maximizes validation portfolio Sharpe after costs, breaking ties by lower turnover and then name. All eight predeclared ML portfolios remain visible as comparisons. Selecting a replacement entry after viewing 2026 would invalidate independence. An expanded snapshot or repeated evaluation does not create a new untouched test. This is a reproducibility workflow, not an adversarial tamper-proof system.

Prices use consistent Yahoo-adjusted OHLC units and no separate dividend cash flows. Execution is an educational approximation. Long-only fractional portfolios pay 10 bps per dollar bought/sold, solve for affordable post-cost allocations, retain empty risk slots in cash, and are marked daily without terminal liquidation. Sharpe assumes zero risk-free rate; cash earns zero. Annualization uses 252 sessions. The close-to-close label differs from the next-open executable holding interval.

Read the generated discussion alongside the tables before making causal or statistical claims. No market-beating result is assumed. Add your own interpretation and personally verified example to the AI-assistance disclosure before submission. Three notebooks are assumed to be permitted by the instructor; they depend on the included `stockml/` package rather than being three standalone source files.

## Maintenance

The notebook source templates were generated by `scripts/build_notebooks.py`. That script **overwrites** the source notebooks; do not rerun it over your annotations. Edit the notebooks directly for your final submission. Generated market data, models, and executed notebooks are git-ignored.
