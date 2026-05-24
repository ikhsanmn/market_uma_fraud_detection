# IDX Pump-and-Dump Detection — ML vs. Deep Learning

Detecting pump-and-dump market manipulation on the **Indonesia Stock Exchange (IDX)** using classical machine learning and deep learning, evaluated on IDX Unusual Market Activity (UMA) announcements from 2021–2026.

---

## What This Project Does

This project trains and compares **6 models** — 3 classical ML and 3 deep learning — to classify (ticker, date) windows as either legitimate trading or a pump-and-dump manipulation event. Ground-truth labels come from official IDX UMA announcements. The primary evaluation metric is **MCC (Matthews Correlation Coefficient)**, chosen because it is robust to the severe class imbalance inherent in fraud detection (~2.8% positive rate in the test set).

| Group | Models |
|---|---|
| Classical ML | Logistic Regression, Random Forest, LightGBM |
| Deep Learning | BiLSTM, CNN-LSTM, Transformer |

---

## Project Structure

```
fraud detection ml/
├── 01_pipeline.ipynb          # Full pipeline: data → features → train → save
├── 02_results.ipynb           # Evaluation: metrics, plots, ML vs DL comparison
├── requirements.txt           # Python dependencies
├── market_manipulation_detection_research.md  # Research context & methodology
│
├── data/
│   ├── raw/
│   │   ├── ohlcv_daily/       # Per-ticker daily OHLCV CSVs (downloaded by pipeline)
│   │   ├── ohlcv_intraday/    # Intraday minute bars (if available)
│   │   ├── uma_events.csv     # UMA announcement labels (REQUIRED — see below)
│   │   ├── idx_tickers.txt    # Full IDX ticker list (optional)
│   │   └── failed_tickers.txt # Tickers yfinance could not retrieve
│   └── processed/
│       ├── features.csv       # Full feature matrix
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
│
├── models/                    # Saved model artifacts
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── lightgbm.pkl
│   ├── bilstm.keras
│   ├── cnn_lstm.keras
│   └── transformer.keras
│
└── results/                   # Evaluation outputs
    ├── ml_predictions.csv
    ├── dl_predictions.csv
    ├── metrics_summary.csv
    ├── confusion_matrices.png
    ├── pr_curves.png
    ├── mcc_threshold_sweep.png
    ├── feature_importance.png
    ├── ml_vs_dl_comparison.png
    └── latency.png
```

> **Note:** `data/`, `models/`, `results/*.csv`, and `venv/` are excluded from version control via `.gitignore`. You need to run the pipeline to regenerate them.

---

## Requirements

- Python 3.11
- ~8 GB disk space for the full IDX ticker download
- CPU is sufficient; a GPU reduces DL training time but is not required

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `yfinance`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `tensorflow`, `keras`, `ta`, `pandas`, `matplotlib`, `seaborn`.

---

## Setup

### 1. Clone / download the project

```bash
git clone <repo-url>
cd "fraud detection ml"
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Provide the UMA label file

The pipeline requires `data/raw/uma_events.csv`. This file must contain the IDX Unusual Market Activity announcements with at least two columns:

| Column | Description |
|---|---|
| `ticker` | Stock ticker in `.JK` format (e.g., `BBCA.JK`) |
| `uma_date` | Announcement date (parseable by pandas, e.g., `2022-07-14`) |

This file is not included in the repository because it requires manual collection from [idx.co.id](https://idx.co.id). Scraping helpers are in `data/raw/` (`download_uma.py`, `find_api*.py`).

### 4. (Optional) Provide the full ticker list

Place all IDX common-stock tickers (one per line, `.JK` suffix) in `data/raw/idx_tickers.txt`. If absent, the pipeline falls back to only the tickers present in `uma_events.csv`.

---

## How to Run

Run the notebooks in order inside Jupyter Lab or VS Code.

```bash
jupyter lab
```

### Step 1 — `01_pipeline.ipynb`

Runs the full pipeline end-to-end:

1. Loads UMA labels
2. Downloads daily OHLCV for all IDX tickers via `yfinance` (batched, ~30–60 min for full universe)
3. Fetches the IHSG benchmark (`^JKSE`)
4. Performs exploratory analysis (UMA timeline, price moves around events)
5. Engineers 15 features per (ticker, date): returns, volume activity, volatility, RSI, moving-average ratio, price acceleration, temporal encoding
6. Labels each window using a ±5-day event window around each UMA date
7. Applies a **time-based split**: train = 2021–2024, test = 2025+
8. Fits `StandardScaler` on the training set and applies SMOTE (training fold only)
9. Trains all 6 models and saves artifacts to `models/` and `results/`

### Step 2 — `02_results.ipynb`

Loads saved predictions and produces all evaluation outputs:

- Confusion matrices for all 6 models at their optimal F1 threshold
- Full metric table (MCC, F1, Precision, Recall, Specificity, Balanced Accuracy, PR-AUC)
- Precision-Recall curves with no-skill baseline
- MCC vs. threshold sweep (0.05 → 0.95)
- Feature importance bar charts (Random Forest + LightGBM)
- ML vs. DL grouped comparison bars
- Inference latency benchmark (ms per 1,000 samples)

All charts are saved to `results/` as PNG at 150 DPI.

---

## Results

Evaluated on the held-out 2025 test set (~165,900 samples, ~2.8% positive rate). Thresholds are tuned to maximize F1 per model.

| Model | Group | Threshold | MCC | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Random Forest** | ML | 0.35 | **0.3405** | **0.3583** | 0.3149 | 0.4154 | 0.2647 |
| LightGBM | ML | 0.95 | 0.2545 | 0.2234 | 0.1352 | **0.6430** | **0.2682** |
| CNN-LSTM | DL | 0.95 | 0.3061 | 0.3240 | **0.3436** | 0.3066 | 0.2500 |
| Transformer | DL | 0.90 | 0.2387 | 0.2579 | 0.2076 | 0.3405 | 0.1656 |
| Logistic Regression | ML | 0.90 | 0.2154 | 0.2382 | 0.2063 | 0.2818 | 0.1350 |
| BiLSTM | DL | 0.05 | 0.1756 | 0.2003 | 0.1741 | 0.2358 | 0.1187 |

**Key findings:**

- **Random Forest is the best overall model** by MCC (0.34) and F1 (0.36). Engineered tabular features already capture the dominant pump-and-dump signals, and deep learning's added complexity is not justified on this dataset at this scale.
- **LightGBM achieves the highest recall (64%)** — useful when missing a manipulation event is the worst outcome — but at the cost of very low precision (14%), generating many false alarms.
- **CNN-LSTM is the strongest deep learning model**, matching Random Forest in MCC (0.31 vs 0.34) and achieving the highest precision (34%) across all DL models.
- **BiLSTM underperformed** relative to its architecture size; the optimal threshold collapsed to 0.05, indicating the model never learned confident P&D probabilities.
- **All PR-AUC values are low (0.12–0.27)**, reflecting the true difficulty of this problem — the no-skill baseline is ~0.028 (the positive rate), so even 0.27 is a meaningful lift.
- **Inference latency**: ML models are 10–100× faster on CPU than DL models, making them more practical for near-real-time surveillance.

---

## What This Project Aims to Achieve

1. **Answer the ML vs. DL question empirically on IDX equities.** Most published pump-and-dump research uses cryptocurrency data. This project applies the same comparison to a regulated Southeast Asian stock exchange, contributing evidence on whether DL complexity is justified in equity market surveillance.

2. **Build a reproducible, leakage-safe benchmark.** The time-based train/test split, SMOTE applied only to the training fold, and the full confusion-matrix reporting framework are designed to avoid the most common methodological errors in imbalanced classification studies.

3. **Inform practical surveillance tooling.** By measuring both accuracy (MCC) and latency, the project frames the ML-vs-DL comparison in operational terms that matter to a market surveillance team, not just offline benchmark numbers.

4. **Lay the groundwork for an academic paper.** The research document (`market_manipulation_detection_research.md`) describes the full methodology, literature positioning, and threats to validity. The pipeline and results notebooks are designed to be directly referenced in the paper.

---

## Strengths

- **Rigorous evaluation framework.** MCC + full confusion matrix + PR-AUC at tuned thresholds — not misleading accuracy on an imbalanced test set.
- **Leakage-free design.** Time-based split; SMOTE and StandardScaler fitted exclusively on the training fold.
- **6-model apples-to-apples comparison.** All models trained on identical features and splits.
- **Latency awareness.** The results notebook benchmarks inference speed, keeping the comparison operationally grounded.
- **Documented threats to validity.** Label noise, survivorship bias, concept drift, and scope limitations are explicitly acknowledged in both the notebook and research document.
- **IDX-specific contribution.** The labeled IDX UMA dataset and equity-market benchmark fill a gap left by crypto-focused prior work.

## Limitations

- **Weak/noisy ground truth.** IDX UMA announcements signal *suspicion*, not a confirmed violation. Mislabeled windows directly corrupt every confusion-matrix cell and cannot be corrected post-hoc.
- **No order-book data.** Daily OHLCV and minute bars cannot capture individual order placements or cancellations, so spoofing is physically undetectable in this dataset and is excluded from scope.
- **Survivorship bias.** Delisted or suspended stocks — often the most heavily manipulated — may be missing from yfinance downloads. Stocks present in the UMA list but absent from the download are logged in `data/raw/failed_tickers.txt` for disclosure.
- **Concept drift.** Models are static snapshots trained on 2021–2024 patterns. Pump-and-dump coordination tactics (e.g., Telegram group strategies) evolve, and the model's effectiveness may degrade over time without retraining.
- **No social signal integration.** The strongest recent results in the literature (Gogol et al. 2025) fuse Telegram/social data with price signals. This project uses price/volume only, which is a ceiling on recall for coordinated schemes that leave no market footprint before the pump begins.
- **All metrics are moderate.** The highest MCC achieved is 0.34 — detectably better than random but far from deployment-ready. This reflects the genuine difficulty of the problem, not a pipeline error.

---

## Resources

### Dataset

| Dataset | Description | Link |
|---|---|---|
| UMA Events + IDX OHLCV | Labeled pump-and-dump windows for IDX common stocks 2021–2026, including UMA announcement dates, daily OHLCV, and engineered features | _coming soon_ |

---

## Citation / Research Context

For the full methodology, literature review, and design decisions, see `market_manipulation_detection_research.md`. Key references:

- Xu & Livshits (USENIX Security, 2019) — Random Forest baseline for P&D detection
- Chadalapaka et al. (arXiv:2205.04646, 2022) — CLSTM and Anomaly-Transformer for crypto P&D
- Gogol et al. (arXiv:2412.18848, 2025) — real-time P&D detection fusing social + order-book data


## Fixes Applied in 1.1 version
- M1 — 01_pipeline.ipynb cell-7 (single-ticker batch corruption)
Added not isinstance(raw.columns, pd.MultiIndex) guard so the manual column-wrapping only runs when yfinance actually returns flat columns — no-op for yfinance ≥ 0.2 which already returns MultiIndex.

- M2 — 01_pipeline.ipynb cell-17 (calendar days → trading days)
Replaced pd.Timedelta(days=EVENT_WINDOW[i]) with pd.offsets.BDay(EVENT_WINDOW[i]). The event window is now always exactly ±5 trading days regardless of what weekday the UMA falls on.

- M3 — 01_pipeline.ipynb new cell after section 9 + 02_results.ipynb cell-5 & cell-7 (test-set threshold leakage)
Added a new cell in the pipeline that computes optimal thresholds from validation data only: ML models use 5-fold TimeSeriesSplit out-of-fold predictions (with SMOTE inside each fold via ImbPipeline); DL models use the last 15% of their training sequences. Thresholds saved to results/optimal_thresholds.json.
02_results.ipynb now loads that JSON instead of searching the test set. optimal_threshold(name) takes the model name, not test labels.

- M4 — 02_results.ipynb cell-20 (stale train/test period text)
Updated "trained on 2021–2023 and tested on 2024" → "trained on 2021–2024 and tested on 2025–2026" throughout the Threats to Validity section.