# Detecting Pump-and-Dump Manipulation on the Indonesia Stock Exchange (IDX) Using Machine Learning and Deep Learning: A Research Foundation

> **Scope of this draft.** Binary fraud / non-fraud classification of **pump-and-dump** events on **IDX-listed common stocks (2021–2024)**, comparing classical machine learning (ML) against deep learning (DL), validated through the **confusion matrix** (primary metric: **MCC**). Labels are derived from IDX **Unusual Market Activity (UMA)** announcements. Spoofing is out of scope (see §6.1). Finalized design decisions are in §6; full methodology in §7.
>
> *Sections 1–5 retain broader market-manipulation context (including spoofing) for the literature review; the project itself is scoped to pump-and-dump per §6.*

---

## 1. Introduction

Financial markets depend on the assumption that prices reflect genuine supply and demand. **Market manipulation** breaks that assumption by deliberately creating artificial price or volume signals to mislead other traders. Two of the most common and well-studied forms are:

- **Pump-and-dump (P&D)** — coordinated groups (increasingly organized over Telegram and Discord) artificially inflate the price of a thinly traded asset, then sell into the demand they manufactured, leaving later buyers with losses. The phases are typically an *announcement*, a *countdown* to build anticipation, and a *release/pump* that triggers a rapid price surge. A Chainalysis report cited across the recent literature found that roughly **24% of tokens launched in 2022** displayed characteristics typical of P&D schemes, illustrating the systemic scale of the problem in crypto markets.
- **Spoofing** — a trade-based manipulation where a trader places large limit orders with no intent to execute, in order to create a false impression of supply or demand, then cancels them once the market moves favorably. Spoofing is harder to detect than P&D because the manipulative intent is hidden inside ordinary-looking order-book activity.

The detection task is naturally framed as a **supervised binary classification problem**: each observation (a coin over a time window, an order, an order-book snapshot, or a trader-time slice) is labeled *manipulative* or *legitimate*, and a model must separate the two. Because the predictions are categorical, the **confusion matrix** is the foundational evaluation device — every metric this research will rely on (precision, recall, F1, specificity, balanced accuracy, MCC) is derived from its four cells. This makes the confusion matrix not just a reporting convenience but the central lens through which the ML-vs-DL comparison is conducted.

**Why ML vs DL specifically.** Classical ML models (Random Forest, XGBoost, SVM, Logistic Regression) are interpretable, cheap to train, and strong on engineered tabular features — they dominated the early manipulation-detection literature. Deep learning models (LSTM/CLSTM, Transformers, anomaly-attention architectures, graph neural networks) can learn temporal and relational patterns directly from raw sequences and order-book data, and recent work reports they outperform classical baselines on the same tasks. The open question this research targets is **when, and by how much, the added complexity of DL is justified** for manipulation detection — measured rigorously through the confusion matrix.

---

## 2. The Problem

### 2.1 Problem statement

> Given market data (trade/price/volume series, limit-order-book events, and/or social-channel messages), build and **compare** ML and DL classifiers that label market activity as **manipulative (positive class)** or **legitimate (negative class)**, and evaluate them using confusion-matrix-derived metrics under realistic, severely imbalanced conditions.

### 2.2 Why this is hard (the core tensions)

1. **Extreme class imbalance.** Genuine manipulation events are rare relative to normal trading — fraud rates in adjacent financial-fraud domains run well below 1% (e.g., reports cite fraud as low as ~0.015% of card payments). A model that predicts "never fraud" can score >99% accuracy while being useless. This is precisely why **accuracy is misleading** and the confusion matrix matters.
2. **The precision–recall trade-off has real costs.** A **false negative** is a missed manipulation (market integrity harmed, investors hurt). A **false positive** is a wrongly flagged legitimate trader (regulatory cost, eroded trust, wasted analyst time). The "right" operating point is a policy decision, not a purely statistical one.
3. **Weak / scarce ground truth.** True labels require prosecuted cases or expensive manual annotation, both of which are scarce. Many studies rely on heuristic or proxy labels, which inject **label noise** that the confusion matrix cannot see.
4. **Concept drift.** Manipulators adapt; patterns that defined fraud last year may not next year. A model frozen at training time silently degrades.
5. **Real-time constraints.** Detecting a pump *after* it completes is academically interesting but operationally too late; useful systems must classify in near real time, which constrains model complexity.

### 2.3 The confusion matrix as the evaluation backbone

For the positive class = *manipulation*:

|  | Predicted: Manipulation | Predicted: Legitimate |
|---|---|---|
| **Actual: Manipulation** | True Positive (TP) | False Negative (FN) — *missed fraud* |
| **Actual: Legitimate** | False Positive (FP) — *false alarm* | True Negative (TN) |

Metrics this research will report (and *why*, given imbalance):

- **Precision** = TP / (TP + FP) — of everything we flagged, how much was real fraud (controls alarm fatigue).
- **Recall / Sensitivity** = TP / (TP + FN) — of all real fraud, how much we caught (controls missed fraud).
- **F1-score** = harmonic mean of precision and recall — single headline number under imbalance.
- **Specificity** = TN / (TN + FP) — how well legitimate activity is preserved.
- **Balanced Accuracy** = mean of recall and specificity — replaces plain accuracy under imbalance.
- **Matthews Correlation Coefficient (MCC)** — arguably the most honest single scalar for imbalanced confusion matrices, since it uses all four cells.
- **PR-AUC** (area under precision–recall curve) — preferred over ROC-AUC when positives are rare.

> **Methodological note for the eventual paper:** report the *full confusion matrix* per model, not just the summary scalars, and fix a defensible decision threshold (or sweep it) rather than defaulting to 0.5 — under heavy imbalance the default threshold almost always under-detects the minority class.

---

## 3. State of the Research So Far (Recent Literature)

The field has moved through three rough generations: **(a)** post-hoc statistical anomaly detection, **(b)** classical ML on engineered features, and **(c)** deep learning on sequences, order books, and graphs. The newest work (2024–2026) is increasingly **real-time**, **order-book-aware**, and **Transformer/representation-learning-based**.

### 3.1 Pump-and-dump detection

- **Xu & Livshits (USENIX Security, 2019)** — a foundational classical-ML approach: **Random Forest** models predict the likelihood that a given coin is the next pump target, using market metrics. Establishes the supervised, feature-engineered baseline that later DL work is measured against.
- **La Morgia et al. (2020)** — anomaly-detection framing of crypto P&D; widely used as a comparison baseline.
- **Chadalapaka et al. (2022), "Crypto Pump and Dump Detection via Deep Learning Techniques" (arXiv:2205.04646)** — first to apply **CLSTM and Anomaly-Transformer** architectures to crypto P&D, reporting that **DL solutions significantly outperform prior P&D detection methods**, including Random Forest baselines, for small-cap volatile coins. Open-source code exists (CLSTM, TransformerTimeSeries, AnomalyTransformer variants).
- **Hu et al. (2023)** — **sequence-based deep-learning** models that leverage channel-specific (per-Telegram-group) features, advancing beyond market-only inputs.
- **Gogol et al. (DeFi'25 / arXiv:2412.18848, 2024–2025), "Machine Learning-Based Detection of Pump-and-Dump Schemes in Real-Time"** — a current state-of-the-art *operational* pipeline: uses **NLP to classify Telegram messages** (identifying 2,079 historical pump events), and crucially **fuses high-frequency order-book and trade data** with social signals. Their Z-score-based model forecasts target coins **seconds before** the pump, ranking the correct coin in the top-5 in ~56% of cases. Directly addresses the prior limitation that earlier methods relied only on historical data and lacked real-time applicability.
- **Atugoda et al. (2026), "Enhancing Stock Market Surveillance Using Machine Learning Algorithms for Fraud Detection"** — extends ML surveillance framing to (traditional) stock markets; useful if you move beyond crypto.

### 3.2 Spoofing & order-book manipulation detection

- **Cartea et al. (2023), "Spoofing and Manipulating Order Books with Learning Algorithms"** — uses learning algorithms (RL-flavored) to *model* how spoofing agents manipulate books; informs feature design for detection.
- **Poutré, Chételat & Morales (2024), "Deep Unsupervised Anomaly Detection in High-Frequency Markets"** — DL anomaly detection that does not require scarce manipulation labels; important given the ground-truth problem.
- **Fabre & Muni Toke (2025, arXiv:2502.04027), "High-Frequency Market Manipulation Detection with a Markov-modulated Hawkes process"** — point-process model capturing the bursty temporal structure of manipulative order flow.
- **Fabre & Challet (2025, arXiv:2504.15908), "Learning the Spoofability of Limit Order Books With Interpretable Probabilistic Neural Networks"** — trains an interpretable **probabilistic neural network** on **Level-3 order-book data** with novel multi-scale Hawkes order-flow features, and frames detection around the *expected manipulation gain* of a spoofing agent. Notably finds **~31% of large orders could spoof the market** in their sample, and the model is light enough to run in real time. Emphasizes that ignoring the **posting distance** of limit orders makes a detector inadequate — a concrete feature-engineering lesson.
- **"Detecting Multilevel Manipulation from Limit Order Book via Cascaded Contrastive Representation Learning" (ICLR 2026 submission, OpenReview)** — most recent: argues spoofing is wrongly simplified as single-level manipulation, proposes **contrastive representation learning** across multiple price levels, with **Transformer architectures achieving state-of-the-art** detection.

### 3.3 Cross-cutting reviews and the imbalance/evaluation literature

- **"Year-over-Year Developments in Financial Fraud Detection via Deep Learning: A Systematic Literature Review" (arXiv:2502.00201, 2025)** — surveys CNN, LSTM, and Transformer models across financial-fraud domains; consolidates precision/recall/F1/AUC-ROC as the standard metric set and flags **imbalanced data, model interpretability, and ethics** as recurring challenges.
- **"Advancing Machine Learning for Financial Fraud Detection: A Comprehensive Review" (2025)** — finds **feature engineering yields the largest performance gains** across nearly all models, and surveys SMOTE, ensemble methods, and GANs (K-CGAN) for imbalance.
- **Empirical studies on evaluation metrics under massive imbalance (e.g., arXiv:2208.11904)** — show there is *no universally agreed* evaluation metric for extreme imbalance, and that human annotation error compounds the problem — reinforcing why this research must report the full confusion matrix and use imbalance-robust scalars (MCC, PR-AUC).

### 3.4 Quick comparison of the recent landscape

| Work (year) | Manipulation type | ML / DL | Data modality | Headline finding |
|---|---|---|---|---|
| Xu & Livshits (2019) | Pump-and-dump | ML (Random Forest) | Market metrics | Baseline target-coin prediction |
| Chadalapaka et al. (2022) | Pump-and-dump | DL (CLSTM, Anomaly Transformer) | Price/volume sequences | DL beats classical baselines |
| Hu et al. (2023) | Pump-and-dump | DL (sequence) | + channel features | Group-specific signals help |
| Gogol et al. (2024–25) | Pump-and-dump | ML + NLP | Telegram + order book | Real-time, pre-pump prediction |
| Poutré et al. (2024) | Order-book anomaly | DL (unsupervised) | High-freq LOB | No labels required |
| Fabre & Muni Toke (2025) | HF manipulation | Hawkes / statistical | Order flow | Models temporal bursts |
| Fabre & Challet (2025) | Spoofing | DL (probabilistic NN) | Level-3 LOB | Interpretable, real-time, posting-distance matters |
| Multilevel Contrastive (2026) | Spoofing (multilevel) | DL (Transformer + contrastive) | LOB | Multilevel SOTA |

---

## 4. Compiled Problems / Challenges the Field Faces

Synthesized from the sources above — these are the gaps your research can position itself against.

1. **Severe class imbalance.** Manipulation is a tiny minority class; naive training and accuracy-based evaluation both fail. *Mitigations in the literature:* SMOTE and variants, GAN/K-CGAN synthetic minority generation, ensemble methods, cost-sensitive learning. *Open issue:* these can inflate apparent recall while hurting precision, so they must be judged on the confusion matrix, not accuracy.

2. **Scarce and noisy ground truth.** Reliable labels need prosecuted cases or costly manual annotation; many datasets use heuristic/proxy labels. Annotation error directly corrupts every confusion-matrix cell. This is pushing the field toward **unsupervised / semi-supervised anomaly detection** (e.g., Poutré et al.).

3. **Lack of standardized evaluation.** No agreed benchmark metric for extreme imbalance; papers report differing metric sets, making cross-study comparison unreliable. *Implication for you:* commit to a fixed, fully-reported metric suite (full confusion matrix + F1 + MCC + PR-AUC) and a fixed threshold policy.

4. **Real-time / latency requirements.** Post-event detection is too late operationally. The newest work prizes models light enough to run on streaming order-book data — which can disadvantage heavy DL architectures and reframes the ML-vs-DL question around the *accuracy-vs-latency* trade-off.

5. **Data access and modality fusion.** The strongest recent results fuse modalities (social messages **+** high-frequency order book), but **Level-2/Level-3 order-book data is expensive and often proprietary**, and social data raises privacy issues. Most public reproducible work is therefore in crypto, not regulated equities.

6. **Concept drift / adversarial adaptation.** Manipulators change tactics; static models decay. Few studies evaluate temporal robustness or adversarial resilience.

7. **Interpretability and regulatory acceptance.** A flag that cannot be explained is hard to act on legally. This drives interest in interpretable DL (e.g., Fabre & Challet's probabilistic NN) over black-box models — a genuine point in classical ML's favor.

8. **Crypto-vs-equity transferability.** Most public P&D/spoofing research is on crypto due to data availability and volatility; whether models transfer to regulated stock markets is largely untested.

---

## 5. Suggested Next Steps for This Research

- **Lock the target market and data source** (crypto exchange order book vs. equity market vs. simulated LOB). This is the single biggest determinant of everything downstream.
- **Decide the labeling strategy** (prosecuted cases, heuristic rules, or unsupervised) and document its noise risk.
- **Fix the model roster** for the comparison, e.g. ML: Random Forest, XGBoost, SVM; DL: LSTM/CLSTM, a Transformer, and (if relational data) a GNN.
- **Pre-register the evaluation protocol:** full confusion matrix per model, F1, MCC, balanced accuracy, PR-AUC; a stated imbalance-handling method; and a *time-based* train/test split to expose drift (not a random split).
- **Add a latency measurement** so the ML-vs-DL verdict reflects operational reality, not just offline scores.

---

## 6. Finalized Design Decisions

| Decision | Choice | Note |
|---|---|---|
| Market & asset class | **Indonesia Stock Exchange (IDX), common stock (equity)** | Emerging market; UMA labeling available |
| Period | **2021–2024, all IDX-listed stocks** | Post-COVID period with elevated retail activity |
| Data granularity | **Daily OHLCV + intraday minute bars** | No order-book data |
| Manipulation type | **Pump-and-dump only** | Spoofing dropped — see §6.1 |
| Labeling | **IDX UMA announcements (primary)** + heuristic/synthetic expansion | Suspicion-level ground truth |
| Imbalance handling | **SMOTE (training fold only)** | Avoid leakage |
| Model roster | **3 ML + 3 DL** (see §7.4) | Identical features & splits |
| Primary selection metric | **MCC + full confusion matrix** | Imbalance-robust |

### 6.1 Scope note: why spoofing is excluded

Spoofing is a *trade-based manipulation* defined by limit orders being **placed and then cancelled** to create a false impression of supply/demand. Detecting it requires **Level-2/Level-3 order-book (order-event) data**. Daily OHLCV and minute bars contain only aggregated price/volume — they do **not** record individual order placements or cancellations, so spoofing is physically unobservable in the chosen dataset. The study is therefore scoped to **pump-and-dump detection**, which is fully detectable from price/volume dynamics. This is a deliberate, defensible scoping decision, not a limitation to apologize for.

---

## 7. Methodology

### 7.1 Data collection plan

1. **Universe:** all common stocks listed on IDX, 2021-01-01 to 2024-12-31.
2. **Price/volume features (free sources):** pull daily OHLCV via the `yfinance` library using `.JK` tickers (e.g., `BBCA.JK`), cross-checked against IDX daily summaries. Acquire intraday minute bars where available (IDX datafeed / broker API / vendor such as Tiingo or EOD Historical Data).
3. **Labels:** scrape/collect the IDX **UMA announcement list** (idx.co.id) for the period, recording ticker and announcement date for each UMA event.
4. **Corporate-action / news filter:** collect stock splits, dividends, and major announcements to avoid mislabeling legitimate events as manipulation.

### 7.2 Labeling protocol

- **Positive class (P&D / manipulation):** a stock-window is labeled positive if the stock received a UMA announcement, using an event window (e.g., the days leading up to and including the announcement). Document the exact window (a common choice in IDX event studies is H-5 to H+5 around the announcement).
- **Positive-class expansion (optional, to ease extreme imbalance before SMOTE):** add heuristic positives using the UMA "special stock" rule of thumb — roughly a **≥25% price move over a short window combined with an abnormal volume spike** — and/or inject **synthetic P&D patterns** into clean series for controlled ground truth.
- **Negative class:** stock-windows with no UMA event and no extreme price/volume anomaly.
- **Noise disclosure (write this in the paper):** a UMA announcement signals *suspicion*, not a *confirmed* violation; the IDX explicitly states UMA does not necessarily indicate a breach. Treat labels as weak/noisy ground truth and acknowledge it as a threat to validity.

### 7.3 Feature engineering

Engineer features per stock per window (these matter more than model choice, per the recent reviews):
- **Returns:** daily/intraday returns, cumulative abnormal return (CAR) vs. the Jakarta Composite Index (IHSG/JCI).
- **Volume:** trading-volume activity (TVA) ratio vs. trailing average, volume spikes, turnover.
- **Volatility:** rolling std-dev, daily price range (high–low)/avg.
- **Microstructure (from minute bars):** intraday volatility, volume concentration within the day, number of trades / trade size proxies.
- **Momentum/technical:** RSI, moving-average crossovers, price acceleration.
- **Temporal:** day-of-week, distance to nearest corporate action.

### 7.4 Model roster (ML vs DL comparison)

| Group | Models | Input form |
|---|---|---|
| **Classical ML** | Logistic Regression (baseline), Random Forest, XGBoost/LightGBM | Engineered tabular feature vectors |
| **Deep Learning** | LSTM/BiLSTM, CNN-LSTM (CLSTM), Transformer | Sequences of daily/minute features |
| **(Optional anomaly)** | Isolation Forest / Autoencoder | Unsupervised, for comparison |

All models trained on **identical features, splits, and labels** so the comparison is fair.

### 7.5 Imbalance handling — SMOTE (leakage-safe)

- Apply **SMOTE to the training fold ONLY**, after the train/test split. Never oversample the test or validation set — doing so leaks synthetic information and inflates results. This is the single most common fatal error in this type of study.
- Consider variants (Borderline-SMOTE, SMOTE-Tomek) as an ablation.
- For DL models, class weights / focal loss are a valid alternative to compare against SMOTE.

### 7.6 Validation design

- **Time-based split, not random:** train on 2021–2023, test on 2024 (or rolling/walk-forward). A random split leaks future information and hides concept drift — critical for a manipulation study.
- **Cross-validation:** time-series CV (expanding window) within the training period for hyperparameter tuning.
- **No look-ahead:** ensure features at time *t* use only data available at *t*.

### 7.7 Evaluation metrics

- **Primary model-selection metric: Matthews Correlation Coefficient (MCC)** — uses all four confusion-matrix cells and is robust under heavy imbalance.
- **Always report the full confusion matrix** (TP/FN/FP/TN) for every model.
- **Supporting metrics:** Precision, Recall (Sensitivity), F1 (fraud class), Specificity, Balanced Accuracy, **PR-AUC** (preferred over ROC-AUC under imbalance).
- **Do NOT use** R²/RMSE/MAE — these are regression metrics and do not apply to a classification confusion matrix. (They would only apply if a separate continuous-target regression sub-task were added.)
- **Optional:** measure inference latency per model to frame the accuracy-vs-cost trade-off in the ML-vs-DL verdict.

### 7.8 Suggested experimental pipeline

1. Collect & clean data (OHLCV + minute bars + UMA list). →
2. Engineer features; build labeled windows. →
3. Time-based train/test split. →
4. SMOTE on training fold. →
5. Train all ML + DL models on identical data. →
6. Evaluate via MCC + full confusion matrix + supporting metrics on the held-out 2024 test set. →
7. Compare ML vs DL; ablate SMOTE variants and feature groups. →
8. Report threats to validity (label noise, drift, scope).

---

## References (to be formatted to your citation style — IEEE / APA)

1. Xu, J., & Livshits, B. (2019). *The Anatomy of a Cryptocurrency Pump-and-Dump Scheme.* USENIX Security Symposium.
2. La Morgia, M., et al. (2020). *Pump and Dump schemes in cryptocurrency markets.*
3. Chadalapaka, V., Chang, K., Mahajan, G., & Vasil, A. (2022). *Crypto Pump and Dump Detection via Deep Learning Techniques.* arXiv:2205.04646.
4. Hu, et al. (2023). *Sequence-based deep learning for pump-and-dump detection.*
5. Gogol, K., et al. (2024/2025). *Machine Learning-Based Detection of Pump-and-Dump Schemes in Real-Time.* arXiv:2412.18848 / DeFi'25.
6. Cartea, Á., Chang, P., & García-Arenas, G. (2023). *Spoofing and Manipulating Order Books with Learning Algorithms.*
7. Poutré, C., Chételat, D., & Morales, M. (2024). *Deep Unsupervised Anomaly Detection in High-Frequency Markets.*
8. Fabre, T., & Muni Toke, I. (2025). *High-Frequency Market Manipulation Detection with a Markov-modulated Hawkes process.* arXiv:2502.04027.
9. Fabre, T., & Challet, D. (2025). *Learning the Spoofability of Limit Order Books With Interpretable Probabilistic Neural Networks.* arXiv:2504.15908.
10. Anonymous (2026, under review). *Detecting Multilevel Manipulation from Limit Order Book via Cascaded Contrastive Representation Learning.* ICLR 2026 submission.
11. Atugoda, R., et al. (2026). *Enhancing Stock Market Surveillance Using Machine Learning Algorithms for Fraud Detection.*
12. (2025). *Year-over-Year Developments in Financial Fraud Detection via Deep Learning: A Systematic Literature Review.* arXiv:2502.00201.
13. (2025). *Advancing Machine Learning for Financial Fraud Detection: A Comprehensive Review of Algorithms, Challenges, and Future Directions.*
14. (2022). *Empirical study of Machine Learning Classifier Evaluation Metrics behavior in Massively Imbalanced and Noisy data.* arXiv:2208.11904.

*Verify every citation against the original source before submission; some details (page numbers, exact venues, author lists) should be confirmed from the linked papers.*
