# PRD v4: Real-Time Indian Stock Market Intelligence Platform
### (Base architecture: Liu et al. LSTM+ANN stacking ensemble → extended with a regime-adaptive meta-model)

**Author:** Pranav Shende | **Status:** Build-ready draft | **Supersedes:** v3 PRD

---

## 0. What changed from v3, and why

v3 proposed a home-grown "regime-weighted ensemble" across four independent models (LR, RF, XGBoost, GRU) with no fixed mathematical structure. This version replaces that with something more defensible: **adopt a specific, published, citable architecture as the base, and extend its one static assumption.**

**Base architecture (adopted, cited):** Liu, Guo, Xing, Sha, Chen, Jin, Zheng & Yu, *"Application of an ANN and LSTM-based Ensemble Model for Stock Market Prediction,"* 2024 IEEE 7th ICISCAE. Their method:
- Two independently trained base learners: an LSTM (2 layers, 100 units each, dropout) and an ANN (2 hidden layers, 100 ReLU units then 50 units)
- A **linear regression meta-model** that stacks their outputs:

  ŷ_meta = β₀ + β₁·ŷ_LSTM + β₂·ŷ_ANN + ε

- *Note:* The original paper claims to test on "860M records" of S&P 500 daily data (mathematically impossible) and contains hallucinated medical citations. We are adopting **only their mathematical architecture** (the LSTM+ANN topology and linear stacking equation). We explicitly reject their flawed experimental methodology.

**The gap we extend:** their meta-model's coefficients (β₀, β₁, β₂) are fit once and frozen. If the market regime shifts (volatility spike, trend reversal, post-earnings drift), the fixed weighting between LSTM and ANN stops being optimal, and nothing in their system detects or responds to that. This is the same structural gap identified in the concept-drift literature (Section 8) — but here it's precise and mathematical, not a vague "add adaptivity" claim: **we are extending one specific, named equation from a specific, cited paper.**

**Our contribution, stated as a modification to their equation:**

ŷ_meta(t) = β₀(r_t) + β₁(r_t)·ŷ_LSTM + β₂(r_t)·ŷ_ANN + ε

where `r_t` is the detected market regime at time t, and the coefficients are re-estimated (or selected from a small per-regime bank) whenever a drift detector flags a regime change — instead of being fixed for the whole series.

This is a three-part, honestly stackable novelty claim for the paper:
1. First application of the Liu et al. LSTM+ANN stacking architecture to Indian NSE equities (no existing paper does this, per the literature scan).
2. A regime-adaptive extension of their meta-model's coefficients (using HMM-detected regimes instead of fixed thresholds), tested against their original static version as an explicit ablation.
3. Replacement of the plain linear/Ridge meta-model re-fit with a Bayesian Ridge Regression, providing posterior uncertainty quantification on how the ensemble weighting (β₁, β₂) shifts across market regimes.

---

## 1. Goals

**Primary:** Build a working system whose central claim is precise and testable — "making the Liu et al. meta-model regime-adaptive improves on its static version for Indian equities" — with an honest ablation (static vs. adaptive) as the paper's headline result.

**Secondary:**
- Fix every earlier reviewer complaint (dataset size, model comparison breadth, statistical validation, references, figures)
- Keep the build achievable solo, alongside coursework
- Produce a demoable dashboard, not just offline notebooks

**Non-goals:** live trading/execution, options/derivatives, tick-level data, production-scale infra, sentiment/news ingestion (future work).

---

## 2. The novel contribution, precisely stated

### 2.1 What Liu et al. do (the base, adopted as-is first)
- Two base learners (LSTM, ANN), independently trained
- One linear meta-model combining their outputs, fit once on the training split, frozen thereafter
- Evaluated with a single TimeSeriesSplit, on pooled multi-company S&P 500 price levels

### 2.2 What we add
**A. Drift detector** — monitors rolling residual error of the meta-model's predictions using a **30-day rolling window** and a **z-score threshold of |z| > 2.0**. (Page-Hinkley test is a stretch upgrade). When error drifts beyond the threshold, it flags a regime change for that stock.

**B. HMM-based regime classifier** — instead of a simple ATR volatility tercile cut, a **Hidden Markov Model (HMM)** is fitted per stock on the returns series to identify latent market states (e.g., low-volatility trending / high-volatility trending / sideways/mean-reverting). The HMM produces a principled probabilistic state sequence with defined transition probabilities, rather than a fixed threshold. The number of hidden states is set to **3** as a starting point; this is a tunable hyperparameter to be confirmed during implementation. *HMM inputs:* daily log-returns and 20-day rolling realized volatility, per stock.

**C. Regime-conditioned meta-model** — instead of one fixed (β₀, β₁, β₂), the coefficients are re-estimated on a recent rolling window (the last 60 trading days) each time the drift detector fires. To prevent singular matrix errors due to potential multicollinearity between `ŷ_LSTM` and `ŷ_ANN`, and to produce posterior uncertainty estimates on the coefficients, **Bayesian Ridge Regression** is used instead of unregularized OLS or plain Ridge Regression. The Bayesian posterior provides explicit confidence intervals on β₁ and β₂, which strengthens the paper's statistical story (we can report how much the meta-model's reliance on LSTM vs. ANN shifts across detected regimes).

All three extensions keep the base learners (LSTM, ANN) untouched — **only the meta-model's combination logic is made adaptive.** This keeps the extension small, focused, and directly attributable to the one equation being modified.

### 2.3 Why this is stronger than the v3 proposal
- It's anchored to a **specific, citable, reproducible** published method instead of an ad hoc ensemble of four unrelated models
- The ablation (static Liu et al. meta-model vs. our regime-adaptive meta-model, same base learners, same data) is a clean, single-variable experiment — exactly what a reviewer wants to see
- It naturally produces two citable claims (Indian-market application + adaptive extension) instead of one

---

## 3. Full scope

### 3.1 Data layer
- 25–30 NSE stocks across 5 sectors (Banking, IT, FMCG, Auto, Pharma), 2+ years daily OHLCV
- Ingestion: Daily End-of-Day (EOD) batch scraper using `yfinance` API calls. 
  - **Error Handling:** Must include a 3-retry exponential backoff per ticker. If `yfinance` permanently fails for a ticker, the script must log the error and continue to the next stock without crashing the pipeline.
  - **Precondition:** Stocks must have a minimum of 400 trading days of history. Stocks with insufficient history must be automatically dropped.
  - (Legacy 5-minute intraday polling is explicitly deprecated to align with daily OHLCV forecasting. Kafka/streaming deferred to future work.)
- Storage: **Supabase (PostgreSQL)**. The ML pipeline will write the final historical and daily predictions to Supabase instead of local Parquet files.
- Unlike Liu et al.'s pooled-across-companies approach, **train and evaluate per stock** — this avoids the price-scale-mixing issue in their pooled design and is more consistent with how a real dashboard would be used (per-stock forecasts)

### 3.2 Feature layer
- Keep existing preprocessing: cleaning, missing values, outliers.
- **Anti-Leakage Rule (Correction to Liu et al.):** All normalization/scaling MUST be fit strictly on the training fold and applied out-of-sample to the validation/test folds. Global scaling before splitting is strictly prohibited to prevent future-data leakage.
- **Target Variable (Correction to Liu et al.):** The primary prediction target `y` is defined as **next-day log returns** ($ln(P_t / P_{t-1})$). The dashboard will convert predicted returns back to predicted price levels for visualization. This avoids R² inflation and non-stationarity issues present in raw price predictions.
- Add technical indicators (RSI, MACD, Bollinger Bands, rolling volatility) — used as model features. Note: the HMM regime classifier operates on log-returns and rolling realized volatility directly, *not* on these technical indicators, to keep the regime signal clean and separate from the prediction features.
- Add explicit lag features as in v1, but also report at least one experiment on **returns** instead of raw price levels, to address the inflated-R² concern discussed when comparing against Liu et al.

### 3.3 Modeling layer
- **Base learners (adopted from Liu et al., cited):** LSTM (2 layers, ~100 units, dropout) and ANN (2 hidden layers, 100 → 50 units, ReLU) — reproduce their architecture as closely as practical, adapted to per-stock NSE data.
- **Baseline meta-model (reproduced for the ablation):** their original static linear regression stacking — this is the paper's control condition.
- **Our meta-model (the contribution):** three-stage adaptive pipeline:
  1. **HMM Regime Classifier** — fitted per stock on (log-returns, 20-day realized volatility); produces a discrete regime label at each timestep. Number of hidden states: 3 [TODO: Verify from implementation].
  2. **Drift Detector** — rolling z-score (30-day window, |z| > 2.0) on meta-model residuals triggers a re-fit event.
  3. **Bayesian Ridge Regression Meta-Model** — on each drift event, re-fits the stacking equation `ŷ_meta(t) = β₀(r_t) + β₁(r_t)·ŷ_LSTM + β₂(r_t)·ŷ_ANN` on the most recent 60-day window. Bayesian Ridge provides posterior distributions over β₁ and β₂, enabling uncertainty-quantified coefficient reporting in the paper.
- **Additional baselines retained from earlier scope (for breadth, not the core claim):** Random Forest and XGBoost, evaluated independently, to keep the "more models compared" rigor fix from earlier reviewer feedback.
- **Validation:** walk-forward / rolling-origin CV. **Parameters:** Training window of 252 days (1 trading year), Testing window of 21 days (1 trading month), rolling step size of 21 days. The original Liu et al. single-TimeSeriesSplit approach is run once, explicitly, as a secondary comparison point.
- **Explainability:** SHAP for the RF/XGBoost baselines (skip for LSTM/ANN — not worth the implementation cost).

### 3.4 Serving & dashboard layer (New Stack)
- **Web Frontend:** React (deployed on Vercel).
- **Mobile Frontend:** React Native application for iOS/Android.
- **Backend API:** Node.js (deployed on AWS or Render).
- **Database:** Supabase (PostgreSQL).
- **Navigation:** Top-nav dropdown selector explicitly listing the 25-30 NSE tickers to allow the user to switch the active stock context.
- **Static-vs-Adaptive comparison view:** Showing the frozen Liu et al. meta-model's predictions next to the regime-adaptive version, with the regime timeline underneath. This displays the pre-computed Walk-Forward CV historical backtest results fetched from Supabase (not on-the-fly inference).
  - **Empty State:** If the user selects a ticker that failed during the ML pipeline (and thus has no data in Supabase), display an empty state card: "Backtest results not available for this stock" instead of crashing.
- **Data Contract:** The Node.js API reads from a defined Supabase table: `[Date, Ticker, Actual_Return, Actual_Price, y_hat_Static, y_hat_Adaptive, Regime_Flag]`.
- Model comparison view (LSTM+ANN ensemble vs. RF vs. XGBoost)
- SHAP panel for tree models

### 3.5 Paper-facing outputs
- Auto-generated experiment report: metrics table (static vs. adaptive meta-model, per stock, per fold) + regime-timeline plots
- Rewritten Related Work section explicitly positioning against Liu et al. (2024) as the base architecture, and against the Operations Research Forum 2026 multi-source ensemble and the concept-drift literature (Section 8) as the gap being closed
- Fixed reference list (remove duplicated entries from the original submission)
- New "Proposed Approach" section built around the modified equation in Section 0

---

## 4. Explicit trade-offs made

| Considered | Decision | Why |
|---|---|---|
| Kafka/Redis Streams ingestion | Cut to future work | Doesn't touch the novelty claim |
| Full ML Pipeline in Node.js | Rejected | Python is required for PyTorch/TensorFlow (LSTM/ANN) and Scikit-learn (Walk-Forward CV, Ridge Regression). The ML pipeline remains in Python, decoupled from the Node.js/React web stack. |
| Four independent baseline models with ad hoc regime-weighting (v3 approach) | Replaced with the Liu et al. base + single modified meta-model | Anchoring to one cited, reproducible equation is a stronger, cleaner novelty claim than an unstructured ensemble |
| Pooled cross-company training (as Liu et al. do) | Per-stock training instead | Avoids mixing price scales across companies; more consistent with a real per-stock dashboard use case |
| SHAP for LSTM/ANN | Cut | Not worth the implementation cost this cycle; SHAP on RF/XGBoost already answers the explainability critique |

---

## 5. Architecture

```
[Python ML Pipeline (Offline/Batch)]
[NSE public data feeds] --> [yfinance EOD scraper]
        |
        v
[Preprocessing & Feature Engineering (Python)]
        |
        v
[Base Learners: LSTM & ANN (PyTorch/Keras)]
        |
        v
[Walk-forward evaluator: Static Meta-Model vs Adaptive Meta-Model (Ridge Regression)]
        |
        v
[Experiment Results & Predictions]
        |
        v (Database Write)
=======================================================
[Supabase (PostgreSQL Database)]
=======================================================
        | (Database Read)
        v
[Node.js API (Render / AWS)]
        | (REST / GraphQL API)
        +----------------------------------+
        |                                  |
        v                                  v
[React Web App (Vercel)]        [React Native Mobile App]
```

---

## 6. Success metrics

- **Novelty claim tested directly:** report per-stock, walk-forward results comparing the frozen Liu et al. meta-model against the regime-adaptive version — a mixed result (adaptive wins in high-volatility regimes, ties elsewhere) is a legitimate, publishable finding
- **Base reproduction sanity check:** the reproduced Liu et al. architecture should produce internally consistent results on NSE data (not necessarily matching their S&P 500 numbers, given different data/scale) — report this as a reproduction, not a replication of their exact metrics
- **Rigor:** 25+ stocks, 2+ years, walk-forward CV, RF/XGBoost as additional baselines, SHAP for tree models, at least one returns-based experiment
- **Paper hygiene:** references fixed, lit review updated, figures redone
- **Demo:** dashboard shows the static-vs-adaptive comparison and regime timeline live for at least one full backtest period

---

## 7. Build sequence (7–9 weeks solo, alongside coursework)

1. **Week 1–2:** Expand Python data collection (25–30 stocks), setup **Supabase** database schema, add technical indicators.
2. **Week 2–3:** Reproduce the Liu et al. LSTM+ANN base learners and static linear meta-model (Python) — get it working and stable first.
3. **Week 3–4:** Build the walk-forward CV harness; re-run the static meta-model under it; add RF/XGBoost as independent baselines.
4. **Week 4–5:** Build the drift detector (rolling z-score first).
5. **Week 5–6:** Build the regime-conditioned meta-model (rolling re-fit using Ridge Regression).
6. **Week 6:** Run the full ablation across all stocks and write the final historical results table to **Supabase**.
7. **Week 7:** Build the **Node.js API** (Render/AWS) to serve the Supabase data, and start the **React web app** (Vercel).
8. **Week 8:** Finish React web app (static-vs-adaptive view, regime timeline). Build the **React Native** mobile app.
9. **Week 9:** Rewrite the paper — Proposed Approach section built around the modified equation, Related Work repositioned around Liu et al. + concept-drift literature, fixed references, redone figures.

**If the deadline is shorter:** cut in this order — dashboard polish → SHAP → RF/XGBoost baselines (keep the LSTM+ANN ablation as the sole result) → Page-Hinkley upgrade (keep the z-score detector). **Never cut the static-vs-adaptive meta-model ablation** — that is now the paper's entire contribution.

---

## 8. Literature informing this PRD

**Base architecture being extended:**
- Liu, Guo, Xing, Sha, Chen, Jin, Zheng & Yu, *"Application of an ANN and LSTM-based Ensemble Model for Stock Market Prediction,"* 2024 IEEE 7th ICISCAE, pp. 390–395. (arXiv: 2410.20253)

**Positioning / gap-closing citations:**
- *"Explainable Multi-source AI Framework for Real-Time Stock Price Monitoring and Prediction,"* Operations Research Forum, 2026 — closest static-ensemble analog for NSE/BSE (TFT+LSTM+RF+XGBoost); does not adapt over time
- *"Evaluating the Impact of Drift Detection Mechanisms on Stock Market Forecasting,"* Knowledge and Information Systems (Springer) — drift-aware evaluation on Brazilian equities; template for the ablation design, no Indian-market equivalent exists
- *"Domain Specific Concept Drift Detectors for Predicting Financial Time Series,"* 2025 — static vs. periodically-retrained vs. self-adaptive models on S&P 500/Bitcoin; template for the static-vs-adaptive comparison framing
- *"Incremental Learning of Stock Trends via Meta-Learning with Dynamic Adaptation,"* arXiv, 2024 — distinguishes predictable vs. unpredictable drift in stock data; useful conceptual grounding for the regime classifier

**Supporting rigor citations (broader comparisons, cite for related work / breadth):**
- Karulkar, Shah & Naik, *"From Data to Decisions,"* SAGE, 2025 — Indian large-cap ML comparison
- *"Machine Learning and Time Series Approaches for Stock Price Prediction: A Case Study of Indian Banks,"* ScienceDirect, 2025
- *"A Hybrid LSTM-GRU Model for Stock Price Prediction,"* IEEE Access, 2025
- Alam et al., *"Enhancing Stock Market Prediction: A Robust LSTM-DNN Model Analysis on 26 Real-Life Datasets,"* IEEE Access, vol. 12, 2024

**One-sentence Introduction pitch:** "We adopt the LSTM+ANN stacking ensemble of Liu et al. (2024) and, for the first time, apply it to Indian equities while extending its static linear meta-model into a regime-adaptive one, closing a gap identified separately in the concept-drift literature but not yet addressed for this specific architecture or market."

---

## 9. Open questions before starting

- Whether to reproduce Liu et al.'s exact hyperparameters (2-layer LSTM, 100 units; ANN 100→50 units) or tune them for NSE data — start with their exact settings for the reproduction, then note any necessary changes explicitly in the paper.
- **HMM number of hidden states:** Starting value is 3 (low-vol trending / high-vol trending / sideways). The optimal number should be validated per stock using BIC/AIC [TODO: Verify from implementation].
- **Bayesian Ridge prior strength (α, λ):** Use scikit-learn defaults initially (`alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6`). Document any tuning explicitly in the paper [TODO: Verify from implementation].
- Coefficient bank vs. rolling re-fit for the adaptive meta-model — rolling re-fit with Bayesian Ridge is the chosen approach. A per-regime coefficient bank remains a contingency if rolling re-fit proves unstable.
- Confirm the resubmission deadline: if under ~4 weeks, ship the reproduced static Liu et al. baseline plus the rigor fixes now, and describe the HMM + Bayesian Ridge adaptive meta-model as "future work" rather than rushing it into the results section.