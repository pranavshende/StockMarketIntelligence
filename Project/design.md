# Technical Design Document (v4: Liu et al. Extended)

## 1. Data Design

### Coverage
- **Universe:** 25–30 NSE stocks across 5 sectors (Banking, IT, FMCG, Auto, Pharma), 5–6 per sector.
- **History:** 2+ years of daily OHLCV data. **Precondition:** Minimum 400 trading days of history required per stock.
- **Training Strategy:** Train and evaluate strictly *per-stock* (avoiding the price-scale mixing issues present in the pooled-company approach of Liu et al.).
- **Ingestion:** Daily End-of-Day (EOD) batch script using `yfinance` API calls. Must include a 3-retry exponential backoff to handle API failures. Kafka/Redis Streams ingestion deferred to future work.

### Feature Store
- **Storage:** **Supabase (PostgreSQL)**.
- **Data Contract:** The final output database schema read by the Node.js API must be exactly: `[Date, Ticker, Actual_Return, Actual_Price, y_hat_Static, y_hat_Adaptive, Regime_Flag]`.
- **Features:** OHLCV, Lag features, and Technical Indicators (RSI, MACD, Bollinger Bands, rolling volatility).
- **Experiment Variations:** Include at least one experiment run on *returns* instead of raw price levels to address potential R² inflation concerns.
- **Target Variable:** The primary target `y` is defined as **next-day log returns**. Dashboard visualizations will convert these back to price levels.
- **Anti-Leakage Scaling:** To correct a major flaw in Liu et al., all scalers (e.g., MinMaxScaler) must be fit *per-fold* on training data only and applied out-of-sample. Global scaling before splitting is prohibited.

## 2. Modeling Framework

### Base Learners (Adopted from Liu et al. 2024)
- **LSTM:** 2 LSTM layers (approx 100 units each), followed by dropout.
- **ANN:** 2 hidden layers (100 ReLU units -> 50 units).
- *Implementation Note:* Start with their exact settings. If hyperparameter tuning is required for NSE data, document it explicitly for the paper.

### The Static Meta-Model (Control Group)
- **Equation:** `ŷ_meta = β₀ + β₁·ŷ_LSTM + β₂·ŷ_ANN + ε`
- **Method:** Standard linear regression fit once on the training split and frozen.

### The Adaptive Meta-Model (The Contribution)
- **Drift Detector:** Rolling residual error monitoring via z-score (30-day window, |z| > 2.0).
- **HMM Regime Classifier:** A **Hidden Markov Model (HMM)** fitted per stock on (daily log-returns, 20-day rolling realized volatility) to infer a discrete latent state sequence. Number of hidden states: 3 [TODO: Verify from implementation]. HMM is preferred over a simple ATR volatility tercile because it produces principled probabilistic states with transition matrices rather than hard threshold cuts. The HMM input series is kept separate from the base-learner prediction features.
- **Mechanism:** Re-fits `β(r_t)` coefficients dynamically upon drift detection using **Bayesian Ridge Regression** (60-day rolling lookback). Bayesian Ridge simultaneously prevents singular matrix errors from multicollinearity between base learner outputs and provides posterior uncertainty estimates (confidence intervals) on β₁ and β₂, which are reported in the paper to quantify how ensemble reliance shifts across regimes.
- **Equation:** `ŷ_meta(t) = β₀(r_t) + β₁(r_t)·ŷ_LSTM + β₂(r_t)·ŷ_ANN + ε`

### Independent Rigor Baselines
- Random Forest and XGBoost evaluated independently, utilizing SHAP for feature explainability.
- **SHAP scope:** Applied to RF and XGBoost only. SHAP for LSTM/ANN base learners is out of scope for this build cycle.

## 3. Evaluation & Reporting

### Walk-Forward Harness
- **Methodology:** Rolling-origin / walk-forward cross-validation (primary evaluation method for all experiments).
- **Walk-Forward Parameters:** Training window = 252 days (1 yr), Testing window = 21 days (1 mo), Step size = 21 days.
- **The Core Ablation:** Directly compare the static Liu et al. meta-model against the regime-adaptive version across all folds.
- **Secondary Comparison:** Run the static meta-model once using the original Liu et al. single-TimeSeriesSplit methodology as a secondary comparison point. This demonstrates the difference between their evaluation approach and the walk-forward harness, and must be clearly labelled as a methodological comparison, not a result.

### Presentation Layer (React / React Native / Node.js)
- **Stock Selector:** A dropdown navigation element to select the active ticker from the 25-30 NSE stocks.
- **Empty State:** UI must gracefully handle missing Supabase records for failed stocks by displaying a "Backtest results not available" card instead of crashing.
- **Static-vs-Adaptive View:** Side-by-side plotting of predictions from the frozen meta-model vs. the adaptive meta-model (loading from pre-computed Supabase results via the Node.js API to avoid on-the-fly ML training timeouts).
- **Regime Timeline:** A visual log underneath the charts showing the detected market regimes and when the adaptive coefficients were updated.
- **Model Comparison View:** LSTM+ANN ensemble vs. RF vs. XGBoost.
- **SHAP Panel:** Feature importance plots for RF/XGBoost.
- **Export:** CSV/Parquet export retained.

## 4. Paper-Facing Outputs

- **Auto-generated experiment report:** Metrics table (static vs. adaptive meta-model, per stock, per fold) and regime-timeline plots, regenerated from real runs.
- **Rewritten Related Work section:** Positioning against Liu et al. (2024) as the base architecture, the Operations Research Forum 2026 multi-source ensemble as the closest NSE/BSE analog, and the concept-drift literature as the gap being closed.
- **Fixed reference list:** Remove duplicated entries from the original submission.
- **New "Proposed Approach" section:** Built around the modified equation (`ŷ_meta(t) = β₀(r_t) + β₁(r_t)·ŷ_LSTM + β₂(r_t)·ŷ_ANN + ε`).

## 5. Non-Goals

- Live trading / order execution
- Options / derivatives data
- Sentiment / news ingestion (flag as future work in paper)
- Temporal Fusion Transformer (TFT) — deferred to future work; cite as related work only
- Production-scale Kafka cluster
- SHAP for LSTM/ANN base learners

## 6. Novelty Claims (three-part, for the paper)

1. **First application** of the Liu et al. LSTM+ANN stacking ensemble to Indian NSE equities.
2. **HMM-based regime-adaptive extension** of their static linear meta-model — a single-variable ablation where only the meta-model coefficients change, using HMM-detected regimes rather than a fixed volatility threshold.
3. **Bayesian Ridge Regression meta-model** — replaces plain Ridge with a Bayesian formulation that provides posterior uncertainty estimates on ensemble weighting coefficients (β₁, β₂), enabling statistically grounded reporting on how base-learner reliance shifts across market regimes.
