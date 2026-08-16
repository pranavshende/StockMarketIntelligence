# Implementation Phases (v4)

**Estimated Timeline:** 7–9 weeks (Solo build alongside coursework)

## Week 1–2: Data Foundation & Preprocessing
- Expand Python data collection to 25–30 NSE stocks (stored per-stock). Enforce 400-day minimum history precondition and 3-retry API backoff.
- Setup **Supabase (PostgreSQL)** schema and define the exact data contract.
- Add technical indicators (vital for both features and regime detection).

## Week 2–3: Reproduce Base Architecture (Control)
- Reproduce the Liu et al. LSTM and ANN base learners.
- Implement the static linear meta-model combining `ŷ_LSTM` and `ŷ_ANN`.
- Get this static control condition stable on the NSE data.

## Week 3–4: Evaluation Harness & Breadth Baselines
- Build the walk-forward CV harness.
- Re-run the static meta-model under the walk-forward harness.
- **Also run** the static meta-model once under the original Liu et al. single-TimeSeriesSplit methodology as an explicit secondary comparison point — clearly document it as a methodological comparison, not a competing baseline.
- Add RF and XGBoost as independent, secondary baselines.

## Week 4–5: Drift Detector
- Build the drift detector module (rolling z-score on meta-model residuals, 30-day window, |z| > 2.0).
- Fit the **HMM Regime Classifier** per stock on (daily log-returns, 20-day rolling realized volatility). Start with 3 hidden states; validate the optimal number using BIC/AIC per stock. *Note: the HMM input series must be computed independently from the base-learner feature set to avoid feature contamination.*

## Week 5–6: Regime-Conditioned Meta-Model (Core Novelty)
- Implement the regime-adaptive meta-model: rolling re-fit of β₀(r_t), β₁(r_t), β₂(r_t) on the most recent 60-day window each time the drift detector fires, using **Bayesian Ridge Regression** (scikit-learn `BayesianRidge`). Record the posterior mean and variance of β₁ and β₂ at each update for per-regime reporting.
- *Note: Budget the most debugging time here. Verify that the Bayesian Ridge prior defaults produce stable fits before considering hyperparameter tuning. A per-regime coefficient bank remains a contingency if rolling re-fit proves unstable.*

## Week 6–7: The Core Ablation Experiment
- Run the full ablation experiment: Static Meta-Model vs. Adaptive Meta-Model across all stocks.
- Run at least one returns-based experiment.
- Generate metrics table and regime-timeline plots.

## Week 7: Backend & Explainability
- Add SHAP analysis for the RF/XGBoost independent baselines.
- Build the **Node.js API** (Render/AWS) to serve the Supabase data.
- Wire all ML outputs into the auto-generated experiment report.

## Week 8: React Web & Mobile App
- Build the **React web app** (Vercel) for the static-vs-adaptive view and regime timeline.
- Implement the empty state UI for missing Supabase data.
- Build the **React Native** mobile app to mirror the web functionality.

## Week 9: Paper Rewrite
- Write the new "Proposed Approach" section centered on extending the Liu et al. equation.
- Rewrite "Related Work" to position against Liu et al. and the concept-drift literature.
- Fix references and redo figures.

---

> [!WARNING]
> **Contingency Plan:** If the deadline is shorter than 9 weeks, cut in this specific order:
> 1. React Native App
> 2. Dashboard polish
> 3. SHAP
> 4. RF/XGBoost baselines (keep the LSTM+ANN ablation as the sole result)
> 
> *Never cut the static-vs-adaptive meta-model ablation — that is the paper's central contribution.*

---

## Non-Goals

The following are **not** in scope for this build:

- Live trading / order execution
- Options / derivatives data
- Tick-level HFT data
- Sentiment / news ingestion (flag as future work in paper)
- Temporal Fusion Transformer (TFT) — cite as related work, do not implement
- Production-scale Kafka cluster
- SHAP for LSTM/ANN base learners

## Success Criteria

- **Novelty tested honestly:** Report per-stock, walk-forward results for static vs. adaptive — a mixed result (adaptive wins in high-volatility, ties elsewhere) is a legitimate finding.
- **Reproduction sanity check:** The reproduced Liu et al. architecture produces internally consistent results on NSE data — reported as a reproduction of their mathematical topology, explicitly correcting their methodological flaws (data leakage, undefined targets).
- **Rigor:** 25+ stocks, 2+ years, walk-forward CV, RF/XGBoost baselines, SHAP for tree models, at least one returns-based experiment.
- **Paper hygiene:** References fixed, lit review updated with 2024–2026 sources, figures redone at readable resolution.
- **Demo:** Dashboard shows the static-vs-adaptive comparison and regime timeline live for at least one full backtest period.
