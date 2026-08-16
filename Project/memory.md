# Project Memory (v4)

## Context & Pivot
- **Project Name:** Real-Time Indian Stock Market Intelligence Platform (v4)
- **Pivot Reason:** While v3 identified "regime adaptivity" as the novel idea, it relied on an unstructured ensemble of 4 disparate models. v4 anchors the novelty to a **specific, published, citable architecture** (Liu et al. 2024, LSTM+ANN Stacking) and extends its single static assumption (fixed meta-model coefficients).

## Key Decisions & The Novelty Claim
1. **The Control Group:** The exact reproduction of the Liu et al. 2024 static stacking ensemble (LSTM + ANN + Linear Meta-Model) applied to NSE data, evaluated under both (a) their original single-TimeSeriesSplit and (b) walk-forward CV, to allow explicit comparison of evaluation methodologies. *Note: We reproduce only their architecture. The original paper has fatal flaws (hallucinated citations, impossible 860M record dataset, and global scaling leakage). We explicitly correct these flaws in our implementation.*
2. **The Contribution:** Extending the static `ŷ_meta = β₀ + β₁·ŷ_LSTM + β₂·ŷ_ANN` to a regime-adaptive version where `β(r_t)` adapts based on a drift detector.
3. **Why this is better:** It creates a mathematically precise, single-variable ablation study (Static Coefficients vs. Dynamic Coefficients). This is significantly stronger and cleaner for peer review than a vague "adaptive vs static" claim across mismatched models.
4. **Data Strategy:** 25–30 NSE stocks across 5 sectors (Banking, IT, FMCG, Auto, Pharma). **Precondition:** Minimum 400 trading days required. `yfinance` ingestion requires 3-retry exponential backoff. Returns-based experiments are mandated to check for R² inflation.
5. **Additional Baselines:** RF and XGBoost are evaluated independently alongside the main LSTM+ANN ablation to satisfy the reviewer's "limited model comparison" critique.
6. **SHAP Scope:** SHAP explainability is applied to RF and XGBoost only. Applying SHAP to LSTM/ANN is out of scope for this build cycle.
7. **Drift Detector:** Rolling z-score threshold (30-day window, |z| > 2.0) on meta-model residuals is the primary implementation. Page-Hinkley test is a stretch upgrade only if time allows.
8. **Regime Definition:** **Hidden Markov Model (HMM)** fitted per stock on (daily log-returns, 20-day rolling realized volatility). Starting number of hidden states: 3 (low-vol trending / high-vol trending / sideways). Optimal number validated per stock via BIC/AIC during implementation. HMM is preferred over ATR volatility terciles because it produces principled probabilistic states with transition matrices. The HMM input series is kept separate from base-learner prediction features.
9. **Meta-Model Implementation:** Rolling re-fit on a 60-day lookback window, triggered by the drift detector, using **Bayesian Ridge Regression** (not plain Ridge or OLS). Bayesian Ridge prevents multicollinearity crashes and provides posterior uncertainty estimates on β₁ and β₂ (how much the ensemble trusts LSTM vs. ANN per regime) — this is a first-class result reported in the paper, not just a safety mechanism. A per-regime coefficient bank remains a contingency if rolling re-fit proves unstable.
10. **Timeline:** 7–9 weeks solo build alongside coursework. Contingency cut order: React Native App → dashboard polish → SHAP → RF/XGBoost baselines → Page-Hinkley upgrade. The static-vs-adaptive meta-model ablation is never cut.
11. **Paper-Facing Outputs:** The build must produce an auto-generated experiment report (metrics table + regime-timeline plots) suitable for direct use in the revised paper, along with a rewritten Related Work section and fixed reference list.
12. **Tech Stack Shift:** Shifted from local Flask/DuckDB to Cloud: Python (ML) + Supabase (DB) + Node.js (API) + React (Web) + React Native (Mobile).

## Non-Goals (explicitly cut)

- Live trading / order execution
- Options / derivatives data
- Kafka/Redis Streams ingestion (deferred to future work)
- Temporal Fusion Transformer / TFT (cite as related work, do not implement)
- Sentiment / news ingestion (future work)
- SHAP for LSTM/ANN base learners

## Open Questions (from master PRD §9)

- Whether to reproduce Liu et al.'s exact hyperparameters (2-layer LSTM, 100 units; ANN 100→50 units) or tune them for NSE data — start with their exact settings, note any necessary changes explicitly in the paper.
- **HMM number of hidden states:** Starting value is 3. Validate per stock using BIC/AIC during implementation [TODO: Verify from implementation].
- **Bayesian Ridge prior hyperparameters (α, λ):** Use scikit-learn defaults initially; document any tuning in the paper [TODO: Verify from implementation].
- Confirm the resubmission deadline: if under ~4 weeks, ship the reproduced static Liu et al. baseline plus the rigor fixes now, and describe the HMM + Bayesian Ridge adaptive meta-model as "future work".

## Final Goal
Ship a single, highly defensible ablation study demonstrating that making a static meta-model regime-adaptive yields superior predictive robustness during market shifts on Indian equities.
