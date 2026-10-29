# Data Catalogue & Feature Documentation

This directory contains the standardized historical and out-of-sample datasets for the Iberian Electricity Market (MIBEL: OMIE Spot + OMIP Derivatives) augmented with zero-shot LLM macroeconomic news sentiment.

## Temporal Coverage
- **Training & Backtesting Horizon (Phase 1, 2, 3)**: 2023-01-01 to 2025-12-31 (1,095 days).
- **Out-of-Sample Shock Horizon (Phase 3 & Table 13)**: 2026-01-01 to 2026-05-31 (148 delivery days; Jan 3 to May 30, 2026).

## Datasets Hierarchy (Factorial Ablation)
1. `Dataset_Inicial.csv`: Core spot price, forward curves (D1--D3, W1--W4, M1--M6), and physical system variables.
2. `Dataset_Exogeno_Fecha.csv`: Adds calendar Fourier cyclical features (day-of-week, month, seasonal harmonics).
3. `Dataset_Exogeno_Exogenas.csv`: Adds international commodity markets (Brent crude, TTF natural gas, API2 coal, EUA carbon allowances).
4. `Dataset_Financiero_NLP.csv`: Financial forward spreads augmented with news sentiment scores.
5. `Dataset_NLP_Global.csv`: Aggregated global market sentiment score.
6. `Dataset_NLP_Subgrupos.csv`: Fine-grained multimodal sentiment categories (short-term, medium-term, geopolitical, regulatory, demand, generation mix).
7. `Dataset_M7_Fechas_Factores.csv`: Physical grid + cyclical calendar + commodity factors.
8. `Dataset_M8_Fisico_NLP.csv`: Full multimodal state representation (Physical fundamentals + Forward curve spreads + LLM Subgroup Sentiments).
