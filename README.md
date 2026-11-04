# Multi-Horizon Industrial Electricity Procurement under Market Dislocation
### Deep Reinforcement Learning Augmented with LLM-Derived News Sentiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)

Official replication code and data repository for the manuscript submitted to ***Engineering Applications of Artificial Intelligence* (Elsevier)**.

---

## 📌 Repository Overview

This repository provides end-to-end reproducibility for the multi-horizon electricity procurement framework evaluated in the Iberian Electricity Market (**MIBEL**: OMIE Spot + OMIP Derivatives) under structural wholesale price dislocations.

```
mibel-drl-procurement/
├── configs/
│   └── hpo_best_params.yaml          # Best Optuna hyperparameters for LSTM and GTrXL
├── data/
│   ├── README.md                     # Feature dictionary and column descriptions
│   ├── processed/
│   │   ├── Datasets_finales/         # Factorial Ablation datasets (2023-2025)
│   │   └── Datasets_finales_2026/    # Out-of-sample evaluation datasets (2026 shock)
│   └── results/                      # Out-of-sample evaluation trajectories across 5 seeds
├── paper_latex/                      # LaTeX manuscript source (EAAI elsarticle format)
│   ├── main.tex                      # Blinded main text (double-blind peer review)
│   ├── title_page.tex                # Title page with author details and declarations
│   ├── highlights.tex                # Official highlights (<= 85 chars per bullet)
│   ├── references.bib                # BibTeX references
│   └── figures/                      # High-resolution vector PDF & PNG figures
├── scripts/
│   └── reproduce_all_tables.py       # Standalone dynamic reproduction script for key empirical tables
├── src/
│   ├── baselines/
│   │   └── systematic_baselines.py   # Mechanical benchmark rules (SPOT, D1, D2, D3)
│   ├── env/
│   │   └── mibel_procurement_env.py  # Canonical Gymnasium MDP environment (H=400, dim=715)
│   ├── nlp/
│   │   └── extract_sentiment_ollama.py # Zero-shot LLM (Llama-3 8B) extraction pipeline ([-1, 1], T=0.0)
│   └── utils/
│       └── metrics.py                # Downside semi-dev, CVaR 0.95, Moving-Block Bootstrap
├── .gitignore
├── environment.yml                   # Conda environment definition
├── LICENSE                           # MIT License
├── README.md                         # This file
└── requirements.txt                  # PIP dependencies
```

---

## 🚀 Quick Start & Installation

### 1. Clone the repository
```bash
git clone https://github.com/MannyKing55/mibel-drl-procurement.git
cd mibel-drl-procurement
```

### 2. Environment Setup
Using Conda:
```bash
conda env create -f environment.yml
conda activate mibel-drl-procurement
```
Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Reproducing Key Results

To execute the verified dynamic evaluation script calculating multi-seed savings, statistical dispersion, and downside risk profiles directly from the raw trajectory logs:
```bash
python scripts/reproduce_all_tables.py
```

This dynamically verifies:
1. **Multi-Seed Walk-Forward Retraining (Table 13)**: Calculates 2026 savings across all 5 seeds (`-12.18%`, `-0.71%`, `+18.66%`, `+17.47%`, `+8.34%`), confirming the multi-seed mean of `+6.32%`, the sample standard deviation $s = 12.96\%$ (`ddof=1`, Table 13), and population standard deviation $\sigma = 11.60\%$ (`ddof=0`, Table 15/Abstract).
2. **Multi-Year Horizon (Table 11)**: Verifies the `+6.04%` multi-year Walk-Forward savings across the full 374-day horizon.
3. **Downside Risk Profile (Table 14)**: Computes Downside Semi-deviation ($\sigma_-$) and Maximum Drawdown, demonstrating that M8-W1 DRL reduces downside semi-deviation by $33.1\%$ ($13.68 \to 9.15$ EUR/MWh) and maximum drawdown by $17.3\%$ ($225.45 \to 186.45$ EUR/MWh) relative to the mechanical $D_2$ baseline.

---

## 🧠 Zero-Shot LLM News Sentiment Extraction

The sentiment extraction pipeline (`src/nlp/extract_sentiment_ollama.py`) interfaces with a locally hosted **Meta Llama-3 8B-Instruct** instance via [Ollama](https://ollama.ai/) ($Q4\_K\_M$ quantization). It extracts 10 fine-grained continuous state features ($\phi_1, \dots, \phi_{10} \in [-1.0, +1.0]$) from daily energy and macroeconomic news using a strict JSON schema, deterministic greedy decoding ($T = 0.0$), and daily retrospective aggregation (strictly news published up to day $t-1$).

---

## ⚖️ License & Double-Blind Peer Review

- **Code & Processed Data License:** [MIT License](LICENSE).
- **Double-Blind Review Notice:** In accordance with EAAI submission guidelines, an anonymous snapshot with scrubbed author and commit metadata is maintained via [Anonymous 4open Science](https://anonymous.4open.science/r/mibel-drl-procurement-EB59/). The public GitHub repository will be updated upon final paper acceptance.
