# Multi-Horizon Industrial Electricity Procurement under Market Dislocation
### Deep Reinforcement Learning Augmented with LLM-Derived News Sentiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green.svg)](https://gymnasium.farama.org/)

Official replication code and data repository for the manuscript submitted to ***Engineering Applications of Artificial Intelligence* (Elsevier)**.

---

## 📌 Repository Overview

This repository provides full end-to-end reproducibility for the multi-horizon electricity procurement framework evaluated in the Iberian Electricity Market (**MIBEL**: OMIE Spot + OMIP Derivatives) under structural price dislocations.

```
mibel-drl-procurement/
├── configs/
│   └── hpo_best_params.yaml          # Best Optuna hyperparameters for LSTM and GTrXL
├── data/
│   ├── README.md                     # Feature dictionary and column descriptions
│   └── processed/
│       ├── Datasets_finales/         # 8 Factorial Ablation datasets (2023-2025)
│       └── Datasets_finales_2026/    # 8 Out-of-sample evaluation datasets (2026 shock)
├── paper_latex/                      # Clean LaTeX manuscript source (EAAI double-column format)
│   ├── main.tex                      # Anonymous main text
│   ├── title_page.tex                # Title page with declarations
│   ├── highlights.tex                # Official highlights (<= 125 chars)
│   ├── references.bib                # BibTeX references
│   └── figures/                      # High-resolution vector PDF & PNG figures
├── scripts/
│   └── reproduce_all_tables.py       # Standalone reproduction script for key empirical tables
├── src/
│   ├── baselines/
│   │   └── systematic_baselines.py   # Mechanical benchmark rules (SPOT, D1, D2, D3)
│   ├── env/
│   │   └── mibel_procurement_env.py  # Gymnasium Markov Decision Process environment
│   ├── nlp/
│   │   └── extract_sentiment_ollama.py # Zero-shot LLM (Llama-3 8B) extraction pipeline
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
git clone https://github.com/mibel-drl-procurement/mibel-drl-procurement.git
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

To execute the verified evaluation script reproducing the benchmark comparison tables:
```bash
python scripts/reproduce_all_tables.py
```

---

## 🧠 Zero-Shot LLM News Sentiment Extraction

The sentiment extraction pipeline (`src/nlp/extract_sentiment_ollama.py`) interfaces with a locally hosted **Meta Llama-3 8B-Instruct** instance via [Ollama](https://ollama.ai/) ($Q4\_K\_M$ quantization). It extracts 10 fine-grained numerical state features ($\phi_1, \dots, \phi_{10} \in [0, 100]$) from weekly energy news digests using a strict JSON schema and low temperature ($T = 0.1$).

---

## ⚖️ License & Double-Blind Peer Review

- **Code & Processed Data License:** [MIT License](LICENSE).
- **Double-Blind Review Notice:** In accordance with EAAI submission guidelines, this repository is configured for anonymous peer review. An anonymous snapshot with scrubbed author and commit metadata is maintained via [Anonymous GitHub](https://anonymous.4open.science/).
