"""
Zero-Shot LLM Macroeconomic & Energy News Sentiment Extraction Pipeline.
Uses Meta Llama-3 8B-Instruct (Q4_K_M) via local Ollama API (v0.1.32).
Outputs structured 10-dimensional continuous sentiment vectors phi_1 to phi_10
in [-1.0, +1.0] for the Multi-Horizon Procurement MDP.
Strictly daily retrospective timestamping (only news published up to day t-1).
"""
import json
from typing import Any, Dict, Optional
import requests

SYSTEM_PROMPT = """You are an expert quantitative energy market analyst for the Iberian Electricity Market (MIBEL).
Analyze the provided daily macroeconomic and energy news corpus and extract 10 fine-grained continuous sentiment dimensions normalized to [-1.0, +1.0]:
- -1.0: Strongly bearish (downward price pressure, oversupply, weak demand, low gas/CO2 costs).
-  0.0: Neutral market outlook.
- +1.0: Strongly bullish (upward price pressure, tightness, strong industrial demand, high gas/CO2 costs, regulatory friction).

The 10 continuous dimensions (phi_1 to phi_10) are:
1. phi_1_short_term: Short-term market outlook (1 to 14 days; weather spikes, unplanned outages, spot tightness).
2. phi_2_medium_term: Medium-term outlook (1 to 3 months; seasonal hydro/gas storage trajectory, maintenance schedules).
3. phi_3_long_term: Long-term outlook (6+ months; macroeconomic cycle, structural policy, renewable additions).
4. phi_4_confidence: Self-assessed extraction confidence score in [0.0, 1.0].
5. phi_5_gas_lng: TTF and Iberian LNG supply pressure and price expectations in [-1.0, +1.0].
6. phi_6_co2_ets: EU ETS carbon allowance price momentum and compliance pressure in [-1.0, +1.0].
7. phi_7_regulatory: MIBEL and European regulatory/fiscal policy impact in [-1.0, +1.0].
8. phi_8_generation_mix: Renewable dispatch (solar/wind) and nuclear/hydro availability in [-1.0, +1.0].
9. phi_9_geopolitical: Geopolitical tension and cross-border trade friction in [-1.0, +1.0].
10. phi_10_demand_macro: Macroeconomic demand indicators and industrial power consumption in [-1.0, +1.0].

Output STRICTLY valid JSON matching this schema with no introductory or trailing text:
{
  "phi_1": <float in [-1.0, 1.0]>,
  "phi_2": <float in [-1.0, 1.0]>,
  "phi_3": <float in [-1.0, 1.0]>,
  "phi_4": <float in [0.0, 1.0]>,
  "phi_5": <float in [-1.0, 1.0]>,
  "phi_6": <float in [-1.0, 1.0]>,
  "phi_7": <float in [-1.0, 1.0]>,
  "phi_8": <float in [-1.0, 1.0]>,
  "phi_9": <float in [-1.0, 1.0]>,
  "phi_10": <float in [-1.0, 1.0]>,
  "justification": "<max 100 words explaining driver divergence across horizons>"
}
"""


def extract_sentiment_scores(
    news_text: str,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "llama3:8b-instruct-q4_K_M",
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    """
    Extracts zero-shot 10-dimensional continuous sentiment from daily news text.
    Configured with temperature=0.0 for deterministic greedy decoding.
    """
    prompt = f"{SYSTEM_PROMPT}\n\n[INICIO_NOTICIA_REAL]\n{news_text.strip()}\n[FIN_NOTICIA_REAL]\n\nJSON Output:"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "top_p": 0.9,
        },
    }

    response = requests.post(ollama_url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    res_data = response.json()
    parsed = json.loads(res_data["response"])
    return parsed
