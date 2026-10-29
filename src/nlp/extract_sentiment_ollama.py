"""
Zero-Shot LLM Macroeconomic & Energy News Sentiment Extraction Pipeline.
Uses Meta Llama-3 8B-Instruct (Q4_K_M) via local Ollama API.
Outputs structured JSON state vectors for the Multi-Horizon Procurement MDP.
"""
import json
import requests
from typing import Dict, Any

SYSTEM_PROMPT = """You are an expert quantitative energy market analyst for the Iberian Electricity Market (MIBEL).
Analyze the provided weekly energy news digest and extract 10 fine-grained numerical sentiment scores (0 to 100):
- 0: Strongly bearish (downward price pressure, oversupply, weak demand, low gas/CO2 costs).
- 50: Neutral.
- 100: Strongly bullish (upward price pressure, tightness, high demand, high commodity costs, regulatory friction).

Output strictly valid JSON matching this schema:
{
  "phi_1_overall": <0-100>,
  "phi_2_short_term": <0-100>,
  "phi_3_medium_term": <0-100>,
  "phi_4_geopolitical": <0-100>,
  "phi_5_regulatory": <0-100>,
  "phi_6_demand_industrial": <0-100>,
  "phi_7_renewable_generation": <0-100>,
  "phi_8_hydro_reserves": <0-100>,
  "phi_9_gas_ttf": <0-100>,
  "phi_10_carbon_eua": <0-100>,
  "justification": "<2-3 concise explanatory sentences>"
}
Do not include any prose outside the JSON object.
"""

def extract_sentiment_scores(news_text: str, ollama_url: str = "http://localhost:11434/api/generate") -> Dict[str, Any]:
    payload = {
        "model": "llama3:8b-instruct-q4_K_M",
        "prompt": f"{SYSTEM_PROMPT}\n\nNews Digest:\n{news_text}\n\nJSON Output:",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "seed": 42
        }
    }
    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    res_json = response.json()
    return json.loads(res_json["response"])
