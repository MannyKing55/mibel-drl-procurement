"""
Canonical Gymnasium Environment for Multi-Horizon Electricity Procurement in MIBEL.
Used for all experiments in:
"Adaptive Multi-Horizon Electricity Procurement Using Reinforcement Learning
 and Macroeconomic News Sentiment" (Elsevier EAAI).
"""
import calendar
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class MibelProcurementEnv(gym.Env):
    """
    Gymnasium Decision Environment for Industrial Electricity Procurement in MIBEL.

    Observation Space (Box, shape=(715,)):
      - 7-day lookback window of 45 normalized market, grid, and sentiment features (315 features).
      - 400-day forward procurement inventory vector (400 positions).
        Unhedged positions are encoded with the sentinel value -10.0.
        Covered forward positions are normalized via Z-score standardisation.
      - Total observation dimension: 45 * 7 + 400 = 715.

    Action Space (Discrete(14)):
      - 0: Passive SPOT procurement for delivery on day T+1 (default unhedged clearing).
      - 1: D1 (1-day ahead forward, delivery on T+1).
      - 2: D2 (2-day ahead forward, delivery on T+2).
      - 3: D3 (3-day ahead forward, delivery on T+3).
      - 4-7: W1 to W4 (Weekly forward contracts, covering Monday-Sunday of ISO weeks 1 to 4).
      - 8-13: M1 to M6 (Monthly forward contracts, covering calendar months 1 to 6).

    Execution and Constraints:
      - Overlapping forward actions are rejected without modifying existing contracts.
      - An invalid action penalty of lambda_pen = -20.0 EUR/MWh is applied for overlapping or out-of-bounds orders.
      - Canonical baseload demand: E_load = 24.0 MWh/day (corresponding to 1 MW industrial baseload).
      - Transaction costs: configurable via transaction_cost_eur_mwh (default 0.0, evaluating 0.10 and 0.25 EUR/MWh).
    """

    metadata = {"render_modes": ["human"]}
    INVALID_PENALTY = -20.0

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[Dict[str, Any]] = None,
        initial_inventory: Optional[np.ndarray] = None,
        initial_inventory_source: Optional[np.ndarray] = None,
        baseload_mwh_day: float = 24.0,
        transaction_cost_eur_mwh: float = 0.0,
        inventory_size: int = 400,
        window_size: int = 7,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        if "Date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
            self.df["Date"] = pd.to_datetime(self.df["Date"])

        self.n_steps = len(self.df)
        self.inventory_size = inventory_size
        self.window_size = window_size
        self.baseload_mwh_day = float(baseload_mwh_day)
        self.transaction_cost_eur_mwh = float(transaction_cost_eur_mwh)
        self.action_space = spaces.Discrete(14)

        self.initial_inventory = initial_inventory
        self.initial_inventory_source = initial_inventory_source

        # Identify numerical feature columns
        self.feature_cols: List[str] = [
            c for c in self.df.columns
            if self.df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            and c not in ["Date"]
        ]
        features_raw = self.df[self.feature_cols].values.astype(np.float32)

        # Feature standardisation statistics
        self.global_price_mean = np.float32(0.0)
        self.global_price_std = np.float32(1.0)
        mean_vec = np.zeros(len(self.feature_cols), dtype=np.float32)
        std_vec = np.ones(len(self.feature_cols), dtype=np.float32)

        if scaler is not None:
            self.global_price_mean = np.float32(scaler.get("global_price_mean", 0.0))
            self.global_price_std = np.float32(scaler.get("global_price_std", 1.0))
            stats = scaler.get("feature_stats", {})
            for i, col in enumerate(self.feature_cols):
                if col in stats:
                    mean_vec[i], std_vec[i] = stats[col]
                else:
                    mean_vec[i] = self.global_price_mean
                    std_vec[i] = self.global_price_std

        std_vec_safe = np.where(std_vec == 0, 1.0, std_vec)
        self.features_array = (features_raw - mean_vec) / std_vec_safe
        self.features_array = np.nan_to_num(self.features_array, nan=0.0).astype(np.float32)

        # Spot clearing prices
        self.spot_prices = (
            self.df["Precio_SPOT_ES"].values.astype(np.float32)
            if "Precio_SPOT_ES" in self.df.columns
            else np.zeros(self.n_steps, dtype=np.float32)
        )

        # Contract matrix: D1..M6 (13 forward contracts)
        contract_names = ["D1", "D2", "D3", "W1", "W2", "W3", "W4", "M1", "M2", "M3", "M4", "M5", "M6"]
        self.contract_names = contract_names
        n_contracts = len(contract_names)
        self.contract_matrix = np.full((self.n_steps, n_contracts), np.nan, dtype=np.float32)
        for j, c in enumerate(contract_names):
            if c in self.df.columns:
                self.contract_matrix[:, j] = self.df[c].values
        self.contract_name_to_idx = {c: j for j, c in enumerate(contract_names)}

        # Precompute forward contract start/end offsets and contract indices for each step
        dates = self.df["Date"].tolist()
        self.action_starts = np.full((self.n_steps, 14), -1, dtype=np.int32)
        self.action_ends = np.full((self.n_steps, 14), -1, dtype=np.int32)
        self.action_contract_idx = np.full((self.n_steps, 14), -1, dtype=np.int32)

        for t in range(self.n_steps):
            today = dates[t]
            wd = today.weekday()  # Monday = 0, Sunday = 6
            year, month = today.year, today.month

            for a in range(1, 14):
                if a == 1:
                    s_off, e_off, cname = 1, 1, "D1"
                elif a == 2:
                    s_off, e_off, cname = 2, 2, "D2"
                elif a == 3:
                    s_off, e_off, cname = 3, 3, "D3"
                elif 4 <= a <= 7:
                    weeks_ahead = a - 3
                    days_to_monday = (7 - wd) % 7
                    if days_to_monday == 0:
                        days_to_monday = 7
                    s_off = days_to_monday + (weeks_ahead - 1) * 7
                    e_off = s_off + 6
                    cname = f"W{weeks_ahead}"
                else:
                    months_ahead = a - 7
                    m = month - 1 + months_ahead
                    y = year + m // 12
                    m = m % 12 + 1
                    first_day = date(y, m, 1)
                    s_off = (first_day - today.date()).days
                    last_day_num = calendar.monthrange(y, m)[1]
                    last_day = date(y, m, last_day_num)
                    e_off = (last_day - today.date()).days
                    cname = f"M{months_ahead}"

                if 0 <= s_off and e_off < self.inventory_size:
                    self.action_starts[t, a] = s_off
                    self.action_ends[t, a] = e_off
                    self.action_contract_idx[t, a] = self.contract_name_to_idx.get(cname, -1)

        # Sentinel value for unhedged forward inventory positions
        self.inv_empty_val = np.float32(-10.0)

        # Observation space definition: 45 * 7 + 400 = 715
        self.obs_dim = (len(self.feature_cols) * self.window_size) + self.inventory_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self._obs_buffer = np.empty(self.obs_dim, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        if self.initial_inventory is not None:
            self.inventory = self.initial_inventory.copy()
        else:
            self.inventory = np.zeros(self.inventory_size, dtype=np.float32)

        if self.initial_inventory_source is not None:
            self.inventory_source = self.initial_inventory_source.copy()
        else:
            self.inventory_source = np.zeros(self.inventory_size, dtype=np.int8)

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        t = self.current_step
        start = max(0, t - self.window_size + 1)
        window = self.features_array[start : t + 1]

        if len(window) < self.window_size:
            pad = np.tile(window[0], (self.window_size - len(window), 1))
            window = np.vstack([pad, window])

        n_feat = self.window_size * len(self.feature_cols)
        self._obs_buffer[:n_feat] = window.ravel()

        norm_inv = self._obs_buffer[n_feat:]
        norm_inv[:] = self.inv_empty_val
        mask = self.inventory > 0
        if mask.any() and self.global_price_std > 0:
            norm_inv[mask] = (self.inventory[mask] - self.global_price_mean) / self.global_price_std

        return self._obs_buffer.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        info: Dict[str, Any] = {}
        t = self.current_step
        action_fee = 0.0

        # 1. Execute action and enforce delivery constraints
        if action > 0:
            s_idx = int(self.action_starts[t, action])
            e_idx = int(self.action_ends[t, action])
            c_idx = int(self.action_contract_idx[t, action])

            if s_idx < 0 or c_idx < 0:
                reward += self.INVALID_PENALTY
                info["invalid_reason"] = "Out_of_Bounds"
            else:
                price = self.contract_matrix[t, c_idx]
                if np.isnan(price) or price <= 0:
                    reward += self.INVALID_PENALTY
                    info["invalid_reason"] = "NaN_price"
                else:
                    # Constraint: Overlapping forward orders are rejected without overwriting portfolio
                    if np.any(self.inventory[s_idx : e_idx + 1] > 0):
                        reward += self.INVALID_PENALTY
                        info["invalid_reason"] = "Overlap"
                    else:
                        self.inventory[s_idx : e_idx + 1] = price
                        self.inventory_source[s_idx : e_idx + 1] = action
                        action_fee = self.transaction_cost_eur_mwh

        # 2. Daily physical delivery and procurement cost
        bot_cost = float(self.inventory[0])
        spot_price = float(self.spot_prices[t])
        if np.isnan(spot_price):
            spot_price = 0.0

        if bot_cost == 0:
            bot_cost = spot_price
            info["fuente_precio"] = "SPOT"
        else:
            info["fuente_precio"] = (
                self.contract_names[int(self.inventory_source[0]) - 1]
                if self.inventory_source[0] > 0
                else "SPOT"
            )

        # Reward: Unit savings relative to Spot benchmark minus transaction friction
        if spot_price > 0:
            unit_saving = spot_price - bot_cost
            reward += unit_saving - action_fee

        info["spot_price"] = spot_price
        info["effective_cost"] = bot_cost
        info["unit_saving"] = spot_price - bot_cost
        info["daily_procurement_cost_eur"] = bot_cost * self.baseload_mwh_day

        # 3. Advance forward inventory by one delivery day
        self.inventory[:-1] = self.inventory[1:]
        self.inventory[-1] = 0.0
        self.inventory_source[:-1] = self.inventory_source[1:]
        self.inventory_source[-1] = 0

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        return self._get_obs(), float(reward), done, False, info
