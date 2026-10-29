import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MIBELEnergyProcurementEnv(gym.Env):
    """
    Gymnasium Environment for Multi-Horizon Industrial Electricity Procurement in MIBEL.
    Supports 14 discrete forward and spot procurement actions with explicit physical delivery inventory.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        baseload_mwh_day: float = 4800.0,
        max_inventory_days: int = 101,
        transaction_cost_pct: float = 0.0015,
        inventory_penalty_coef: float = 0.01,
        lookback_lags: int = 7
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.baseload = baseload_mwh_day
        self.max_inv_days = max_inventory_days
        self.tc_pct = transaction_cost_pct
        self.penalty_coef = inventory_penalty_coef
        self.lookback = lookback_lags

        # Action space: 14 discrete actions
        # 0: Pure SPOT default (no forward hedge)
        # 1-3: Daily forwards D1, D2, D3
        # 4-7: Weekly forwards W1, W2, W3, W4
        # 8-13: Monthly forwards M1, M2, M3, M4, M5, M6
        self.action_space = spaces.Discrete(14)

        # Feature observation columns (excluding Date)
        self.feature_cols = [c for c in df.columns if c not in ['Date']]
        self.num_market_features = len(self.feature_cols)

        # State space: [Market features over lookback L, Inventory vector m_t(h)]
        total_dim = (self.num_market_features * self.lookback) + self.max_inv_days
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.inventory = np.zeros(self.max_inv_days, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.inventory = np.zeros(self.max_inv_days, dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        # Extract historical market features window
        feat_slice = self.df.iloc[self.current_step - self.lookback : self.current_step][self.feature_cols].values
        feat_flat = feat_slice.flatten()
        # Concatenate with current forward delivery inventory
        obs = np.concatenate([feat_flat, self.inventory]).astype(np.float32)
        return obs

    def step(self, action: int):
        t_row = self.df.iloc[self.current_step]
        spot_price = float(t_row['Precio_SPOT_ES'])
        
        # 1. Check if today was already covered by an existing forward contract
        existing_coverage = self.inventory[0]
        if existing_coverage > 0:
            effective_price = existing_coverage
            procurement_cost = effective_price * self.baseload
        else:
            effective_price = spot_price
            procurement_cost = spot_price * self.baseload

        # 2. Shift inventory by 1 day
        self.inventory = np.roll(self.inventory, -1)
        self.inventory[-1] = 0.0

        # 3. Apply selected action
        action_cost = 0.0
        if action > 0:
            contract_map = {
                1: ('D1', 1, 1), 2: ('D2', 2, 1), 3: ('D3', 3, 1),
                4: ('W1', 7, 7), 5: ('W2', 14, 7), 6: ('W3', 21, 7), 7: ('W4', 28, 7),
                8: ('M1', 30, 30), 9: ('M2', 60, 30), 10: ('M3', 90, 30),
                11: ('M4', 120, 30), 12: ('M5', 150, 30), 13: ('M6', 180, 30)
            }
            c_name, offset, duration = contract_map[action]
            if c_name in t_row and not np.isnan(t_row[c_name]) and float(t_row[c_name]) > 0:
                p_quote = float(t_row[c_name])
                # Check overlap penalty
                for d in range(offset - 1, min(offset - 1 + duration, self.max_inv_days)):
                    if self.inventory[d] > 0:
                        action_cost += self.penalty_coef * p_quote * self.baseload
                    self.inventory[d] = p_quote
                # Transaction fee
                action_cost += self.tc_pct * p_quote * self.baseload * duration

        # Reward = Spot Baseline Cost - Actual Total Cost
        spot_baseline_cost = spot_price * self.baseload
        total_cost = procurement_cost + action_cost
        reward = (spot_baseline_cost - total_cost) / (spot_baseline_cost + 1e-6)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        info = {
            "spot_price": spot_price,
            "effective_price": effective_price,
            "spot_baseline_cost": spot_baseline_cost,
            "total_cost": total_cost,
            "savings_eur": spot_baseline_cost - total_cost
        }
        return self._get_obs(), reward, terminated, truncated, info
