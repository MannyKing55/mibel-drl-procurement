import numpy as np
import pandas as pd

def compute_procurement_metrics(df_results: pd.DataFrame, spot_col='Baseline_Cost', bot_col='Bot_Cost', q=4800.0):
    """
    Computes audited procurement metrics:
    - Total Cost (€)
    - Savings (€ and %)
    - Win Rate (%)
    - Maximum Drawdown (€/MWh)
    - Downside Semi-deviation (sigma_-)
    - CVaR 0.95 (Expected Shortfall at 5% lower tail)
    """
    tot_spot = df_results[spot_col].sum()
    tot_bot = df_results[bot_col].sum()
    tot_savings_eur = tot_spot - tot_bot
    savings_pct = (tot_savings_eur / tot_spot) * 100.0 if tot_spot != 0 else 0.0

    # Daily unit savings g_t in EUR/MWh
    daily_spot_price = df_results[spot_col] / q
    daily_bot_price = df_results[bot_col] / q
    g_t = daily_spot_price - daily_bot_price

    win_rate = (g_t > 0).mean() * 100.0
    cum_pnl = g_t.cumsum()
    max_dd = (cum_pnl.cummax() - cum_pnl).max()

    neg_g = g_t[g_t < 0]
    sigma_minus = np.sqrt(np.mean(neg_g**2)) if len(neg_g) > 0 else 0.0

    q05 = g_t.quantile(0.05)
    cvar_05 = g_t[g_t <= q05].mean() if len(g_t[g_t <= q05]) > 0 else 0.0

    return {
        "Total_Spot_EUR": tot_spot,
        "Total_Bot_EUR": tot_bot,
        "Savings_EUR": tot_savings_eur,
        "Savings_Pct": savings_pct,
        "Win_Rate_Pct": win_rate,
        "Max_Drawdown_EUR_MWh": max_dd,
        "Downside_SemiDev_sigma_minus": sigma_minus,
        "CVaR_0.95_ES": cvar_05
    }

def moving_block_bootstrap_ci(df_results: pd.DataFrame, spot_col='Baseline_Cost', bot_col='Bot_Cost', B=5000, b=7, seed=42):
    """
    Moving-block bootstrap confidence interval for percentage savings.
    """
    np.random.seed(seed)
    n = len(df_results)
    n_blocks = int(np.ceil(n / b))
    blocks = [df_results.iloc[i : i + b] for i in range(n - b + 1)]
    boot_savings = []
    for _ in range(B):
        sampled = [blocks[idx] for idx in np.random.randint(0, len(blocks), size=n_blocks)]
        boot_df = pd.concat(sampled, ignore_index=True).iloc[:n]
        s_spot = boot_df[spot_col].sum()
        s_bot = boot_df[bot_col].sum()
        boot_savings.append(((s_spot - s_bot) / s_spot) * 100.0)

    return np.percentile(boot_savings, 2.5), np.percentile(boot_savings, 97.5)
