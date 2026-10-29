import pandas as pd
import numpy as np

def evaluate_systematic_baselines(df_full: pd.DataFrame, year=None, q=4800.0):
    """
    Evaluates passive mechanical procurement rules:
    - Pure SPOT (Ref.)
    - Passive D1 (buy 1-day ahead forward daily)
    - Passive D2 (buy 2-day ahead forward daily)
    - Passive D3 (buy 3-day ahead forward daily)
    Unhedged weekend/holiday delivery days clear at daily SPOT.
    """
    df = df_full.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if year is not None:
        df = df[df['Date'].dt.year == year].reset_index(drop=True)

    df_spot = df_full.set_index(pd.to_datetime(df_full['Date']))['Precio_SPOT_ES']

    results = {}
    for c, offset in [('Pure_SPOT', 0), ('D1', 1), ('D2', 2), ('D3', 3)]:
        total_spot_cost = 0.0
        total_strat_cost = 0.0
        daily_pnls = []

        for _, row in df.iterrows():
            t_date = row['Date']
            deliv_date = t_date if offset == 0 else t_date + pd.Timedelta(days=offset)
            if deliv_date in df_spot.index:
                spot_p = float(df_spot.loc[deliv_date])
                strat_p = spot_p
                if offset > 0 and c in row and pd.notna(row[c]) and float(row[c]) > 0:
                    strat_p = float(row[c])
                
                total_spot_cost += spot_p * q
                total_strat_cost += strat_p * q
                daily_pnls.append(spot_p - strat_p)

        pnl_arr = np.array(daily_pnls)
        cum_pnl = np.cumsum(pnl_arr)
        max_dd = (np.maximum.accumulate(cum_pnl) - cum_pnl).max() if len(cum_pnl) > 0 else 0.0
        neg_p = pnl_arr[pnl_arr < 0]
        sigma_m = np.sqrt(np.mean(neg_p**2)) if len(neg_p) > 0 else 0.0
        cvar = np.mean(pnl_arr[pnl_arr <= np.percentile(pnl_arr, 5)]) if len(pnl_arr) > 0 else 0.0

        savings_eur = total_spot_cost - total_strat_cost
        savings_pct = (savings_eur / total_spot_cost) * 100.0 if total_spot_cost > 0 else 0.0
        win_rate = (pnl_arr > 0).mean() * 100.0 if len(pnl_arr) > 0 else 0.0

        results[c] = {
            "Total_Cost_EUR": total_strat_cost,
            "Savings_EUR": savings_eur,
            "Savings_Pct": savings_pct,
            "Win_Rate": win_rate,
            "Max_Drawdown": max_dd,
            "Downside_SemiDev": sigma_m,
            "CVaR_0.95": cvar
        }
    return pd.DataFrame(results).T
