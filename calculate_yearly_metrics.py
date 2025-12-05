# === performance_metrics.py ===
import numpy as np

# Cumulative net value
def calculate_netvalue(pct_series):
    return (1 + pct_series).cumprod()

# Cumulative return
def calculate_return(netvalue):
    return netvalue.iloc[-1] - 1

# Annualized return
def calculate_annualreturn(netvalue, period_num, period=240):
    return netvalue.iloc[-1] ** (period / (period_num - 1)) - 1

# Maximum drawdown
def calculate_mdd(netvalue):
    drawdown = 1 - netvalue / netvalue.cummax()
    return drawdown.max()

# Annualized volatility
def calculate_volatility(pct_series, period=240):
    return pct_series.std() * np.sqrt(period)

# Sharpe ratio
def calculate_sharpe(annual_return, volatility, risk_free_rate=0.025):
    return (annual_return - risk_free_rate) / volatility if volatility != 0 else np.nan

# Information ratio
def calculate_info_ratio(annual_return, volatility):
    return annual_return / volatility if volatility != 0 else np.nan
