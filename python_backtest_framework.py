import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from data_loader import update_price_data, update_hs300_data, get_st_status,get_ipo_info,get_index_stock
from year_calculate import (
    calculate_netvalue,
    calculate_return,
    calculate_annualreturn,
    calculate_mdd,
    calculate_volatility,
    calculate_sharpe,
    calculate_info_ratio
)

stocks_choice = 'SH00300'
# 1. Read Amihud feather file
df_wide = pd.read_feather(r"D:\GF实习\单因子框架构建\agru_dailyquote.feather")

# date type process and convert
df_wide.index.name = 'trade_date'
df_wide.index = pd.to_datetime(df_wide.index)
df_wide.columns = [
    col[-2:] + col[:6] if isinstance(col, str) and len(col) >= 9 else col
    for col in df_wide.columns
]

df = df_wide.stack().reset_index()
df.columns = ['trade_date', 'ts_code', 'amihud']
print(df)
print('done')

# Set the backtest date range
date_start = '2019-12-30'
date_end = '2025-07-20'
start_date = pd.to_datetime(date_start)
end_date = pd.to_datetime(date_end)

# Filter stock data by date
df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
#print(df)

# 2. Incrementally extract daily stock price data via Tinysoft
price_save_path = r"D:\GF实习\回测相关代码(单因子)\数据提取csv\all_stock_daily_final_1.csv"
hs300_save_path = r"D:\GF实习\回测相关代码(单因子)\数据提取csv\hs300_index_daily_final_1.csv"

price = update_price_data(date_start, date_end, price_save_path)
hs300_df = update_hs300_data(date_start, date_end, hs300_save_path,"SH000300")
price = price[(price['trade_date'] >= start_date) & (price['trade_date'] <= end_date)]
hs300_df = hs300_df[(hs300_df['date'] >= start_date) & (hs300_df['date'] <= end_date)]

# 3. Merge factor values with closing prices and calculate future returns, keeping SH/SZ exchange stocks only
df = df.merge(price, on=['trade_date', 'ts_code'], how='left')
df = df[df['ts_code'].str.startswith('SZ') | df['ts_code'].str.startswith('SH')]
df = df.sort_values(['ts_code', 'trade_date'])
df = df[df['amihud'].notna() & (df['amihud'] != 0)]

# Filter index constituents by each trading day
if stocks_choice != 'None':
    members_df = get_index_stock(date_start,date_end,stocks_choice)
    df = df.merge(members_df, on=['trade_date', 'ts_code'], how='inner')
# print(df)

# Add ST Situation and IPO date
st_df = get_st_status(20191230,20250720)   # ['ts_code', 'date', 'IsST']
ipo_df = get_ipo_info()    # ['ts_code', 'IPO_date']
#print(st_df)
df = df.merge(ipo_df, on='ts_code', how='left')
df = df.merge(st_df.rename(columns={'date': 'trade_date'}), on=['ts_code', 'trade_date'], how='left')


# Exclude ST stocks
df_before_st = df.shape[0]
df = df[~df['IsST'].astype(str).str.contains('ST', na=False)]
#print(f"[Exclude ST stocks] Exclude Number: {df_before_st - df.shape[0]}, 剩余: {df.shape[0]}")

N = 20
df_before_ipo = df.shape[0]
df = df[df['IPO_date'] <= df['trade_date'] - pd.Timedelta(days=N)]

# 4. Prepare rolling rebalancing data
df['trade_date'] = df.groupby('ts_code')['trade_date'].shift(-1)
df = df.dropna(subset=['trade_date', 'amihud', 'amt_per_vol'])
df = df[df['amt_per_vol'] != 'NaN']

# 5.Perform rolling rebalancing and assign groups
rolling_returns = []
all_dates = sorted(df['trade_date'].unique())
holding_period = 20
group_holdings = {}

for date in all_dates:
    df_day = df[df['trade_date'] == date].copy()
    if df_day.empty:
        continue
    df_day['rank'] = df_day['amihud'].rank(ascending=False)
    df_day['group'] = pd.qcut(df_day['rank'], 10, labels=[f'Q{i}' for i in range(1, 11)])
    df_day['start_date'] = date
    rolling_returns.append(df_day[['ts_code', 'group', 'start_date']])
rolling_df = pd.concat(rolling_returns)

# Calculate daily Q1 turnover in rolling rebalancing
holding_period = 20
rebalance_dates = sorted(rolling_df['start_date'].unique())

turnover_records = []

q1_dict = {
    date: set(rolling_df[(rolling_df['start_date'] == date) & (rolling_df['group'] == 'Q1')]['ts_code'])
    for date in rebalance_dates
}

for i in range(len(rebalance_dates) - holding_period):
    date_t = rebalance_dates[i]
    date_tp = rebalance_dates[i + holding_period]

    q1_t = q1_dict.get(date_t, set())      # Q1 holding stocks on day t
    q1_tp = q1_dict.get(date_tp, set())    # Q1 holding stocks on day t+20

    if not q1_t:
        continue
    turnover = len(q1_t - q1_tp) / len(q1_t)

    turnover_records.append({
        'date': date_tp,    
        'turnover': turnover
    })

# DataFrame Construct
q1_turnover_df = pd.DataFrame(turnover_records)
print("Q1 turnover rate:")
print(q1_turnover_df.head(10))


# 6.Construct daily VWAP data table for all stocks
price_df = price.copy()
price_df['amt_per_vol'] = price_df['amt_per_vol'].replace(['NaN', 'nan', '', 'null', 'NULL', None], np.nan)
price_df = price_df.dropna(subset=['amt_per_vol'])
price_df = price_df.sort_values(['ts_code', 'trade_date'])
vwap_pivot = price_df.pivot(index='trade_date', columns='ts_code', values='amt_per_vol')

 # 7. Expand real portfolio holding-period returns
fee_rate = 0.003
expanded_rows = []

# Trading day list
all_trading_days = vwap_pivot.index.to_list()

for idx, row in rolling_df.iterrows():
    ts_code = row['ts_code']
    group = row['group']
    start_date = row['start_date']

    if start_date not in all_trading_days:
        continue

    try:
        start_idx = all_trading_days.index(start_date)
    except ValueError:
        continue

    # Skip if remaining trading days are shorter than the holding period
    if start_idx + holding_period >= len(all_trading_days):
        continue

    # Get consecutive trading days from the rebalance day for holding_period + 1 (include start and end)
    holding_window = all_trading_days[start_idx : start_idx + holding_period + 1]

    # Expand holding returns day by day
    for i in range(1, len(holding_window)):
        prev_day = holding_window[i - 1]
        curr_day = holding_window[i]

        try:
            prev_vwap = vwap_pivot.loc[prev_day, ts_code]
            curr_vwap = vwap_pivot.loc[curr_day, ts_code]

            if pd.notna(prev_vwap) and pd.notna(curr_vwap) and prev_vwap != 0:

                # add buy fee on the first day, deduct sell fee on the last day
                if i == 1:
                    prev_vwap *= (1 + fee_rate)  
                elif i == holding_period:
                    curr_vwap *= (1 - fee_rate)  

                daily_ret = curr_vwap / prev_vwap - 1

                expanded_rows.append({
                    'holding_date': curr_day,
                    'ts_code': ts_code,
                    'group': group,
                    'daily_ret': daily_ret
                })
        except KeyError:
            continue

print(f"expanded_rows：{len(expanded_rows):,}")
expanded_df = pd.DataFrame(expanded_rows)
print(f"expanded_df done，shape {expanded_df.shape}")


# 8. portfolio daily return & net_value
daily_group_ret = expanded_df.groupby(['holding_date', 'group'])['daily_ret'].mean().unstack()
daily_group_ret = daily_group_ret.sort_index()
net_value_df = (1 + daily_group_ret).cumprod()

print(daily_group_ret)
hs300_ret_df = hs300_df[['date', 'ret']].rename(columns={'date': 'holding_date', 'ret': 'HS300'})
hs300_nv_df = hs300_df[['date', 'net_value']].rename(columns={'date': 'holding_date', 'net_value': 'HS300'})
combined_ret_df = daily_group_ret.merge(hs300_ret_df, on='holding_date', how='left')
combined_nv_df = net_value_df.merge(hs300_nv_df, on='holding_date', how='left')
# 1. Excess return：Q1 - HS300
combined_ret_df['Q1_HS300_excess'] = combined_ret_df['Q1'] - combined_ret_df['HS300']

# 2. Long return：Q1 - Q10
combined_ret_df['Q1_Q10_longshort'] = combined_ret_df['Q1'] - combined_ret_df['Q10']

# 3. Excess net_value：Q1net / HS300net 
combined_nv_df['Q1_HS300_excess'] = combined_nv_df['Q1'] / combined_nv_df['HS300']

#4. Long-short net_value：Q1net / Q10net
combined_nv_df['Q1_Q10_longshort'] = combined_nv_df['Q1'] / combined_nv_df['Q10']

print(combined_ret_df)
print(combined_nv_df)

# Calculation of yearly performance
def analyze_yearly_performance(df, column, start_year, end_year, label=None):
    if label is None:
        label = column  

    date_col = 'holding_date' if 'holding_date' in df.columns else 'date'
    print(date_col)
    results = []

    for year in range(start_year, end_year + 1):
        df_year = df[df[date_col].dt.year == year]
        if len(df_year) < 20 or column not in df_year.columns:
            continue

        pct = df_year[column].dropna()
        print(pct)
        if len(pct) < 2:
            continue

        dates = df_year[date_col].loc[pct.index]
        netvalue = calculate_netvalue(pct)

        netvalue_df = pd.DataFrame({
            'date': dates.values,
            'net_value': netvalue.values
        })
        print(f"\n===== {label} | {year} Annual net value sequence =====")
        print(netvalue_df)

        period_num = len(pct)
        total_ret = calculate_return(netvalue)
        ann_ret = calculate_annualreturn(netvalue, period_num)
        mdd = calculate_mdd(netvalue)
        vol = calculate_volatility(pct)
        sharpe = calculate_sharpe(ann_ret, vol)
        info_ratio = calculate_info_ratio(ann_ret, vol)
        ret_mdd = ann_ret / mdd if mdd != 0 else np.nan

        results.append({
            'Portfolio': label,
            'Year': f"{year}",
            'Total Return': f"{total_ret:.2%}",
            'Annualized Return': f"{ann_ret:.2%}",
            'Maximum Drawdown': f"{mdd:.2%}",
            'Annualized Volatility': f"{vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Information Ratio': f"{info_ratio:.2f}",
            'Return-Drawdown Ratio': f"{ret_mdd:.2f}"
        })

    return pd.DataFrame(results)

start_year = 2020
end_year = 2025

# 1. HS 300
res_hs300 = analyze_yearly_performance(hs300_df, column='ret', start_year=start_year, end_year=end_year, label='HS300')

# 2. Q1 long
res_q1 = analyze_yearly_performance(combined_ret_df, column='Q1', start_year=start_year, end_year=end_year, label='Q1long')

# 3. Q1-HS300excess
res_excess = analyze_yearly_performance(combined_ret_df, column='Q1_HS300_excess', start_year=start_year, end_year=end_year, label='Q1-沪深300')

# 4. Q1-Q10long
res_longshort = analyze_yearly_performance(combined_ret_df, column='Q1_Q10_longshort', start_year=start_year, end_year=end_year, label='Q1-Q10多空')

# Merge to a complete file
year_eva_df = pd.concat([res_hs300, res_q1, res_excess, res_longshort], ignore_index=True)
print(year_eva_df)

# 9. Average Return
mean_group_returns_df = daily_group_ret.mean().to_frame(name='average')
mean_group_returns_df.loc['HS300'] = combined_ret_df['HS300'].mean()
mean_group_returns_df.loc['Q1-HS300 excess'] = combined_ret_df['Q1_HS300_excess'].mean()
order = [
    'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
    'HS300', 'Q1-HS300excess'
]
mean_group_returns_df = mean_group_returns_df.reindex(order)

print(mean_group_returns_df)

# RankIC Calculation
fac = df.pivot(index='trade_date', columns='ts_code', values='amihud')

valid_pairs = df[['trade_date', 'ts_code']].drop_duplicates()
price_filtered = price.merge(valid_pairs, on=['trade_date', 'ts_code'], how='inner')
vwap = price_filtered.pivot(index='trade_date', columns='ts_code', values='amt_per_vol')

# Construct future returns: buy on day t+1 and sell on day t+20
ret = vwap.shift(-21) / vwap.shift(-1) - 1

# Align fac and ret
fac, ret = fac.align(ret, join='inner', axis=0)

# Compute daily cross-sectional Spearman correlations
rankic_series = fac.corrwith(ret, axis=1, method='spearman')

# Build rankic_df and the summary statistics
rankic_df = rankic_series.dropna().to_frame(name='rank_ic').reset_index().rename(columns={'index': 'date'})
rankic_df['rank_ic_cumsum'] = rankic_df['rank_ic'].cumsum()

rankic_mean = rankic_df['rank_ic'].mean()
rankic_std = rankic_df['rank_ic'].std()
rankic_win_rate = (rankic_df['rank_ic'] > 0).mean()
icir = rankic_mean / rankic_std if rankic_std != 0 else np.nan

summary_df = pd.DataFrame({
    'index': ['RankIC Average', 'RankIC_win_rate', 'ICIR'],
    'numerical value': [f'{rankic_mean:.4f}', f'{rankic_win_rate:.2%}', round(icir, 4)]
})
print(rankic_df.head())
print(summary_df)


output_path = r"D:\GF实习\单因子框架构建\backtest_final_2020_沪深股_新amihud_1.xlsx"
combined_nv_df = combined_nv_df.rename(columns={
    'holding_date':'Date',
    'HS300': 'HS 300',
    'Q1_HS300_excess': 'Q1–HS 300 excess',
    'Q1_Q10_longshort':'Q1–Q10 long-short'
})
combined_ret_df = combined_ret_df.rename(columns={
    'holding_date':'Date',
    'HS300': 'HS300',
    'Q1_HS300_excess': 'Q1-HS300excess',
    'Q1_Q10_longshort':'Q1-Q10 long-short'
})
rankic_df = rankic_df.rename(columns={
    'date':'Date',
    'rank_ic': 'RankIC',
    'rank_ic_cumsum': 'Cumulative RankIC'
})

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 1. Daily_net_value
    combined_nv_df.to_excel(writer, sheet_name='net_value', index=False)

    # 2. Daily_return
    combined_ret_df.to_excel(writer, sheet_name='return', index=False)

    # 3. Q1–Q10 Average Return
    mean_group_returns_df.to_excel(writer, sheet_name='Average return by quantile group', index=True)

    # 4. Yearly performance metrics
    year_eva_df.to_excel(writer, sheet_name='Yearly portfolio performance summary', index=True, header=True)

    # 5. Daily RankIC and cumulative RankIC
    rankic_df.to_excel(writer, sheet_name='History RankIC', index=False)

    # 6. RankIC summary statistics
    summary_df.to_excel(writer, sheet_name='RankIC summary statistics', index=False)

    # 7. Historical daily Q1 rebalance stock list
    df_q1 = rolling_df[rolling_df['group'] == 'Q1'].copy()
    df_q1 = df_q1[['start_date', 'ts_code']].sort_values(['start_date', 'ts_code'])

    q1_dict = {}
    for date, group in df_q1.groupby('start_date'):
        q1_dict[date.strftime('%Y-%m-%d')] = group['ts_code'].tolist()

    max_len = max(len(v) for v in q1_dict.values())
    q1_pivot_df = pd.DataFrame({k: v + [np.nan] * (max_len - len(v)) for k, v in q1_dict.items()})
    q1_pivot_df.to_excel(writer, sheet_name='Historical long portfolio constituents', index=False)
    # 8. Q1 turnover per rebalance
    q1_turnover_df.to_excel(writer, sheet_name='Q1 turnover_rate', index=False)
print("All backtest results have been successfully written to Excel, including the daily Q1 long portfolio rebalance table.")
