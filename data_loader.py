
import sys
import pandas as pd
import numpy as np
import os
# TinySoft path
sys.path.append("C:/Program Files/Tinysoft/Analyse.NET")
import TSLPy3 as ts

def tsbytestostr(data):
        if isinstance(data, (tuple)) or isinstance(data, (list)):
            lendata = len(data)
            ret = []
            for i in range(lendata):
                ret.append(tsbytestostr(data[i]))
        elif isinstance(data, (dict)):
            lendata = len(data)
            ret = {}
            for i in data:
                ret[tsbytestostr(i)] = tsbytestostr(data[i])
        elif isinstance(data, (bytes)):
            ret = data.decode("gbk")
        else:
            ret = data
        return ret


def get_price_data(start_date: str, end_date: str) -> pd.DataFrame:
    ts.DefaultConnectAndLogin("test")
    print(ts.RemoteExecute("return 1;", {}))

    ts_str = f"""
SetSysParam(pn_precision(), 6);
stocks:=getbk('A股;终止上市');
st_date:=StrToDate('{start_date}');
en_date:=StrToDate('{end_date}');
setsysparam(pn_rate(), 1);
Setsysparam(pn_rateday(),-1);
setsysparam(pn_cycle(), cy_day());

data:=select
    datetimetostr(['date']) as 'date',
    ['StockID'] as 'ts_code',
    ['open'],
    ['price'] as 'close',
    ['amount'],
    ['vol']
from markettable datekey st_date to en_date of stocks end;
return data;
"""
    raw = ts.RemoteExecute(ts_str, {})
    data_list = tsbytestostr(raw[1])
    data = pd.DataFrame(data_list)

    if data.empty:
        print(f"{start_date} ~ {end_date} no return data")
        return data  

# Ensure that all necessary fields are included; otherwise, an error will be reported
    required_cols = ['date', 'ts_code', 'open', 'close', 'amount', 'vol']
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Missing fields：{col}")

    data['trade_date'] = pd.to_datetime(data['date'])
    data['amt_per_vol'] = data['amount'] / data['vol'].replace(0, np.nan)
    data = data[['trade_date', 'ts_code', 'open', 'close', 'amount', 'vol', 'amt_per_vol']]
    return data



def get_hs300_index_data(start_date: str, end_date: str, exponent: str) -> pd.DataFrame:
    ts.DefaultConnectAndLogin("test")
    print(ts.RemoteExecute("return 1;", {}))
    ts_str = f"""
begint:=StrToDate('{start_date}');
endt:=StrToDate('{end_date}');
SetSysParam(PN_Stock(),'{exponent}');
setsysparam(pn_rate(), 1);
Setsysparam(pn_rateday(),-1);
setsysparam(pn_cycle(), cy_day());

data:=select
    datetimetostr(['date']) as 'date',
    ['StockID'] as 'code',
    ['open'],
    ['price'] as 'close',
    ['high'],
    ['low'],
    ['vol'],
    ['amount']
from markettable DateKey begint to endt Of DefaultStockID() end;
return data;
"""
    # Execute the Tinysoft script and convert the return value
    raw_a = ts.RemoteExecute(ts_str, {})
    data_list = tsbytestostr(raw_a[1])
    data_a = pd.DataFrame(data_list)
    if data_a.empty:
        print(f"Data for the HS 300 is empty:{start_date} ~ {end_date}")
        return pd.DataFrame(columns=['date', 'close', 'ret', 'net_value'])
    required_cols = ['date', 'close']
    for col in required_cols:
        if col not in data_a.columns:
            raise KeyError(f"Missing fields '{col}', The return format of the Tinysoft may be incorrect.")

    # data type convert
    data_a['date'] = pd.to_datetime(data_a['date'])
    data_a = data_a.sort_values('date')
    data_a['close'] = data_a['close'].astype(float)
    data_a['ret'] = data_a['close'].pct_change()
    data_a['net_value'] = (1 + data_a['ret']).cumprod()

    return data_a[['date', 'close', 'ret', 'net_value']]



def get_index_stock(start_date: str, end_date: str, exponent: str) -> pd.DataFrame:
    ts.DefaultConnectAndLogin("test")
    print(ts.RemoteExecute("return 1;", {}))
    ts_str = f"""
SetSysParam(PN_Cycle(), cy_day());       // daily_period
BegT := StrToDate('{start_date}');             //
EndT := StrToDate('{end_date}');             // 
dayList := MarketTradeDayQk(BegT, EndT);  // trading day information

data := array();
k := 0;

for i := 0 to length(dayList) - 1 do
begin
    curDate := dayList[i];
    stocks := GetBkByDate('{exponent}', curDate);  // HS300 stocks
    for j := 0 to length(stocks) - 1 do
    begin
        data[k]['date'] := datetostr(curDate);  // integer data format
        data[k]['code'] := stocks[j];           // stock code
        k := k + 1;
    end;
end;
return data;
"""
    # Execute the Tinysoft script and convert the return value
    data = pd.DataFrame(tsbytestostr(ts.RemoteExecute(ts_str, {})[1]))

    # data type convert
    data['trade_date'] = pd.to_datetime(data['date'])
    data['ts_code'] = data['code']
    return data[['trade_date', 'ts_code']]


def get_ipo_info() -> pd.DataFrame:
    ts.DefaultConnectAndLogin("test")
    ts_str = f"""
     // Initialization parameter settings
     stocks:=getbk('A股;终止上市');    //stock pool
     setsysparam(pn_cycle(),cy_day()); // set period

     // Allocate space
     data:=array();
     k:=0;

     for i:=0 to length(stocks)-1 do
     begin
         // set up parameters
         setsysparam(pn_stock(),stocks[i]); //Set the current security code

         // 
         data[k]['stockcode']:=stocks[i];

         // Obtain stock status data
         data[k, 'IPO_date']:=datetostr(inttodate(base(12017)));     // listing date
         DelistingDate:=StockDelistingDate();     // delisting date
         if DelistingDate<>0.0 then data[k, 'end_IPO_date']:=datetostr(DelistingDate);
         
         //Other indicators can be added
         k++;
     end;

     return data;"""
    raw = ts.RemoteExecute(ts_str, {})
    df = pd.DataFrame(tsbytestostr(raw[1]))
    df = df.rename(columns={'stockcode': 'ts_code'})
    df['IPO_date'] = pd.to_datetime(df['IPO_date'], errors='coerce')
    print(df)
    return df


# ST status information
def get_st_status(start_date: str,end_date: str) -> pd.DataFrame:
    ts.DefaultConnectAndLogin("test")
    print(ts.RemoteExecute("return 1;", {}))
    ts_str = f"""
     // Initialization parameter settings
     begint:= inttodate(20191230); 
     endt:=inttodate(20210101); 
     stocks:=getbk('A股;终止上市');    //stock pool
     setsysparam(pn_cycle(),cy_day()); // set period

     // Allocate space
     data:=array();
     k:=0;

     for i:=0 to length(stocks)-1 do
     begin
         // set parameters
         setsysparam(pn_stock(),stocks[i]); //Set the current security code
         dayList:= stocktradedayqk(begint, endt); //Take the trading time series of the current security

         for j:=0 to length(dayList)-1 do
         begin
             dayEnd:=dayList[j];
             setsysparam(pn_date(),dayEnd);//
             
             //
             data[k]['code']:=stocks[i];
             data[k]['date']:=datetostr(dayEnd);

             // get ST status
             data[k, 'IsST']:=IsST_3(dayEnd);     // ST judgement

             //Other indicators can be added
             k++;
         end;
     end;

     return data;"""
    data = pd.DataFrame(tsbytestostr(ts.RemoteExecute(ts_str,{})[1]))
    data = data.rename(columns={'code': 'ts_code', 'date': 'trade_date'})
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    return data

# Incremental updating
def update_price_data(start_date: str, end_date: str, save_path: str) -> pd.DataFrame:
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, parse_dates=['trade_date'])
        last_date = existing_df['trade_date'].max()
        fetch_start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"There are individual stock data available. The latest date is {last_date.date()}. Start incremental extraction:{fetch_start_date}")
    else:
        existing_df = None
        fetch_start_date = start_date
        print(f"No historical data, begin full data fetch:{fetch_start_date}")

    if pd.to_datetime(fetch_start_date) > pd.to_datetime(end_date):
        print("The individual stock data is already the latest and does not need to be updated.")
        return existing_df

    new_data = get_price_data(fetch_start_date, end_date)

    if existing_df is not None:
        combined = pd.concat([existing_df, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['trade_date', 'ts_code']).sort_values(['trade_date', 'ts_code'])
    else:
        combined = new_data

    combined.to_csv(save_path, index=False)
    print(f"The individual stock data has been updated and saved to:{save_path}")
    return combined

# HS300 data incremental updating
def update_hs300_data(start_date: str, end_date: str, save_path: str,exponent: str) -> pd.DataFrame:
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, parse_dates=['date'])
        last_date = existing_df['date'].max()
        fetch_start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"We already have the data for the HS 300, and the latest date is {last_date.date()}, Start incremental extract：{fetch_start_date}")
    else:
        existing_df = None
        fetch_start_date = start_date
        print("No historical data, begin full data fetch")

    if pd.to_datetime(fetch_start_date) > pd.to_datetime(end_date):
        print("The HS 300 stock data is already the latest and does not need to be updated.")
        return existing_df

    new_data = get_hs300_index_data(fetch_start_date, end_date,exponent)

    if existing_df is not None:
        combined = pd.concat([existing_df, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date')
    else:
        combined = new_data

    combined.to_csv(save_path, index=False)
    print(f"HS 300 data has been updated and saved to:{save_path}")
    return combined

# Obtaining index constituent stocks
def get_index_members_bydate(start_date: str, end_date: str, exponent_code: str) -> pd.DataFrame:
    ts.DefaultConnectAndLogin("test")
    print(ts.RemoteExecute("return 1;", {}))

    ts_str = f"""
begint := StrToDate('{start_date}');
endt := StrToDate('{end_date}');
dayList := StockTradeDayQk(begint, endt);
data := array();
k := 0;
for i := 0 to length(dayList) - 1 do
begin
    curDate := dayList[i];
    stocks := GetBkByDate('{exponent_code}', curDate);
    for j := 0 to length(stocks) - 1 do
    begin
        data[k]['date'] := DateToStr(curDate);
        data[k]['code'] := stocks[j];
        k := k + 1;
    end;
end;
return data;
"""

    raw = ts.RemoteExecute(ts_str, {})
    data_list = tsbytestostr(raw[1])
    df = pd.DataFrame(data_list)

    if df.empty:
        print(f"The data of the index constituent stocks is empty:{start_date} ~ {end_date}")
        return pd.DataFrame(columns=['date', 'code'])

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)

    return df[['date', 'code']]