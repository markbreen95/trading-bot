# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:49:07 2020

@author: mark.breen
"""


import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime as dt
from datetime import datetime, date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


BASE_URL = 'https://api.binance.com'

endpoints = {
    'time': '/api/v3/time',
    'kline': '/api/v3/klines'
    }

def fetch_kline_data(symbol, start_date=None, end_date=None, interval='1h'):
    if start_date and end_date:
        start_timestamp = time.mktime(start_date.timetuple())
        end_timestamp = time.mktime(end_date.timetuple())
        params = {'symbol': symbol,
                  'interval': interval,
                  'startTime': int(start_timestamp)*1000,
                  'endTime': int(end_timestamp)*1000
                  }
    else:
        params = {'symbol': symbol,
                  'interval': interval}
        
    r = requests.get(BASE_URL + endpoints['kline'], params=params)

    kline = pd.DataFrame(json.loads(r.content))
    kline.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                 'close_time', 'quote_asset_volume', 'no_trades', 
                 'taker_buy_base_asset_vol', 'taker_buy_quote_asset_vol', 
                 'ignore']

    kline = kline.astype({'open': 'float32', 'high': 'float32', 'low': 'float32',
                          'close': 'float32', 'volume': 'float32', 
                          'quote_asset_volume': 'float32', 'no_trades': 'float32',
                          'taker_buy_base_asset_vol': 'float32',
                          'taker_buy_quote_asset_vol': 'float32',
                          'open_time': 'int64'})
    
    kline = kline.drop(['close_time', 'ignore'], axis=1)
    
    kline.open_time = kline.open_time.apply(lambda x: datetime.fromtimestamp(x/1000))
    kline = kline.set_index('open_time', drop=True)

    kline = kline.loc[~kline.index.duplicated(keep='first')]

    return kline

def get_historical_data(symbol, start_date, end_date):        
    intervals = get_intervals(start_date, end_date)
    
    dfs = []
    for interval in intervals:
        dfs.append(fetch_kline_data('BTCUSDT', start_date = interval[0], end_date = interval[1]))
    return pd.concat(dfs)

def calc_delta(start_date, end_date):
    start = isinstance(start_date, date)
    end = isinstance(end_date, date)
    if start and end:
        return (end_date - start_date).days

def moving_average(df, window):
    return df.rolling(window).mean()

def exponential_moving_average(df, window):
    return df.ewm(span=window).mean()

def crossover_strategy(df, ma_short, ma_long, l_b, u_b):
    
    ma_div = ma_short/ma_long 
   
    lower_lim = np.full(len(ma_div), l_b)
    upper_lim = np.full(len(ma_div), u_b)
    
    buys = []
    sells = []
    for i in range(1, len(df.index)):
        if (ma_div.loc[df.index[i-1], 'close'] < lower_lim[i-1] and
            ma_div.loc[df.index[i], 'close'] > lower_lim[i-1]):
            buys.append(df.index[i])
        if (ma_div.loc[df.index[i-1], 'close'] > upper_lim[i-1] and
            ma_div.loc[df.index[i], 'close'] < upper_lim[i-1]):
            sells.append(df.index[i])
    
    sell_trades = df.loc[sells, ['close']]
    sell_trades['trade']= ['sell' for x in sell_trades.close]
    
    buy_trades = df.loc[buys, ['close']]
    buy_trades['trade']= ['buy' for x in buy_trades.close]
    
    trades = pd.concat([sell_trades, buy_trades], axis=0).sort_index()
    
    return trades


def get_portfolio_ts(df, trades, trade_allocation, starting_btc = 1.0, starting_usd = 0.0):
    portfolio = []
    btc_value = starting_btc
    usd_value = starting_usd
    max_positions = int(1 / trade_allocation)
    if starting_btc > 0:
        open_positions = 1
    else:
        open_positions = 0
    for idx, row in df.iterrows():
        if idx in trades.index:
            print('transaction')
            if trades.loc[idx, 'trade'] == 'buy':
                if open_positions < max_positions and usd_value > 0:
                    btc_value += trade_allocation * usd_value / row['close']
                    usd_value = usd_value * (1 - trade_allocation)
                    print('Buy {} BTC @ ${}, cash remaining ${}'.format(btc_value, 
                                                                        row['close'],
                                                                        usd_value))
                    open_positions += 1
            if trades.loc[idx, 'trade'] == 'sell':
                if open_positions > 0 and btc_value > 0:
                    usd_value += btc_value / open_positions * row['close']
                    print('Sell {} BTC @ ${}, cash remaining ${}'.format(btc_value, 
                                                                        row['close'],
                                                                        usd_value))
                    btc_value = btc_value * (open_positions - 1) / open_positions
                    open_positions -= 1
        portfolio.append(btc_value * df.loc[idx, 'close'] + usd_value)
            
    df['portfolio'] = portfolio
    return df
    

def get_intervals(start_date, end_date):
    day_delta = calc_delta(start_date, end_date)
    intervals = []
    if day_delta < 20:
        intervals.append((start_date, end_date))
        return intervals
    else:
        temp_date = start_date + dt.timedelta(days=20)       
        intervals.append((start_date, temp_date))        
        while calc_delta(temp_date, end_date) >= 20:
            temp_date += dt.timedelta(days=20)
            intervals.append((intervals[-1][1]+dt.timedelta(days=1), temp_date))
        intervals.append((temp_date+dt.timedelta(days=1), end_date))
        return intervals
    

start_date = date(2020, 1, 1)
end_date = date(2020, 4, 20)

btc = get_historical_data('BTCUSDT', start_date=start_date, end_date=end_date)

btc.to_csv('btc_2018_2020.csv', index=None)

ema = exponential_moving_average(btc, 10)
ma = moving_average(btc, 20)

port, trades = crossover_strategy(btc, ema, ma, 1, 1)

test_df = get_portfolio_ts(port, trades, 0.8)

plt.plot(ema.close)
plt.plot(ma.close)

plt.plot(test_df.close, label='BTCUSDT')
plt.plot(test_df.portfolio, label='Portfolio')
plt.legend()
plt.show()

plt.plot(btc.close)
plt.plot(ma.close)

lm = LinearRegression()
X = btc.close.values.reshape(-1, 1)
y = eth.close.values.reshape(-1, 1)
lm.fit(X, y)

lm.coef_ # 0.03611594
lm.intercept_ # -33.21228, -47.31891

plt.scatter(btc.close, btc.quote_asset_volume, s=0.5)
plt.plot(X, lm.predict(X))
plt.show()

eth_pred = lm.predict(X)

plt.plot(eth.index, eth_pred, label='eth forecast price')
plt.plot(eth.close)
plt.legend()

np.corrcoef(btc.close, eth.close)

plt.plot(btc.close / eth.close)

"""

Moving Average Plots

"""

ma_10 = ma(btc, 10)
ma_30 = ma(btc, 30)

plt.plot(btc.close, label='BTCUSDT')
plt.plot(ma_10.close, label='MA10')
plt.plot(ma_30.close, label='MA30')
plt.legend()

"""

Exponential Moving Average Plots

"""

ema_10 = btc.ewm(span=10).mean()

plt.plot(ema_10.close, label='EMA10')
#plt.plot(ma_10.close, label='MA10')
plt.plot(ma_30.close, label='MA30')
plt.plot(btc.close)
plt.legend()

buys = []
sells = []
for i in range(1, len(btc.index)):
    if (ma_10.loc[btc.index[i-1], 'close'] < ma_30.loc[btc.index[i-1], 'close'] and
        ma_10.loc[btc.index[i], 'close'] > ma_30.loc[btc.index[i], 'close']):
        buys.append(btc.index[i])
    if (ma_10.loc[btc.index[i-1], 'close'] > ma_30.loc[btc.index[i-1], 'close'] and
        ma_10.loc[btc.index[i], 'close'] < ma_30.loc[btc.index[i], 'close']):
        sells.append(btc.index[i])


plt.scatter(ma_10.close, btc.close)

ma_div = ema_10/ma_30

l_b = 1.0
u_b = 1.0

lower_lim = np.full(len(ma_div), l_b)
upper_lim = np.full(len(ma_div), u_b)

buys = []
sells = []
for i in range(1, len(btc.index)):
    if (ma_div.loc[btc.index[i-1], 'close'] < lower_lim[i-1] and
        ma_div.loc[btc.index[i], 'close'] > lower_lim[i-1]):
        buys.append(btc.index[i])
    if (ma_div.loc[btc.index[i-1], 'close'] > upper_lim[i-1] and
        ma_div.loc[btc.index[i], 'close'] < upper_lim[i]):
        sells.append(btc.index[i])

plt.plot(ma_div.close)
plt.hlines(1.0, ma_div.index.min(), ma_div.index.max())
#plt.plot(returns.close+1.0)
plt.hlines(u_b, ma_div.index.min(), ma_div.index.max(), colors='r')
plt.hlines(l_b, ma_div.index.min(), ma_div.index.max(), colors='r')
plt.show()

plt.plot(ma_div.close)
plt.plot(ma_div)

"""

Simple returns strategy

returns = btc.pct_change()

plt.plot(returns.index, returns.close)
plt.hlines(2*returns.close.std(), returns.index.min(), returns.index.max())
plt.hlines(-2*returns.close.std(), returns.index.min(), returns.index.max())
plt.show()

sells = []
buys = []
for idx, row in returns.iterrows():
    if row['close'] > 2.7*returns.close.std():
        sells.append(idx)
    if row['close'] < -2.7*returns.close.std():
        buys.append(idx)

"""

btc_value = 1.0
usd_starting = 0.0
usd_value = usd_starting

trade_allocation = 0.8
max_positions = int(1 / trade_allocation)

open_positions = 0

portfolio = []

for idx, row in returns.iterrows():
    print(row.close)
    if row['close'] > 0.005:
        if open_positions > 0:
            usd_value += btc_value / open_positions * btc.loc[idx, 'close']
            print('Sell {} BTC @ ${}, cash remaining ${}'.format(btc_value, 
                                                                btc.loc[idx, 'close'],
                                                                usd_value))
            btc_value = btc_value * (open_positions - 1) / open_positions

            open_positions -= 1
    if row['close'] < -0.005:        
        if open_positions < max_positions:
            btc_value += trade_allocation * usd_value / btc.loc[idx, 'close']
            usd_value = usd_value * (1 - trade_allocation)
            print('Buy {} BTC @ ${}, cash remaining ${}'.format(btc_value, 
                                                                btc.loc[idx, 'close'],
                                                                usd_value))
            open_positions += 1
    portfolio.append(btc_value * btc.loc[idx, 'close'] + usd_value)
    
btc['portfolio'] = portfolio

plt.plot(btc.close, label='BTCUSDT')
plt.plot(btc.portfolio, label='Portfolio')
plt.legend()
    
final_btc_value = btc_value * btc.loc[btc.index[-1], 'close']

plt.plot(btc.close)
plt.scatter(sells, btc.loc[sells, 'close'], c='r', label='sells')
plt.scatter(buys, btc.loc[buys, 'close'], c='k', label='buys')
plt.legend()
plt.show()


sell_trades = btc.loc[sells, ['close']]
sell_trades['trade']= ['sell' for x in sell_trades.close]

buy_trades = btc.loc[buys, ['close']]
buy_trades['trade']= ['buy' for x in buy_trades.close]

trades = pd.concat([sell_trades, buy_trades], axis=0).sort_index()

cap_all = np.arange(0.01, 1.0, 0.01)
port_val = []

for trade_allocation in cap_all:
    btc_value = 0.0
    usd_starting = 100.0
    usd_value = usd_starting
    
    #trade_allocation = 0.2
    max_positions = 1 / trade_allocation
      
    open_positions = 0
    
    for idx, row in trades.iterrows():
        if row['trade'] == 'buy':
            if open_positions < max_positions:
                btc_value += trade_allocation * usd_value / row['close']
                usd_value = usd_value * (1 - trade_allocation)
                #print('Buy {} BTC @ ${}, cash remaining ${}'.format(btc_value, 
                #                                                    row['close'],
                #                                                    usd_value))
                open_positions += 1
        if row['trade'] == 'sell':
            if open_positions > 0:
                usd_value += btc_value / open_positions * row['close']
                btc_value = btc_value * (open_positions - 1) / open_positions
                #print('Sell {} BTC @ ${}, cash remaining ${}'.format(btc_value, 
                #                                                    row['close'],
                #                                                    usd_value))
    
                open_positions -= 1
    
    final_btc_value = btc_value * btc.loc[btc.index[-1], 'close']
    
    port_val.append(final_btc_value+usd_value)    
    #print('Final portfolio value is: ${}'.format(final_btc_value + usd_value))

btc_start = btc.loc[btc.index[0], 'close']
btc_end = btc.loc[btc.index[-1], 'close']

market_return = usd_starting * (1+(btc_end-btc_start)/btc_start)
plt.plot(cap_all, port_val)
plt.hlines(market_return, min(cap_all), max(cap_all))

btc_wallet = 0.0
cash_wallet = 100

std_dev = returns.close.std()

for idx, row in btc.iterrows():
    

r = requests.get(BASE_URL + endpoints['time'])

print(r.status_code)
print(r.content)

r = requests.get(BASE_URL + '/api/v3/exchangeInfo')

print(r.content)

for con in r.content:
    print(con)

params = {
    'symbol': 'ETHBTC',
    'interval': '1h',
    'startTime': int(unixtime)*1000
    }

r = requests.get(BASE_URL + endpoints['kline'], params=params)

print(r.status_code)
print(r.headers)
print(r.content)

kline = pd.DataFrame(json.loads(r.content))

kline.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                 'close_time', 'quote_asset_volume', 'no_trades', 
                 'taker_buy_base_asset_vol', 'taker_buy_quote_asset_vol', 
                 'ignore']

kline = kline.astype({'open': 'float32', 'high': 'float32', 'low': 'float32',
                      'close': 'float32', 'volume': 'float32', 
                      'open_time': 'int64'})

kline.open_time = kline.open_time.apply(lambda x: datetime.fromtimestamp(x/1000))

d = date(2020, 4, 10)

unixtime = time.mktime(d.timetuple())

plt.plot(kline.open_time, kline.close)
plt.show()