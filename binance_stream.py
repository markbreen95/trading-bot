# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:49:07 2020

@author: mark.breen
"""


import requests
import json
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
import datetime as dt
from datetime import datetime, date
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier

BASE_URL = 'https://api.binance.com'

endpoints = {
    'time': '/api/v3/time',
    'kline': '/api/v3/klines'
    }

def fetch_kline_data(symbol, start_date=None, end_date=None, interval='1h'):
    '''Get kline (OHLC) data from Binance
    
    Parameters:
        symbol (str): Trading symbol
        start_date (Datetime.date): Start date
        end_date (Datetime.date): End date
        interval (str): Interval frequency
        
    Returns:
        pd.DataFrame: Pandas dataframe containing OHLC data.
    '''
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
    '''Fetch historical data between a start and end date for a specified 
    symbbol
    
    Parameters:
        symbol (str): Trading symbol
        start_date (Datetime.date): Start date
        end_date (Datetime.date): End date
        
    Returns:
        pd.DataFrame: Pandas dataframe with OHLC data
    '''
    intervals = get_intervals(start_date, end_date)
    
    dfs = []
    for interval in intervals:
        dfs.append(fetch_kline_data('BTCUSDT', start_date = interval[0], end_date = interval[1]))
    return pd.concat(dfs)

def calc_delta(start_date, end_date):
    '''Calculate difference in days between two dates
    
    Parameters:
        start_date (Datetime.date): Start date
        end_date (Datetime.date): End date
        
    Returns:
        int: Number of days between two dates
    '''
    start = isinstance(start_date, date)
    end = isinstance(end_date, date)
    if start and end:
        return (end_date - start_date).days

def moving_average(df, window):
    '''Calculates simple moving average for time series data
    
    Parameters:
        df (pd.DataFrame): Containing time series data
        window (int): Window to calculate SMA over
        
    Returns:
        pd.DataFrame: Pandas dataframe containing SMA data
    '''
    return df.rolling(window).mean()

def exponential_moving_average(df, window):
    '''Calculates exponential moving average for time series data
    
    Parameters:
        df (pd.DataFrame): Containing time series data
        window (int): Window to calculate the EMA over
        
    Returns:
        pd.DataFrame: Pandas dataframe containing EMA data
    '''
    return df.ewm(span=window).mean()


def calc_rsi(df, n=14):
    delta = df['close'].diff()
    #-----------
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    
    RolUp = dUp.rolling(14).mean()
    RolDown = dDown.rolling(14).mean().abs()
      
    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    df['rsi'] = rsi
    return df


def crossover_strategy(df, ma_short, ma_long, l_b, u_b):
    '''Simulate the crossover strategy between two moving averages
    
    Paramters:
        df (pd.DataFrame): Pandas dataframe to simulate strategy for close price
        ma_short (pd.DataFrame): Shorter window moving average data
        ma_long (pd.DataFrame): Longer window moving average data
        l_b (float): Lower bound threshold for simulation 
        u_b (float): Upper bound threshold for simulation
        
    Returns: 
        pd.DataFrame: Pandas dataframe containing trades executed
    '''
    df = df.copy()
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
    '''Get the portfolio time series for a particular strategy
    
    Parameters:
        df (pd.DataFrame): Pricing data
        trades (pd.DataFrame): Dataframe containing trades made
        trade_allocation (float): Propotion of capital to allocate per trade
        starting_btc (float): Starting amount in Bitcoin
        starting_usd (float): Starting amount in USD
        
    Returns:
        pd.DataFrame: Dataframe containing original data along with portfolio timeseries
    '''
    df = df.copy()
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
            if trades.loc[idx, 'trade'] == 'buy':
                if open_positions < max_positions and usd_value > 0:
                    btc_value += trade_allocation * usd_value / row['close']
                    usd_value = usd_value * (1 - trade_allocation)
                    print('Timestamp: {}'.format(idx))
                    print('Buy {} BTC @ ${}, cash remaining ${}\n'.format(btc_value, 
                                                                        row['close'],
                                                                        usd_value))
                    open_positions += 1
            if trades.loc[idx, 'trade'] == 'sell':
                if open_positions > 0 and btc_value > 0:
                    usd_value += btc_value / open_positions * row['close']
                    print('Timestamp: {}'.format(idx))
                    print('Sell {} BTC @ ${}, cash remaining ${}\n'.format(btc_value, 
                                                                        row['close'],
                                                                        usd_value))
                    btc_value = btc_value * (open_positions - 1) / open_positions
                    open_positions -= 1
        portfolio.append(btc_value * df.loc[idx, 'close'] + usd_value)
            
    df['portfolio'] = portfolio
    return df
    

def get_intervals(start_date, end_date):
    '''The max number of days worth of hourly data that can be returns from
    the Binance API is roughly 20. This function will calculate the intermediate
    intervals between two dates if the window is too large
    
    Paramters:
        start_date (Datetime.date): Starting date
        end_date (Datetime.date): Ending date
        
    Returns:
        list: Containing tuples of intervals
    '''
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
    

start_date = date(2019, 1, 1)
end_date = date(2019, 6, 1)

start_date = date(2018, 1, 1)
end_date = date(2019, 4, 20)

btc = get_historical_data('BTCUSDT', start_date=start_date, end_date=end_date)

btc.to_csv('btc_2018_2020.csv', index=None)

ema = exponential_moving_average(btc, 10)
ma = moving_average(btc, 20)

trades = crossover_strategy(btc, ema, ma, 1, 1)

test_df = get_portfolio_ts(btc, trades, 0.8)

plt.plot(ema.close)
plt.plot(ma.close)

plt.plot(test_df.close, label='BTCUSDT')
plt.plot(test_df.portfolio, label='Portfolio')
plt.legend()
plt.show()


'''

    Strategy 2: Classification of next n period returns

'''

returns = btc.pct_change()

window = 5

next_returns = []

for i in range(len(btc)-window):
    tmp_df = returns.loc[returns.index[i+1:i+window+1]]
    next_returns.append(tmp_df['close'].sum())
    btc.loc[btc.index[i], 'next_return'] = tmp_df['close'].sum()


plt.plot(btc.close)
plt.plot(btc.close + btc.next_return)

btc = calc_rsi(btc)


def gen_signal(x, lower_bound, upper_bound):
    if x > upper_bound:
        return 'buy'
    if x < lower_bound:
        return 'sell'
    if x < upper_bound and x > lower_bound:
        return 'hold'

lower_bound = -0.03
upper_bound = 0.03
btc['signal'] = btc['next_return'].apply(lambda x: gen_signal(x, lower_bound, upper_bound))    

buy_points = btc[ btc['signal'] == 'buy'].index
sell_points = btc[ btc['signal'] == 'sell'].index

#buys = btc.loc[buy_points, 'close']
#sells = btc.loc[sell_points, 'close']

sell_trades = btc.loc[sell_points, ['close']]
sell_trades['trade']= ['sell' for x in sell_trades.close]

buy_trades = btc.loc[buy_points, ['close']]
buy_trades['trade']= ['buy' for x in buy_trades.close]

trades = pd.concat([sell_trades, buy_trades], axis=0).sort_index()

test_df = get_portfolio_ts(btc, trades, 0.5)

plt.plot(test_df.close)
plt.plot(test_df.portfolio)

btc['ma'] = ma.close
btc['ema'] = ema.close
model_df = btc.dropna()
x_cols = ['close', 'volume', 'ma', 'ema', 'rsi']
X = model_df[x_cols]
y = model_df.signal

clf_rf = RandomForestClassifier()
clf_rf.fit(X, y)

clf_rf.score(X, y)

y_pred = clf_rf.predict(X)

model_df['trade'] = y_pred
trades = model_df[ model_df['trade'].isin(['buy', 'sell'])]
trades = trades[['trade', 'close']]

test_df = get_portfolio_ts(btc, trades, 0.3, 1.0, 0)

buys = trades[ trades['trade'] == 'buy']
sells = trades[ trades['trade'] == 'sell']

plt.plot(test_df.close, label='BTCUSDT')
plt.plot(test_df.portfolio, label='Portfolio')
plt.scatter(buys.index, buys.close, label='Buys', color='k')
plt.scatter(sells.index, sells.close, label='Sells', color='red')
plt.legend()
plt.show()

# Compare with crossover strategy

crossover_trades = crossover_strategy(btc, ema, ma, 1, 1)

crossover_df = get_portfolio_ts(btc, crossover_trades, 0.8)

plt.plot(crossover_df.portfolio, label='Crossover Strategy')
plt.plot(test_df.portfolio, label='Random Forest')
plt.legend()
plt.show()

plt.plot(0.5*crossover_df.portfolio+0.5*test_df.portfolio)

returns = []
weight_1 = []
weight_2 = []
sharpe = []

for n in range(10000):
    
    a = random.uniform(0, 1)
    b = random.uniform(0, 1)
    
    w1 = a / (a + b)
    w2 = b / (a + b)
    
    portfolio_agg = crossover_df.portfolio*w1 + test_df.portfolio*w2
    
    R = portfolio_agg.pct_change().cumsum()
    
    r = R.diff()
    
    sr = r.mean()/r.std() * np.sqrt(365)
    
    returns.append(portfolio_agg.pct_change().sum())
    weight_1.append(w1)
    weight_2.append(w2)
    sharpe.append(sr)

results_df = pd.DataFrame({'crossover_weight': weight_1, 'rf_weight': weight_2,
              'roi': returns, 'risk': sharpe})

results_df.sort_values('roi')

plt.scatter(results_df.roi, results_df.risk)

results_df[ ['roi', 'risk'] ].max(axis=1)

optimal_portfolio = np.argmax(results_df.roi+results_df.risk)
results_df.iloc[optimal_portfolio, :]

ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
zdata = results_df.roi
xdata = results_df.rf_weight
ydata = results_df.crossover_weight
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

N = 100
R = pd.DataFrame(np.random.normal(size=100)).cumsum()

# Approach 1
r = (R - R.shift(1))/R.shift(1)

# Approach 2
r = R.diff()

sr = r.mean()/r.std() * np.sqrt(252)

'''

    ETHUSDT Charts
    
'''

plt.plot(btc.close, label='BTCUSDT')
plt.scatter(buys.index, buys, label='buys')
plt.scatter(sells.index, sells, label='sells')

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