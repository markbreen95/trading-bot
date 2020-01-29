# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:38:25 2018

@author: Mark Breen
"""

import websocket
from json import dumps, loads
from datetime import datetime
import numpy as np
import pandas as pd
import time
from collections import deque
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model

try:
    import thread
except ImportError:
    import _thread as thread


date = deque()
price = deque()

k1 = 15
k2 = 20
k3 = 25
t = 0.00065
btc_wallet = 1.0
dollar_wallet = 0.0
portfolio_val = []

buys_x = []
buys_y = []

sales_x = []
sales_y = []

def on_message(ws, message):
    global dollar_wallet
    global btc_wallet
    parsed_msg = loads(message)
    print(parsed_msg["product_id"], parsed_msg["price"], str(datetime.now()))

    #d.append({"date":time.time(),"price":float(parsed_msg["price"])})
    date.append(time.time())
    price.append(float(parsed_msg["price"]))
    
    if len(price) > k3:        
        X1_train = np.array(list(itertools.islice(date, len(date)-k1, len(date)))).reshape((k1,1))
        y1_train = np.array(list(itertools.islice(price, len(price)-k1, len(price)))).reshape((k1,1))
        
        X2_train = np.array(list(itertools.islice(date, len(date)-k2, len(date)))).reshape((k2,1))
        y2_train = np.array(list(itertools.islice(price, len(price)-k2, len(price)))).reshape((k2,1))
        
        X3_train = np.array(list(itertools.islice(date, len(date)-k3, len(date)))).reshape((k3,1))
        y3_train = np.array(list(itertools.islice(price, len(price)-k3, len(price)))).reshape((k3,1))
                
        X_test = date[-1]
        price_current = price[-1]
        
        clf1 = linear_model.Lasso(alpha=0.1)
        clf1.fit(X1_train, y1_train)
        
        clf2 = linear_model.Lasso(alpha=0.1)
        clf2.fit(X2_train, y2_train)
        
        clf3 = linear_model.Lasso(alpha=0.1)
        clf3.fit(X3_train, y3_train)
        
        y1_pred = clf1.predict(X_test)
        y2_pred = clf2.predict(X_test)
        y3_pred = clf3.predict(X_test)
                
        y1 = clf1.predict(X1_train)
        y2 = clf2.predict(X1_train)
        y3 = clf3.predict(X1_train)
        
        y1_diff = []
        y2_diff = []
        y3_diff = []
        
        diff_train = []
        
        for j in range(1,len(y1)):
            y1_diff.append((y1[j] - y1_train[j-1])/y1_train[j-1])
        
        for j in range(1,len(y2)):
            y2_diff.append((y2[j] - y1_train[j-1])/y1_train[j-1])
        
        for j in range(1,len(y3)):
            y3_diff.append((y3[j] - y1_train[j-1])/y1_train[j-1])
        
        for j in range(1, len(y1_train)):
            diff_train.append((y1_train[j]-y1_train[j-1])/y1_train[j-1])
            
        X_train = np.column_stack((y1_diff, y2_diff, y3_diff))
        
        diff_train = np.array((diff_train))
        diff_train = diff_train.reshape((len(diff_train),1))  

        
        X_train = X_train.reshape((len(X_train),3))                                           
        
        clf = linear_model.Lasso(alpha=1)
        clf.fit(X_train, diff_train)
        
        diff_test = np.column_stack(((y1_pred-price_current)/price_current,
                                    (y2_pred-price_current)/price_current,
                                    (y3_pred-price_current)/price_current))

        mean_delta = clf.predict(diff_test)
        
        if mean_delta > t:
            if dollar_wallet > 0.0:
                print("BUY %f ETH for $%f"%(dollar_wallet/price_current,price_current))
                btc_wallet = 0.997*dollar_wallet / price_current
                dollar_wallet = 0.0
                buys_x.append(X_test)
                buys_y.append(price_current)
                
        if mean_delta < -t:
            if btc_wallet > 0.0:
                print("SELL %f ETH for $%f"%(btc_wallet,price_current))
                dollar_wallet  = 0.997*price_current * btc_wallet
                btc_wallet = 0.0
                sales_x.append(X_test)
                sales_y.append(price_current)
                
def on_open(ws):
    def run(*args):
        params = {
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": ["BTC-EUR"]}]
        }
        ws.send(dumps(params))
    thread.start_new_thread(run, ())

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://ws-feed.gdax.com", on_open=on_open, on_message = on_message)
    ws.run_forever()


data = pd.DataFrame({"date":list(date),"price":list(price)})

data.to_csv('data_02_19_01_btc.csv', index=False)

data = pd.read_csv('data_02_08_07_btc.csv')

import matplotlib.pyplot as plt
    
X = data.loc[:,'date'].values
y = data.loc[:,'price'].values

X = X.reshape((len(X),1))
y = y.reshape((len(y),1))

plt.plot(X, y)
plt.plot(buys_x, buys_y, 'ro')
plt.plot(sales_x, sales_y, 'ko')
plt.show()