# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:35:18 2018

@author: som402
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from functools import partial
import scipy
from scipy import stats
import statsmodels.api as sm

def compute_log_returns(X):
    n = len(X)
    X_ret = np.zeros((n, 1))
    for i in range(1, n):
        X_ret[i] = np.log(X[i] / X[i-1])
    return X_ret*100

def compute_sigmas(X, initial_sigma, theta):
    a0 = theta[0]
    a1 = theta[1]
    b1 = theta[2]
    
    T = len(X)
    sigmas = np.zeros((T, 1))
    sigmas[0] = initial_sigma
    
    for t in range(1, T):
        sigmas[t] = a0 + a1 * X[t-1]**2 + b1 * sigmas[t-1]
    print(min(sigmas))
    return sigmas

def negative_log_likelihood(X, theta):
    
    T = len(X)
    
    # Estimate initial sigma squared
    initial_sigma = np.var(X)
    
    # Generate the squared sigma values
    sigmas = compute_sigmas(X, initial_sigma, theta)
    
    LogLike = sum([-1/2*np.log(sigmas[t]**2)-1/2*(X[t]**2)/sigmas[t] for t in range(T)])
    return -LogLike

def constraint1(theta):
    return 1 - (theta[1] + theta[2])

def constraint2(theta):
    return theta[1]

def constraint3(theta):
    return theta[2]

def constraint4(theta):
    return theta[0]

def double_exp_smoothing(X, alpha, gamma):
    T = len(X)
    
    S_t = np.zeros((T,1))
    b_t = np.zeros((T,1))
    
    b_t[0] = X[1] - X[0]
    S_t[0] = X[0]
    
    for t in range(1, T):
        S_t[t] = alpha*X[t] + (1-alpha)*(S_t[t-1] + b_t[t-1])
        b_t[t] = gamma*(S_t[t] - S_t[t-1]) + (1-gamma)*b_t[t-1]
        
    return S_t, b_t

def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

data_file_name = "coinbase_fixed"

data = pd.read_csv(data_file_name+".csv")

X = data['Close'].values
X = X.reshape((len(X)), 1)
X_raw = X.astype('float32')

train_size = 300000

X_train_raw = X_raw[:train_size,:]
X_test_raw = X_raw[train_size:(len(X_raw)):,:]

X_train, _ = double_exp_smoothing(X_train_raw, 0.3, 0.3)
X_test, _ = double_exp_smoothing(X_test_raw, 0.3, 0.3)

X_ret_train = compute_log_returns(X_train_raw)
X_ret_test = compute_log_returns(X_test_raw)

objective_train = partial(negative_log_likelihood, X_ret_train)
objective_test = partial(negative_log_likelihood, X_ret_test)
    
cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4})

# Actually do the minimization
result_train = scipy.optimize.minimize(objective_train, (1, 0.5, 0.5),
                        method='SLSQP',
                        constraints = cons)
result_test = scipy.optimize.minimize(objective_test, (1, 0.5, 0.5),
                        method='SLSQP',
                        constraints = cons)
theta_mle_train = result_train.x
theta_mle_test = result_test.x

sigma_hats_train = compute_sigmas(X_ret_train, np.var(X_ret_train), theta_mle_train)
sigma_hats_test = compute_sigmas(X_ret_test, np.var(X_ret_test), theta_mle_test)

look_back = 20

n = 100

rsi = relative_strength(X_train, n)

X_train = X_train[n:]

trainX, trainY = create_dataset(X_train, look_back)

s_hat = sigma_hats_train[(look_back+n+1):]

trainX = np.concatenate((trainX, s_hat, rsi[:-21].reshape((len(s_hat),1))), axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
trainY = scaler.fit_transform(trainY)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back+2), activation='softsign'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adagrad')
model.fit(trainX, trainY, epochs=100, verbose=1)

print(model.summary())

trainPredict = model.predict(trainX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))

plt.plot(trainPredict, label='Training Prediction')
plt.plot(X_train_raw, label='Training Set')
plt.legend()

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


rsi_test = relative_strength(X_test, n)

X_test = X_test[n:]
s_hat_test = sigma_hats_test[n:]


# Implement trading strategy
i = 0
t1 = 0.005
t2 = 0.005

k = look_back
K = look_back

T = len(X_test)

btc_wallet = 1.0
dollar_wallet = 0.0

portfolio_val = []

sales_x = []
sales_y = []
buys_x = []
buys_y = []

X_vals = []

X_pred = []


while i < T - K - 1:
    print(i)
    X_testing = X_test[i:k,:]
    
    X_var = np.var(X_test)
    
    p_current = X_test_raw[k+n, :][0]
    X_vals.append(i)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_testing = np.vstack((X_testing, s_hat_test[k,:], rsi_test[k]))
    X_testing = scaler.fit_transform(X_testing)
    X_testing = np.reshape(X_testing, (X_testing.shape[1], 1, X_testing.shape[0]))
    X_predict = model.predict(X_testing)
    X_predict = scaler.inverse_transform(X_predict)
    X_pred.append(X_predict[0,0])
    delta = (X_predict - p_current)/p_current
    
    if delta > t1:
        if dollar_wallet > 0.0 and p_current > 0:
            btc_wallet = dollar_wallet / p_current
            dollar_wallet = 0
            buys_x.append(i)
            buys_y.append(p_current)
    if delta < -t2:
        if btc_wallet > 0.0:
            dollar_wallet = p_current * btc_wallet
            btc_wallet = 0
            sales_x.append(i)
            sales_y.append(p_current)
    portfolio_val.append(dollar_wallet+p_current*btc_wallet)
        
    
    i+=1
    k+=1
    

init_invest = portfolio_val[0]
plt.plot(range(K, len(X_pred)), X_test_raw[K:len(X_pred),:], label='Actual Price')
#plt.plot(X_vals, X_pred, label='predicted price')
#plt.plot(sales_x, sales_y, 'ro')
#plt.plot(buys_x, buys_y, 'bo')
plt.plot(X_vals, portfolio_val, label='Portfolio Value')
plt.ylabel('Dollars')
#plt.xlim(30000, 50000)
plt.legend()
plt.savefig('lstm_20.eps')
plt.show()

profit_lstm = []
profit_lstm.append(1)

for i in range(len(portfolio_val)):
    profit_lstm.append(100*(portfolio_val[i]-portfolio_val[0]) / portfolio_val[0])

profit_btc = []
profit_btc.append(1)

for i in range(len(X_test_raw)):
    profit_btc.append(100*(X_test_raw[i] - X_test_raw[0])/X_test_raw[0])

xvals = range(len(profit_lstm))

plt.plot(profit_lstm, label='LSTM-GARCH')
plt.plot(profit_lin, label='Lasso-GARCH')
plt.plot(profit_btc, label='Holding Bitcoin')
plt.ylabel('Return on Investment (Percent)')
plt.legend()
plt.savefig('ROI.eps')
plt.show()

print("ROI over interval: %f %%, tolerance=%f" % ((100*(portfolio_val[-1]-init_invest)/init_invest),t1))


# forecast k steps in the future
# k steo ahead prediction

k = 1000

pred = []

X_temp = X_test[100000:100010]

for i in range(k):
    X_original = X_temp[-10:]
    X_original = X_original.reshape((len(X_original), 1))
    sigma_curr = sigma_hats_test[9,:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_present = np.vstack((X_original, sigma_curr))
    X_present = scaler.fit_transform(X_present)
    X_present = np.reshape(X_present, (X_present.shape[1], 1, X_present.shape[0]))
    X_predict = model.predict(X_present)
    X_predict = scaler.inverse_transform(X_predict)[0,0]
    pred.append(X_predict)
    X_temp = np.append(X_temp, X_predict)


portfolio_ret= compute_log_returns(portfolio_val)

btc_ret = compute_log_returns(X_test_raw)

plt.plot(portfolio_ret)
plt.plot(btc_ret)

plt.hist(portfolio_ret, bins=50, density=True)
plt.hist(btc_ret, bins=50, density=True)

plt.boxplot(portfolio_ret)
plt.boxplot(btc_ret)

dp_data = [portfolio_ret, btc_ret]

fig, ax = plt.subplots()

ax.boxplot(dp_data)
plt.ylim(-0.05, 0.05)


objective_port = partial(negative_log_likelihood, portfolio_ret)
objective_btc = partial(negative_log_likelihood, btc_ret)
    
cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4})

# Actually do the minimization
result_port = scipy.optimize.minimize(objective_train, (1, 0.5, 0.5),
                        method='SLSQP',
                        constraints = cons)
result_btc = scipy.optimize.minimize(objective_test, (1, 0.5, 0.5),
                        method='SLSQP',
                        constraints = cons)
theta_mle_port = result_port.x
theta_mle_btc = result_btc.x

sigma_hats_port = compute_sigmas(portfolio_ret, np.var(portfolio_ret), theta_mle_port)
sigma_hats_btc = compute_sigmas(btc_ret, np.var(btc_ret), theta_mle_btc)

plt.plot(portfolio_ret)
plt.plot(sigma_hats_port)

plt.plot(btc_ret)
plt.plot(sigma_hats_btc)

