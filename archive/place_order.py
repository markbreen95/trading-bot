# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:37:56 2018

@author: som402
"""

import json, hmac, hashlib, time, requests, base64
from requests.auth import AuthBase

# Create custom authentication for Exchange
class GDAXRequestAuth(AuthBase):
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
    
    def __call__(self, request):
        timestamp = str(time.time())
        message = timestamp + request.method + request.path_url + (request.body or '')
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest())
        request.headers.update({
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        })
        return request

API_KEY = "<YOUR_API_KEY>"
API_SECRET = "<API_SECRET>"
API_PASS = "<API_PASS>"

api_url = 'https://api.gdax.com/'
auth = GDAXRequestAuth(API_KEY, API_SECRET, API_PASS)

# Get accounts
r = requests.get(api_url + 'accounts', auth=auth)
print (r.json())

# Place an order
order = {
    'size': 1.0,
    'price': 1.0,
    'side': 'buy',
    'product_id': 'BTC-USD',
}
r = requests.post(api_url + 'orders', json=order, auth=auth)
print( r.json() )