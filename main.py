import requests

COINBASE_URL = 'https://api-public.sandbox.pro.coinbase.com'
PRODUCT_ENDPOINT = '/products'

def make_request(endpoint, product=None):
    r = requests.get(COINBASE_URL + endpoint)
    if r.status_code == 200:
        return r.json()
    else:
        return False

def main():
    print(make_request(PRODUCT_ENDPOINT))
    print(make_request(PRODUCT_ENDPOINT+'/BTC-USD/ticker'))
    r = requests.get(COINBASE_URL + '/products/BTC-USD/candles',
                        params={'start':'2020-01-01','end':'2020-01-02','granularity':3600})
    print(r.status_code)
    print(r.content)

if __name__ == '__main__':
    main()