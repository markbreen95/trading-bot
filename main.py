import requests
import pandas as pd
from datetime import datetime


class Instrument:
    def __init__(self, ticker, base_currency):
        self.ticker = ticker
        self.base_currency = base_currency
        self.pairing = '{}-{}'.format(self.ticker, self.base_currency)
        self.coinbase_url = 'https://api-public.sandbox.pro.coinbase.com'
        self.product_endpoint = '/products'

    def make_request(self, url, params=None):
        r = requests.get(url=url, params=params)
        print(r.url)
        if r.status_code == 200:
            return r.json()
        else:
            return False

    def get_ticker(self):
        url = '{}/products/{}'.format(self.coinbase_url, self.pairing)
        return self.make_request(url)

    def fetch_historical(self, start_date, end_date, granularity):
        url = '{}/products/{}/candles'.format(self.coinbase_url, self.pairing)
        params = {'start': start_date, 'end': end_date, 'granularity': granularity}
        req = self.make_request(url, params)
        df_historical = pd.DataFrame(req)
        df_historical.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
        df_historical['time'] = df_historical['time'].apply(lambda x: datetime.utcfromtimestamp(x))
        return df_historical


def main():
    btc = Instrument('BTC', 'USD')
    print(btc.coinbase_url)
    print(btc.get_ticker())
    df = btc.fetch_historical('2020-01-01', '2020-01-02', 3600)
    print(df)
    """
    print(make_request(PRODUCT_ENDPOINT))
    print(make_request(PRODUCT_ENDPOINT+'/BTC-USD/ticker'))
    r = requests.get(COINBASE_URL + '/products/BTC-USD/candles',
                        params={'start':'2020-01-01','end':'2020-01-02','granularity':3600})
    print(r.status_code)
    print(r.content)
"""


if __name__ == '__main__':
    main()
