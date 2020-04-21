import requests
import pandas as pd
from datetime import datetime, timedelta
import time


class Instrument:
    def __init__(self, ticker, base_currency):
        self.ticker = ticker
        self.base_currency = base_currency
        self.pairing = '{}-{}'.format(self.ticker, self.base_currency)
        self.coinbase_url = 'https://api.pro.coinbase.com'
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

    def fetch_candle(self, start_date, end_date, granularity=3600):
        url = '{}/products/{}/candles'.format(self.coinbase_url, self.pairing)
        params = {'start': start_date, 'end': end_date, 'granularity': granularity}
        req = self.make_request(url, params)
        df_historical = pd.DataFrame(req)
        df_historical.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
        df_historical['time'] = df_historical['time'].apply(lambda x: datetime.utcfromtimestamp(x))
        return df_historical.sort_values('time').reset_index(drop=True)

    def fetch_order_book(self, level=1):
        url = '{}/products/{}/book'.format(self.coinbase_url, self.pairing)
        params = {'level': level}
        return self.make_request(url, params)

    def fetch_trades(self):
        url = '{}/products/{}/trades'.format(self.coinbase_url, self.pairing)
        return self.make_request(url)

    def fetch_currencies(self):
        url = '{}/currencies'.format(self.coinbase_url)
        return self.make_request(url)


def main():
    btc = Instrument('BTC', 'USD')
    print(btc.coinbase_url)
    print(btc.get_ticker())

    start_date = '2019-01-01'
    end_date = '2020-02-05'
    date_fmt = '%Y-%m-%d'
    start_dt = datetime.strptime(start_date, date_fmt)
    end_dt = datetime.strptime(end_date, date_fmt)

    while start_dt < end_dt:
        time.sleep(1)
        if (end_dt - start_dt).days > 10:
            int_dt = start_dt + timedelta(days=10)
            # get data
            date_start = datetime.strftime(start_dt, date_fmt)
            date_end = datetime.strftime(int_dt, date_fmt)
            df = btc.fetch_candle(date_start, date_end, 3600)
            df.to_csv('data/btc_data_{}_{}.csv'.format(date_start, date_end), index=False)
            print('Got data for ({}, {})'.format(datetime.strftime(start_dt, date_fmt),
                                                 datetime.strftime(int_dt, date_fmt)))
            start_dt = int_dt + timedelta(days=1)
        else:
            date_start = datetime.strftime(start_dt, date_fmt)
            date_end = datetime.strftime(end_dt, date_fmt)
            df = btc.fetch_candle(date_start, date_end, 3600)
            df.to_csv('data/btc_data_{}_{}.csv'.format(date_start, date_end), index=False)
            print('Got data for ({}, {})'.format(datetime.strftime(start_dt, date_fmt),
                                                 datetime.strftime(end_dt, date_fmt)))
            # get data
            start_dt = end_dt


    print(df)
    #df.sort_values()
    #print(btc.fetch_currencies())
    #print(btc.fetch_trades())

if __name__ == '__main__':
    main()
