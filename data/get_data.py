import yfinance as yf
import pandas as pd

stocks = ['ADA-USD', 'BCH-USD','BNB-USD','BTC-USD','DASH-USD','EOS-USD','ETH-USD','MIOTA-USD','LINK-USD','LTC-USD','TRX-USD','USDT-USD','XLM-USD','XMR-USD','XRP-USD']

def get_data(start, end):
    df0 = None
    for i in stocks:
        df = yf.download(i, start=start, end=end, progress=False)
        if df0 is not None:
            df0 = pd.merge(df0, df['Close'], left_index=True, right_index=True)
        else:
            df0 = pd.DataFrame(df['Close'])

    df0.columns = stocks
    return df0



if __name__ == '__main__':
    dftrain = get_data('2018-01-02','2019-01-01')
    dftrain.to_csv('crypto_portfolio_train.csv', sep=',', encoding='utf-8')

    dftest = get_data('2019-01-02','2019-07-01')
    dftest.to_csv('crypto_portfolio_test.csv', sep=',', encoding='utf-8')
