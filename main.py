from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def SHA256(text):
    return sha256(text.encode('ascii')).hexdigest()


if __name__ == '__main__':
    p_weights = []
    p_returns = []
    p_risk = []
    p_sharpe = []
    count = 5000

    stocks = ['AAPL', 'AMZN', 'MSFT', 'TSLA']
    data = yf.download(stocks, start='2015-01-01')['Close']
    x = data.pct_change()

    for k in range(0, count):
        wts = np.random.uniform(size=len(x.columns))
        wts = wts/np.sum(wts)
        p_weights.append(wts)

        # returns
        mean_ret = (x.mean() * wts).sum()*252
        p_returns.append(mean_ret)

        # volatility
        ret = (x * wts).sum(axis =1)
        annual_std = np.std(ret) * np.sqrt(252)
        p_risk.append(annual_std)

        # Sharpe ratio
        sharpe = (np.mean(ret) / np.std(ret)) * np.sqrt(252)
        p_sharpe.append(sharpe)

        max_ind = np.argmax(p_sharpe)

    # Max Sharpe ratio
    print(p_sharpe[max_ind])

    # weights
    print(p_weights[max_ind])

    s = pd.Series(p_weights[max_ind], index=x.columns)
    s.plot(kind='bar')

    # plot
    plt.subplots()
    plt.scatter(p_risk, p_returns, c=p_sharpe, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')

    plt.scatter(p_risk[max_ind], p_returns[max_ind], color='r', marker='*', s=500)
    plt.show()

