import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_ind_returns():
    """
    load and format the hedge fund index return
    """
    ind = pd.read_csv('../data/ind30_m_vw_rets.csv', header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def random_weights(n):
    k = np.random.rand(n)
    return k/sum(k)

def random_portfolio(returns):
    p = np.mean(returns, axis=0)
    w = random_weights(returns.shape[1])
    C = returns.cov()
    C = pd.Series(np.diag(C), index=[C.index, C.columns])
    mu = np.dot(w, p.T)
    sigma = np.sqrt(np.dot(w*C, w.T))
    return mu, sigma

def scatter_plot(mu, sigma):
    plt.xlabel("Std")
    plt.xlim([0.005, 0.015])
    plt.ylabel("Mean")
    plt.ylim([0.01, 0.02])
    plt.scatter(mu, sigma, c='green', edgecolors='black')
    plt.show()

if __name__ == "__main__":
    rets = get_ind_returns()
    mu =[]
    sigma = []
    for i in range(7000):
        l_mu, l_sig = random_portfolio(rets)
        mu.append(l_mu)
        sigma.append(l_sig)
    scatter_plot(mu, sigma)
