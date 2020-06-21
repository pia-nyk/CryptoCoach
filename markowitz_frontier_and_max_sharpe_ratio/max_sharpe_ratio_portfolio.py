import pandas as pd
from efficient_frontier import get_ind_returns, annualized_rets, portfolio_return, portfolio_vol, plot_ef
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def max_sharpe_ratio(er, cov):
    rf = 0.1
    w_msr = msr(rf, er, cov)
    print("Weights corresponding to max sharpe ratio: ")
    print(w_msr)
    r_msr = portfolio_return(w_msr, er)
    vol_msr = portfolio_vol(w_msr, cov)
    #adding the capital market line
    cml_x = [0, vol_msr]
    cml_y = [0, r_msr]

    return cml_x, cml_y

def capital_mkt_line_plot(cml_x, cml_y):
    ax = plot_ef(20, er, cov)
    ax.set_xlim(left=0)
    ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed")
    plt.show()

if __name__ == "__main__":
    rets = get_ind_returns()
    er = annualized_rets(rets, 12)
    cov = rets.cov()
    print(cov.shape)
    cml_x, cml_y = max_sharpe_ratio(er, cov)
    capital_mkt_line_plot(cml_x, cml_y)
