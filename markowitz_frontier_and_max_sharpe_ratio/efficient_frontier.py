import pandas as pd
from scipy.optimize import minimize
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

def annualized_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compunded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compunded_growth**(periods_per_year/n_periods) - 1


def optimal_weights(n_points, er, cov):
    """
    list  of returns to run the optimizer on to get the weights
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    @ is matrix multiplication
    """
    return weights.T @ returns

def portfolio_vol(weights, comvat):
    """
    Weights -> Volatility
    @ is matrix multiplication
    """
    return (weights.T @ comvat @ weights)**0.5 #weights.T @ comvat @ weights gives us variance

def minimize_vol(target_return, er, cov):
    """
    target_return -> weight
    """
    n = er.shape[0]
    init = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    #constraints for the weights
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er) #the target_return should be the one obtained from portfolio
    }

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(
            portfolio_vol, init,
            args=(cov,), method="SLSQP",
            options = {'disp': False},
            constraints = (return_is_target, weights_sum_to_1),
            bounds = bounds
    )
    return results.x

def plot_ef(n_points, er, cov):
    """
    plots the N asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vol(w, cov) for w in weights]
    rv = pd.DataFrame({
    "returns": rets,
    "volatility": vol
    })
    return rv.plot.line(x="volatility", y="returns", style=".-")


# if __name__ == '__main__':
#     ind = get_ind_returns()
#     er = annualized_rets(ind, 12)
#     cov = ind.cov()
#     l = ["Smoke", "Fin", "Games", "Coal"]
#     plot_ef(25, er, cov)
