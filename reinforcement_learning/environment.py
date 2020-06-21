import pandas as pd
import numpy as np
from utils import portfolio

class CryptoEnvironment:

    def __init__(self, train_prices = '../data/crypto_portfolio_train.csv', test_prices='../data/crypto_portfolio_test.csv', capital = 1e6):
        self.capital = capital
        self.train_data = self.load_data(train_prices)
        self.test_data = self.load_data(test_prices)

    def load_data(self, dataset):
        data =  pd.read_csv(dataset)
        try:
            data.index = data['Date']
            data = data.drop(columns = ['Date'])
        except:
            data.index = data['date']
            data = data.drop(columns = ['date'])
        return data

    def preprocess_state(self, state):
        return state

    def get_state(self, t, lookback, is_cov_matrix = True, is_raw_time_series = False, is_eval=False):

        assert lookback <= t
        if not is_eval:
            decision_making_state = self.train_data.iloc[t-lookback:t]
        else:
            decision_making_state = self.test_data.iloc[t-lookback:t]
        decision_making_state = decision_making_state.pct_change().dropna()

        if is_cov_matrix:
            x = decision_making_state.cov()
            return x
        else:
            if is_raw_time_series:
                if not is_eval:
                    decision_making_state = self.train_data.iloc[t-lookback:t]
                else:
                    decision_making_state = self.test_data.iloc[t-lookback:t]
            return self.preprocess_state(decision_making_state)

    def get_reward(self, action, action_t, reward_t, alpha = 0.01, is_eval=False):

        def local_portfolio(returns, weights):
            weights = np.array(weights)
            rets = returns.mean() # * 252
            covs = returns.cov() # * 252
            P_ret = np.sum(rets * weights)
            P_vol = np.sqrt(np.dot(weights.T, np.dot(covs, weights)))
            P_sharpe = P_ret / P_vol
            return np.array([P_ret, P_vol, P_sharpe])
        if not is_eval:
            data_period = self.train_data[action_t:reward_t]
        else:
            data_period = self.test_data[action_t:reward_t]
        weights = action
        returns = data_period.pct_change().dropna()
        rew = (data_period.values[-1] - data_period.values[0]) / data_period.values[0]

        return np.dot(returns, weights), rew
