import pandas as pd
import numpy as np

class CryptoEnvironment:
    def __init__(self, train_prices='../data/crypto_portfolio_train.csv', test_prices='../data/crypto_portfolio_test.csv'):
        self.train_prices = self.get_data(train_prices)
        self.test_prices = self.get_data(test_prices)

    def get_data(self, prices_dir):
        data = pd.read_csv(prices_dir)
        data.index = data['Date'] #the index has to be date for pandas operations like pct_change & cov
        data = data.drop(columns = ['Date'])
        return data

    def preprocess_state_data(self, state, cov=False):
        """
            Given a state, get the data converted to pct_change & find cov
        """
        x = state.pct_change().dropna()
        if cov:
            x = x.cov()
        return x

    def get_state(self, t, lookback, is_eval=False):
        """
            Get the data in window t-lookback to t - we operate on window of data as
            states rather than a single data point
        """
        assert lookback <= t   #throw error if forming the window isnt possible
        if is_eval:
            curr_state = self.test_data.iloc[t-lookback:t]
        else:
            curr_state = self.train_data.iloc[t-lookback:t]
        curr_state = preprocess_state_data(curr_state, True)
        return curr_state

    def get_reward(self, start_period, comp_period, action, is_eval=False):
        weights = action
        if not is_eval:
            state = self.train_prices[start_period:comp_period]
        else:
            state = self.test_prices[start_period:comp_period]
        returns = preprocess_state_data(state)
        reward = (state.values[-1] - state.values[0])/state.values[0] #reward is the increase/decrease in returns within the window_size after taking a set of actions
        return np.dot(returns, weights), reward
