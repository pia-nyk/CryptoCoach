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

    def get_state(self, start_period, comp_period, is_eval=False):
        """
            Get the data in window t-lookback to t - we operate on window of data as
            states rather than a single data point
        """
        # print("Start period: " + str(start_period), end=" ")
        # print("Completion period: " + str(comp_period))
        assert start_period <= comp_period   #throw error if forming the window isnt possible
        if is_eval:
            curr_state = self.test_prices.iloc[start_period:comp_period]
        else:
            curr_state = self.train_prices.iloc[start_period:comp_period]
        curr_state = self.preprocess_state_data(curr_state, True)
        return curr_state

    def get_reward(self, action, start_period, comp_period, is_eval=False):
        # print("Start period: " + str(start_period), end = " ")
        # print("Comp period: " + str(comp_period))
        weights = action
        if not is_eval:
            state = self.train_prices[start_period:comp_period]
        else:
            state = self.test_prices[start_period:comp_period]
        returns = self.preprocess_state_data(state)
        reward = (state.values[-1] - state.values[0])/state.values[0] #reward is the increase/decrease in returns within the window_size after taking a set of actions
        return np.dot(returns, weights), reward #weighted_returns, rewards
