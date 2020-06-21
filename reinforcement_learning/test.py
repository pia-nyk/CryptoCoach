import matplotlib.pyplot as plt
import numpy as np
from environment import *
from agent import *
from utils import plot_results

N_ASSETS = 15
agent = Agent(N_ASSETS, True)
env = CryptoEnvironment()

window_size = 180
episode_count = 50
batch_size = 32
rebalance_period = 90

def test():
    actions_equal, actions_rl = [], []
    result_equal, result_rl = [], []

    for t in range(window_size, len(env.test_data), rebalance_period):

        date1 = t-rebalance_period
        s_ = env.get_state(t, window_size)
        action = agent.act(s_)

        weighted_returns, reward = env.get_reward(action[0], date1, t)
        weighted_returns_equal, reward_equal = env.get_reward(
            np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

        result_equal.append(weighted_returns_equal.tolist())
        actions_equal.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

        result_rl.append(weighted_returns.tolist())
        actions_rl.append(action[0])
        result_equal_vis = [item for sublist in result_equal for item in sublist]
        result_rl_vis = [item for sublist in result_rl for item in sublist]
        plt.figure()
        plt.plot(np.array(result_equal_vis).cumsum(), color='orange', label='Deep Q-Learning')
        plt.plot(np.array(result_rl_vis).cumsum(), color='blue', label='Baseline')
        plt.legend(loc="upper left")
        plt.show()
        plot_results(result_equal, result_rl, actions_rl, N_ASSETS,env.test_data.columns, 'Deep RL portfolio', './images_test1/rl/', 'series')

if __name__ == '__main__':
    test()
