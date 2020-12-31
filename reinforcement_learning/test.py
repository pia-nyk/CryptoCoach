import matplotlib.pyplot as plt
import numpy as np
from environment import *
from agent import *
from utils import plot_results

N_ASSETS = 15

batch_size = 10
max_experiences = 100
min_experiences = 1

agent = Agent(N_ASSETS, batch_size, max_experiences, min_experiences)
env = CryptoEnvironment()

window_size = 180


def test():
    actions_baseline, actions_rl = [], []
    result_baseline, result_rl = [], []

    for t in range(window_size, len(env.test_prices)):

        date1 = t-window_size
        s_ = env.get_state(date1, t)
        action = agent.policy(s_)

        weighted_returns, reward = env.get_reward(action, date1, t)
        weighted_returns_baseline, reward_basline = env.get_reward(
            np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

        result_baseline.append(weighted_returns_equal.tolist())
        actions_baseline.append(np.ones(agent.portfolio_size) / agent.portfolio_size)

        result_rl.append(weighted_returns.tolist())
        actions_rl.append(action)
        result_baseline_graph = [item for sublist in result_baseline for item in sublist]
        result_rl_graph = [item for sublist in result_rl for item in sublist]
        plt.figure()
        plt.plot(np.array(result_baseline_graph).cumsum(), color='orange', label='Deep Q-Learning')
        plt.plot(np.array(result_rl_graph).cumsum(), color='blue', label='Baseline')
        plt.legend(loc="upper left")
        plt.show()

if __name__ == '__main__':
    test()
