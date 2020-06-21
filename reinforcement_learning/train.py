from agent import *
from environment import *
import numpy as np
import matplotlib.pyplot as plt

N_ASSETS = 15
agent = Agent(N_ASSETS)
env = CryptoEnvironment()

window_size = 180
episode_count = 300
batch_size = 32
rebalance_period = 90

def train():
    for e in range(episode_count):

        agent.is_eval = False
        data_length = len(env.train_data)

        returns_history = []
        returns_history_equal = []

        rewards_history = []
        equal_rewards = []

        actions_to_show = []

        print("Episode " + str(e) + "/" + str(episode_count), 'epsilon', agent.epsilon)

        s = env.get_state(np.random.randint(window_size+1, data_length-window_size-1), window_size)
        total_profit = 0

        for t in range(window_size, data_length, rebalance_period):

            date1 = t-rebalance_period

            s_ = env.get_state(t, window_size)
            action = agent.act(s_)

            actions_to_show.append(action[0])

            weighted_returns, reward = env.get_reward(action[0], date1, t)
            weighted_returns_equal, reward_equal = env.get_reward(
                np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

            rewards_history.append(reward)
            equal_rewards.append(reward_equal)
            returns_history.extend(weighted_returns)
            returns_history_equal.extend(weighted_returns_equal)

            done = True if t == data_length else False
            agent.memory4replay.append((s, s_, action, reward, done))

            if len(agent.memory4replay) >= batch_size:
                agent.expReplay(batch_size)
                agent.memory4replay = []

            s = s_

        rl_result = np.array(returns_history).cumsum()
        equal_result = np.array(returns_history_equal).cumsum()

    print("Done")
    model_json = agent.train_model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    agent.train_model.save_weights("models/model.h5")

    plt.figure(figsize = (12, 2))
    plt.plot(rl_result, color = 'black', ls = '-')
    plt.plot(equal_result, color = 'red', ls = '--')
    plt.show()

    plt.figure(figsize = (12, 2))
    for a in actions_to_show:
        plt.bar(np.arange(N_ASSETS), a, color = 'red', alpha = 0.25)
        plt.xticks(np.arange(N_ASSETS), env.train_data.columns, rotation='vertical')
    plt.show()

if __name__ == '__main__':
    train()
