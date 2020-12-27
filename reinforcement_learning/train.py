from agent import *
from environment import *
import numpy as np
import matplotlib.pyplot as plt

N_ASSETS = 15
agent = Agent(N_ASSETS)
env = CryptoEnvironment()

window_size = 180
N = 300
batch_size = 32
rebalance_period = 90
copy_step = 25
portfolio_size = N_ASSETS
action_size = 3
input_shape = (portfolio_size, action_size)
hidden_units = [100, 50]

TargetNet = DQN(input_shape, hidden_units, action_size, portfolio_size)

def train():
    for e in range(N):
        agent.is_eval = False
        data_length = len(env.train_data) #total data available for training

        returns_history = []
        rewards_history = []

        #for equal weight allocation case - base case
        returns_history_equal = []
        equal_rewards = []

        actions_to_show = []

        print("Episode " + str(e) + "/" + str(episode_count), 'epsilon', agent.epsilon)

        s = env.get_state(np.random.randint(window_size+1, data_length-window_size-1), window_size) #any state of len window_size
        total_profit = 0
        iter = 0

        for t in range(window_size, data_length, rebalance_period):

            start_period = t-rebalance_period

            s_ = env.get_state(t, window_size)
            action = agent.policy(s_)

            actions_to_show.append(action[0])

            weighted_returns, reward = env.get_reward(action[0], start_period, t)
            weighted_returns_equal, reward_equal = env.get_reward(
                np.ones(agent.portfolio_size) / agent.portfolio_size, start_period, t)

            rewards_history.append(reward)
            equal_rewards.append(reward_equal)
            returns_history.extend(weighted_returns)
            returns_history_equal.extend(weighted_returns_equal)

            done = True if t == data_length else False
            agent.memory4replay.append((s, s_, action, reward, done))

            if len(agent.memory4replay) >= batch_size:
                agent.train(TargetNet)
            s = s_
            iter+=1

            if iter % copy_steps == 0: #copying the weights to TargetNet at specified intervals
                TargetNet.copy_weights(agent.train_model)

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
