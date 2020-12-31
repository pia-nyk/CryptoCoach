from agent import *
from environment import *
import numpy as np
import matplotlib.pyplot as plt

N_ASSETS = 15
batch_size = 10
max_experiences = 100
min_experiences = 1
agent = Agent(N_ASSETS, batch_size, max_experiences, min_experiences)
env = CryptoEnvironment()

window_size = 180
N = 300
# rebalance_period = 90 ignoring rebalancing for now
copy_step = 25
portfolio_size = N_ASSETS
action_size = 3
input_shape = (portfolio_size, portfolio_size, )
hidden_units = [100, 50]
copy_steps = 10
num_expReplay = 0
TargetNet = DQNModel(input_shape, hidden_units, action_size, portfolio_size)

def train():
    global num_expReplay
    for e in range(N):
        agent.is_eval = False
        data_length = len(env.train_prices) #total data available for training

        returns_history = []
        rewards_history = []

        #for equal weight allocation case - base case
        returns_history_equal = []
        equal_rewards = []

        actionBuffer = []

        print("Episode " + str(e) + "/" + str(N), 'epsilon', agent.epsilon)

        random_start = np.random.randint(window_size+1, data_length-window_size-1)
        s = env.get_state(random_start, random_start+window_size) #any state of len window_size
        total_profit = 0
        iter = 0

        for comp_period in range(window_size, data_length):
        #, rebalance_period):

            start_period = comp_period-window_size #-rebalance_period

            s_ = env.get_state(start_period, comp_period) #for first iteration - start till window_size
            action = agent.policy(s_)
            actionBuffer.append(action)

            weighted_returns, reward = env.get_reward(action, start_period, comp_period)
            weighted_returns_equal, reward_equal = env.get_reward(
                np.ones(agent.portfolio_size) / agent.portfolio_size, start_period, comp_period)

            rewards_history.append(reward)
            equal_rewards.append(reward_equal)
            returns_history.extend(weighted_returns)
            returns_history_equal.extend(weighted_returns_equal)

            done = True if comp_period == data_length else False
            agent.add_experience({"s": s, "s2": s_, "a": action, "r": reward, "done": done}) #adding this iteration vars to experience buffer
            num_expReplay+=1

            if num_expReplay >= batch_size: #start training only if there are enough examples in replay buffer
                agent.train(TargetNet.get_model())
            s = s_
            if iter % 10 == 0:
                print(iter)
            iter+=1

            if iter % copy_steps == 0: #copying the weights to TargetNet at specified intervals
                TargetNet.copy_weights(agent.train_model)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        print("Epsilon: " + str(agent.epsilon))

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
    for a in actionBuffer:
        plt.bar(np.arange(N_ASSETS), a, color = 'red', alpha = 0.25)
        plt.xticks(np.arange(N_ASSETS), env.train_data.columns, rotation='vertical')
    plt.show()

if __name__ == '__main__':
    train()
