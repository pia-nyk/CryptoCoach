import numpy as np
from model import DQNModel
import random
from keras.models import model_from_json


class Agent:
    def __init__(self, portfolio_size, batch_size, max_experiences, min_experiences, is_eval = False):
        self.portfolio_size = portfolio_size
        self.action_size = 3 # sit, buy, sell
        self.input_shape = (self.portfolio_size, self.portfolio_size, )
        self.is_eval = is_eval

        #replay buffer hyperparameters
        self.expReplayBuffer = {'s':[], 'a':[], 'r':[], 's2':[],'done':[]}
        self.expReplayBufferSize = 0
        self.batch_size = batch_size #for replay buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        #training hyperparameters
        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.05 #decay rate after every iteration

        #models
        self.hidden_units = [100, 50]
        self.train_model = DQNModel(self.input_shape, self.hidden_units, self.action_size, self.portfolio_size).get_model()
        self.test_model = self.get_model()

    def get_model(self):
        """
            Load the saved model
        """
        json_file = open("models/model.json", 'r')
        loaded_json_file = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_json_file)
        loaded_model.load_weights("models/model.h5")
        return loaded_model

    def predictions_to_weights(self, pred):
        """
            Helper function - Convert the model predictions to the form of weights associated with the portfolio stocks
        """
        weights = np.zeros(len(pred))
        raw_weights = np.argmax(pred, axis=-1)

        for stock, action in enumerate(raw_weights): #should be pred
            if action == 0:
                weights[stock] = 0
            elif action == 1:
                weights[stock] = np.abs(pred[stock][0][action]) #bcoz pred is array of arrays
            else:
                weights[stock] = -np.abs(pred[stock][0][action]) #bcoz pred is array of arrays
        return weights

    def policy(self, state):
        if self.is_eval: #testing the model, get the model predictions directly irrespective of epsilon
            pred = self.test_model.predict(np.expand_dims(state.values, 0)) #np.expand_dims is required because we will predict 3 cases from the state position
        else:
            if random.random() <= self.epsilon: #during training, epsilon probability of choosing randomly
                weights = np.random.normal(0, 1, size = (self.portfolio_size, ))
                saved_sum = np.sum(weights)
                weights = weights/saved_sum #sum of all weights should be 1
                return weights
            else:
                pred = self.train_model.predict(np.expand_dims(state.values, 0))
        return self.predictions_to_weights(pred)

    def weights_to_predictions(self, action_weights, rewards, Q_star):
        Q = np.zeros((self.portfolio_size, self.action_size))
        for i in range(self.portfolio_size):
            if action_weights[i] == 0:
                Q[i][0] = rewards[i] + self.gamma * np.max(Q_star[i][0])
            elif action_weights[i] > 0:
                Q[i][1] = rewards[i] + self.gamma * np.max(Q_star[i][1])
            else:
                 Q[i][2] = rewards[i] + self.gamma * np.max(Q_star[i][2])
        return Q

    def train(self, TargetNet):
        # print("Training in progress")
        ids = np.random.randint(low=0, high=len(self.expReplayBuffer['s']), size=self.batch_size) #get batchsize exp data for training
        #store the experience data in vars for easy access
        # states = np.asarray([self.expReplayBuffer['s'][i] for i in ids])
        # actions = np.asarray([self.expReplayBuffer['a'][i] for i in ids])
        # rewards = np.asarray([self.expReplayBuffer['r'][i] for i in ids])
        # states_next = np.asarray([self.expReplayBuffer['s2'][i] for i in ids])
        # dones = np.asarray([self.expReplayBuffer['done'][i] for i in ids])

        for i in range(len(self.expReplayBuffer['s'])):
            state = self.expReplayBuffer['s'][i]
            action = self.expReplayBuffer['a'][i]
            reward = self.expReplayBuffer['r'][i]
            state_next = self.expReplayBuffer['s2'][i]
            done = self.expReplayBuffer['done'][i]
            #predict the q values for the states_next using TargetNet as the variables of that net would be more stable
            # print("Shape: " + str(state_next.shape))
            values_next = np.max(TargetNet.predict(np.expand_dims(state_next, axis=0)), axis=1)
            # print("Action vals")
            # print(action)
            # actual_values = np.where(dones, rewards, rewards+self.gamma*values_next)
            Q_learned_values = self.weights_to_predictions(action, reward, values_next)
            Q_val = TargetNet.predict(np.expand_dims(state, axis=0))
            #Q learing formula
            Q_val = [np.add(a * (1-self.alpha), q * self.alpha) for a, q in zip(Q_val, Q_learned_values)]

            #train the main model
            self.train_model.fit(np.expand_dims(state, 0), Q_val, epochs=1, verbose=0)
            #decrease the exploration rate after every iteration

    def add_experience(self, experience):
        """
            add experience to the expReplayBuffer
        """
        # print("Length: " + str(self.expReplayBufferSize))
        if self.expReplayBufferSize >= self.max_experiences:
            for key in self.expReplayBuffer.keys():
                self.expReplayBuffer[key].pop(0) #remove an old experience to make place for a new one FIFO
        for key, value in experience.items():
            self.expReplayBuffer[key].append(value) #add the new experience
