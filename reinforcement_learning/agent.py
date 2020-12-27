import numpy as np
from model import DQN
import random


class Agent:
    def __init__(self, portfolio_size, batch_size, max_experiences, min_experiences, is_eval = False):
        self.portfolio_size = portfolio_size
        self.action_size = 3 # sit, buy, sell
        self.input_shape = (self.portfolio_size, self.action_size, )
        self.is_eval = is_eval

        #replay buffer hyperparameters
        self.expReplayBuffer = {'s':[], 'a':[], 'r':[], 's2':[],'done':[]}
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
        self.train_model = DQN(self.input_shape, self.hidden_units, self.action_size, self.portfolio_size)
        self.test_model = get_model()

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
        pass

    def policy(self, state):
        if self.is_eval: #testing the model, get the model predictions directly irrespective of epsilon
            pred = self.test_model.predict(np.expand_dims(state.values, 0)) #np.expand_dims is required because we will predict 3 cases from the state position
        else:
            if random.random() <= self.epsilon: #during training, epsilon probability of choosing randomly
                weights = np.random.normal(0, 1, size = (self.portfolio_size, ))
                saved_sum = np.sum(weights)
                weights = weights/saved_sum #sum of all weights should be 1
                return weights, saved_sum
            else:
                pred = self.train_model.predict(np.expand_dims(state.values, 0))
        return self.predictions_to_weights(pred)

    def train(self, TargetNet):
        ids = np.random.randint(low=0, high=len(self.expReplayBuffer['s']), size=self.batch_size) #get batchsize exp data for training
        #store the experience data in vars for easy access
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        #predict the q values for the states_next using TargetNet as the variables of that net would be more stable
        values_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)
        Q_val = TargetNet.predict(states)
        #Q learing formula
        Q_val = (1-self.alpha)*Q_val + self.alpha*actual_values

        #train the main model
        self.train_model.fit(np.expand_dims(s, 0), Q, epochs=1, verbose=0)
        #decrease the exploration rate after every iteration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def add_experience(self, experience):
        """
            add experience to the expReplayBuffer
        """
        if len(self.expReplayBuffer['s']) >= self.max_experiences:
            for key in self.expReplayBuffer.keys():
                self.expReplayBuffer[key].pop(0) #remove an old experience to make place for a new one FIFO
        for key, value in experience.items():
            self.expReplayBuffer[key].append(value) #add the new experience

    def copy_weights(self, TargetNet):
        """
            after a specified number of training steps, copy the trained value in local net to target net for future ref
        """
        local_net_weights = self.train_model.get_weights()
        target_net_weights = TargetNet.get_weights()
        #manually set the trainable variables of TrainNet to TargetNet
        for v1, v2 in zip(local_net_weights, target_net_weights):
            v1.assign(v2.numpy())
