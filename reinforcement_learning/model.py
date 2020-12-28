from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
import numpy as np

class DQNModel():

    def __init__(self, input_shape, hidden_units, num_actions, portfolio_size):
        inputs = Input(shape=input_shape)
        z = Flatten()(inputs)
        for i in hidden_units:
            z = Dense(
                units = i,
                activation='tanh',
                kernel_initializer='RandomNormal',
                bias_initializer='zeros'
            )(z)
            # z = Dropout(0.5)(z)
        predictions = []
        for i in range(portfolio_size):
            asset_dense = Dense(
                units = num_actions, #for each portfolio asset, 3 actions are possible, predictions wrt all 3 must be provided by model
                activation='linear',
                kernel_initializer='RandomNormal',
                bias_initializer='zeros'
            )(z)
            predictions.append(asset_dense)

        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer='adam', loss='mse')

    def get_model(self):
        return self.model

    def copy_weights(self, TargetNet):
        """
            after a specified number of training steps, copy the trained value in local net to target net for future ref
        """
        local_net_weights = self.model.get_weights()
        target_net_weights = TargetNet.get_weights()
        # print("Local Net Weights: " + str(local_net_weights))
        # print("Target Net Weights: " + str(target))
        #manually set the trainable variables of TrainNet to TargetNet
        for v1, v2 in zip(local_net_weights, target_net_weights):
            # v1.assign(v2.numpy())
            v1 = np.array(v2)
