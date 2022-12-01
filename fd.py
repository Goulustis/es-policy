import numpy as np

from copy import deepcopy


class FiniteDiff:
    def __init__(self, model, learning_rate, noise_std):
        self.model = model 
        self.lr = learning_rate
        self.noise_std = noise_std

        self._population = [self.model]
    

    def generate_population(self):
        self._population = [self.model]

        for i, layer in enumerate(self.model.W):

            for j in range(len(layer)):
                new_model = deepcopy(self.model)
                e = np.zeros(layer.shape)
                e[j] = 1
                noise = e*self.noise_std

                # since fitness fnc is in [-1, 1]
                new_model.W[i] = np.clip(new_model.W[i] + noise, -1, 1)

                self._population.append(new_model)
        
        return self._population
    
    def calculate_grad(self, rewards):
        ori_rew = rewards[0]

        # always only one layer
        for i, layer in enumerate(self.model.W):
            # calculate gradient
            grad = (rewards[1:] - ori_rew)/self.noise_std
        
        return grad



    def update_model(self, rewards):
        ori_rew = rewards[0]

        # always only one layer
        for i, layer in enumerate(self.model.W):
            # calculate gradient
            grad = (rewards[1:] - ori_rew)/self.noise_std

            # update
            new_weights = layer + self.lr * grad 
            self.model.set_weights(new_weights)