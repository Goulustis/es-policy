import numpy as np
from fit_fncs import FITNESS_FNS

class Model:
    def __init__(self, pos = None) -> None:
        if pos is None:
            self.W = [np.array([[0,0]]).T]
        else:
            self.W = [np.array(pos).reshape(2, 1)]
    
    def predict(self, x = None):
        return self.W[0]
    
    def set_weights(self, weight):
        self.W = [weight]
    
    def get_weights(self):
        return self.W[0]
    
    def get_pos(self):
        return self.get_weights()

def make_policy(config):
    return Model(FITNESS_FNS[config["fnc_type"]]["start_pos"])