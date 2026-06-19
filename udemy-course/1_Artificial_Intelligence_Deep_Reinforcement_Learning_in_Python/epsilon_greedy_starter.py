import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITY = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p: true win rate
        self.p = p
        self.p_estimate = None
        self.N = NUM_TRIALS

    def pull(self):
        # draw a 1 with a probablity p
        return np.random.random() < self.p
    
    def update(self, x):
        self.N = None
        self.p_estimate = None
        return
    