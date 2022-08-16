import numpy as np
from random import randrange


np.random.seed(0)
class GroupSampler():
    def __init__(self, prob_dist):
        self.prob_dist = prob_dist
    
    def sample_one(self):
        return np.random.choice(len(self.prob_dist), 1, p=self.prob_dist)[0]
    
    def get_group_indices(self):
        return np.arange(len(self.prob_dist))
