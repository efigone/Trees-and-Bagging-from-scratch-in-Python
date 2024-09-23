import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner:
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20) for i in range(20)]
        
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    
    def query(self, points):
        return np.mean(np.array([learner.query(points) for learner in self.learners]), axis=0)
    
    def author(self):
        return "efigone3"