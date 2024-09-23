import numpy as np

class BagLearner:
    def __init__(self, learner, kwargs={"argument1": 1, "argument2": 2}, bags=20, boost=False, verbose=False):
        self.bags = bags
        self.learner = learner
        self.kwargs = kwargs
        self.learners = [self.learner(**self.kwargs) for i in range(self.bags)]
        
    def add_evidence(self, data_x, data_y):
        for j in self.learners:
            bag = np.random.choice(range(data_x.shape[0]), size=data_x.shape[0], replace=True)
            bootx = data_x[bag]
            booty = data_y[bag]
            j.add_evidence(bootx, booty)
    
    def query(self, points):
        return np.mean(np.array([learner.query(points) for learner in self.learners]),axis=0)
    
    def author(self):
        return "efigone3"