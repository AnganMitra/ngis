import numpy as np
from learner import Learner
class MemoryTable:
    def __init__(self,key,numSensor,sensorLabels, data) -> None:
        self.key = key
        self.numSensor = numSensor
        self.table = np.zeros((numSensor,numSensor))
        self.sensorLabels = sensorLabels
        self.Learner = Learner()
        self.data = data
        pass
    
    def populateBySingleTask(self, task):
        for support in range(self.numSensor-1):
            for approximated in range(support+1, self.numSensor):
                
                self.Learner.initLearning(support, approximated)
                self.table[support, approximated] = self.Learner.getLoss(support)

    def evaluate(self,approximatedSet, supportSet):
        loss = []
        for i in approximatedSet:
            for j in supportSet:
                loss.append(self.table[i,j])

        return np.mean(loss)
        pass