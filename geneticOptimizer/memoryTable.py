import numpy as np
from learner import Learner
class MemoryTable:
    def __init__(self,key,numSensor,sensorLabels, data) -> None:
        self.key = key
        self.numSensor = numSensor
        self.table = np.zeros((numSensor,numSensor))
        self.sensorLabels = sensorLabels
        self.Learner = Learner(data)
        self.data = data
        pass
    
    def populateBySingleTask(self, task):
        for support in range(self.numSensor-1):
            for approximated in range(support+1, self.numSensor):
                # import pdb; pdb.set_trace()
                score = self.Learner.initOneMapperLearning(self.sensorLabels[support], self.sensorLabels[approximated])
                
                self.table[support, approximated] = score
                self.table[approximated,support] = score
                
                # self.table[support, approximated] = self.Learner.test(self, support, yTarget, X_test, y_true)
        print (self.table)
    def evaluate(self,approximatedSet, supportSet):
        loss = []
        for i in approximatedSet:
            for j in supportSet:
                loss.append(self.table[i,j])

        return np.sum(loss)
        pass