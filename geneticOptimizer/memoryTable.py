import pdb
import numpy as np
from learner import Learner
import pandas as pd
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
        # pdb.set_trace()
        for support in range(self.numSensor):
            for approximated in range(self.numSensor):
                score=0
                if support==approximated:
                    score = -1000
                else:
                    score = self.Learner.initOneMapperLearning(self.sensorLabels[support], self.sensorLabels[approximated])
                
                self.table[support, approximated] = score
                # self.table[approximated,support] = score
                
                # self.table[support, approximated] = self.Learner.test(self, support, yTarget, X_test, y_true)
        # print (self.table)
        self.table = pd.DataFrame(self.table)
        self.table.columns= self.sensorLabels
        self.table.to_csv("./PaperAnalysis/"+self.key+".csv")
    

    def evaluate(self,approximatedSet, supportSet):
        loss = []
        # pdb.set_trace()
        for i in approximatedSet:
            for j in supportSet:
                loss.append(self.table[i,j])

        return np.sum(loss)
        pass