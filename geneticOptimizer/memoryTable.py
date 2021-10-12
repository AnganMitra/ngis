import pdb
import numpy as np
from learner import Learner
import pandas as pd
class MemoryTable:
    def __init__(self,key,numSensor,sensorLabels, data, output_dir) -> None:
        self.key = key
        self.output_dir=output_dir
        self.numSensor = numSensor
        self.table = np.zeros((numSensor,numSensor))
        self.sensorLabels = sensorLabels
        self.Learner = Learner(data)
        self.data = data
        pass
    
    def populateBySingleTask(self):
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
        self.table.to_csv(self.output_dir+self.key+".csv")
    

    def evaluate(self,approximatedSet, supportSet):
        loss = []
        # pdb.set_trace()  # net loss 
        for i in approximatedSet:
            lossFromApproximation= []
            if len(supportSet)==0:break
            for j in supportSet:
                lossFromApproximation.append(self.table.iloc[i,j])
            # print (lossFromApproximation)
            loss.append(min(lossFromApproximation))
        try:
            loss = np.sum(loss)
        except:
            loss = 10000
        finally:
            return loss
        # pass