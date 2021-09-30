from geneticOptimizer.memoryTable import MemoryTable
from dataLoader import getDataDictionary
import numpy as np
class TaskGenerator:
    def __init__(self, dataPath) -> None:
        
        self.dataPath = dataPath
        self.dataDictionary = None
        self.loadData()
        self.zones = self.dataDictionary.keys()
        self.megaMemory = {}
        self.sensorLabels = ["ACPower","lightPower","appPower","temperature","humidity","lux"]

    def initMemoryTable(self):
        for zone in self.zones:
            self.megaMemory[zone] = MemoryTable(key=zone, numSensor= len(self.sensorLabels), sensorLabels= self.sensorLabels, data=self.dataDictionary[zone])

    def loadData(self):
        self.dataDictionary = getDataDictionary(self.dataPath)
        pass

    def loadTask(self):
        
        pass

    def generateZonalTasks(self, chr, threshold=0.2):
        start_index = 0
        taskList = {}
        for key,value in self.dataDictionary.items():
            end_index= start_index+len(value.columns)
            approximatedSet = [i for i,v in enumerate(chr[start_index:end_index]) if v <threshold]
            supportSet = [i for i,v in enumerate(chr[start_index:end_index]) if v > threshold]
            taskList[key]={
                "approximatedSet" : approximatedSet,
                "supportSet" : supportSet
            }
            start_index = end_index
        return taskList

    def evaluateAILoss(self, chr, taskType): ## Forecasting error difference
        loss = []
        taskList = self.generateZonalTasks(chr)
        for zone, task in taskList.items():
            lossPerZone = self.megaMemory[f"{taskType}Loss"][zone].evaluate(approximatedSet=task["approximatedSet"], supportSet=task["supportSet"])
            loss.append(lossPerZone)
        return np.mean(loss)
        pass
    
    def evaluateBusiness(self, chr, taskType): ## Forecasting error difference
        cost = []
        taskList = self.generateZonalTasks(chr)
        for zone, task in taskList.items():
            costPerZone = self.megaMemory[f"{taskType}Loss"][zone].evaluate(approximatedSet=task["approximatedSet"], supportSet=task["supportSet"])
            cost.append(costPerZone)
        return np.mean(cost)
        pass
    