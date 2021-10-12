from copy import Error
import pdb
from sensorMetadata import SensorMetadata
from memoryTable import MemoryTable
from dataLoader import getDomainGroupData, getRandomGroupData, getSpatialGroupData
import numpy as np

metadata = SensorMetadata()
class TaskGenerator:
    def __init__(self, dataPath, start_index, end_index, floors, groupBy, output_path) -> None:
        
        self.dataPath = dataPath
        self.output_path=output_path
        self.dataDictionary = None
        self.sensorLabels =  None #["ACPower","lightPower","appPower","temperature","humidity","lux"]
        if groupBy =="zone":
            self.dataDictionary, self.sensorLabels = getSpatialGroupData(self.dataPath, start_index, end_index, floors,)
        elif groupBy =="domain":
            self.dataDictionary, self.sensorLabels = getDomainGroupData(self.dataPath, start_index, end_index, floors,)
        elif groupBy =="random":
            self.dataDictionary, self.sensorLabels = getRandomGroupData(self.dataPath, start_index, end_index, floors,)
        else:
            assert(Error)
        self.groups = [i for i in self.dataDictionary.keys()]
        self.megaMemory = {}
        
        self.taskTypes = ["prediction"]

    def initMemoryTable(self):

        for taskType in self.taskTypes:
            self.megaMemory[f"{taskType}Loss"] = {}
            for zone in self.groups:
                # print (zone)
                
                try:
                    self.megaMemory[f"{taskType}Loss"][zone] = MemoryTable(key=zone, numSensor= len(self.sensorLabels[zone]), sensorLabels= self.sensorLabels[zone], data=self.dataDictionary[zone], output_dir=self.output_path)
                    self.megaMemory[f"{taskType}Loss"][zone].populateBySingleTask()

                except Error as err:
                    # print (err)
                    pdb.set_trace()
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
    

    