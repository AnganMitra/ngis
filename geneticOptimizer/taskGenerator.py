class TaskGenerator:
    def __init__(self, dataPath,) -> None:
        
        self.dataPath = dataPath

    def loadData(self):

        pass

    def loadTask(self):

        pass

    def generateTasks(self):
        
        pass

    def evaluateForecastingLoss(self, chr): ## Forecasting error difference
        pass
    def evaluateOperationalCost(self, chr):  ## Cost of sensors to be installed
        pass
    def evaluateOperationalPower(self, chr):  ## power needed to run the solution 
        pass
    def evaluateControlLoss(self, chr): # control power variation
        pass

