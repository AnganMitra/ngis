'''
1. start with C Pareto-optimal configurations at time t
2. Track hardness metric till t+ T
3. Including performance metric to objectives
4. relearn robust configuration
'''
import pandas as pd
from geneticOptimizer.solver import SolverBuildings
tmin = 0
delT = 1000
tmax= 100000
sensorGroupSize = 24 # for intra-domain learning

def deployPerf(vsfConf, t1,t2):
    for groupNo, sensorGroup in enumerate(0, len(vsfConf), sensorGroupSize):
        chromosomeEncoding = vsfConf[sensorGroup : sensorGroup+sensorGroupSize]
        supportGroupIndices = [groupNo*sensorGroupSize+index for index,encoding in chromosomeEncoding if encoding == 1]
        approximateGroupIndices = [groupNo*sensorGroupSize+index for index,encoding in chromosomeEncoding if encoding == 0]
        estimatePerf(supportGroupIndices, approximateGroupIndices, t1,t2)
    pass

def estimatePerf(supportGroupIndices, approximateGroupIndices, t1,t2):
    for supportElement in supportGroupIndices:
        selfCreationLoss = min([ hypothesisLoss(missingElement,supportElement, t1,t2) for missingElement in approximateGroupIndices ])
        
    for approximatedElement in approximateGroupIndices:
        bestSupportElement = min([ hypothesisLoss(supportElement,approximatedElement, t1,t2) for supportElement in supportGroupIndices ])

def hypothesisPrediction(u,v,t1,t2):
    ''''
    engage u to predict v and return vector of predictions from t1 to t2 in as group g
    '''


def estimationFromVsf(vsfConf, t1,t2):
    for groupNo, sensorGroup in enumerate(0, len(vsfConf), sensorGroupSize):
        chromosomeEncoding = vsfConf[sensorGroup : sensorGroup+sensorGroupSize]
        for element in sensorGroup:
            if element == 0:

            elif element == 1:

                
            else:
                raise(Exception) 

    pass

def trueLoss(vsfConf, t1,t2):
    for groupNo, sensorGroup in enumerate(0, len(vsfConf), sensorGroupSize):
        chromosomeEncoding = vsfConf[sensorGroup : sensorGroup+sensorGroupSize]
        for element in sensorGroup:
            if element == 0:

            elif element == 1:

                
            else:
                raise(Exception)

    pass

def trackPerfOverTime(confList, t1,t2):
    runTimePerf ={
        "reconstructionLoss" : [],
        "estimationFromVsf" : [],
        "contributionToVsf" : [],
        "trueLoss" : [],
        }
    for vsfConf in confList:
        
        runTimePerf["reconstructionLoss"].append(reconstructionError(vsfConf, t1,t2))
        runTimePerf["estimationFromVsf"].append(estimationFromVsf(vsfConf, t1,t2))
        runTimePerf["contributionToVsf"].append(contributionToVsf(vsfConf, t1,t2))
        runTimePerf["trueLoss"].append(trueLoss(vsfConf, t1,t2))
        
    runTimePerf = pd.DataFrame.from_dict(runTimePerf)
    return runTimePerf
    pass

def getOptimalConfigurations(t1, t2, runTimePerf):
    ''''
    Call the solver once without runTimePerf => Pre evaluation stage
    Call the solver once with runTimePerf => Post deployment stage
    '''
    pass


for t in range(tmin, tmax, delT):
    optimalConf = getOptimalConfigurations(t, t+delT, None)
    runTimePerf = trackPerfOverTime(optimalConf, t, t+delT)
    optimalConfLL = getOptimalConfigurations(t, t+delT, runTimePerf)
    runTimePerfLL = trackPerfOverTime(optimalConfLL, t, t+delT)
    

    