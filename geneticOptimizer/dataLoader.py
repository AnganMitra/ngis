import os
import pandas as pd

def getDataDictionary(dataPath = "./BKDataCleaned/"):
    
    dataDictionary={}
    for file in os.listdir(dataPath):
        if file.endswith(".csv"):
            zoneSpecifier = file.strip().split(".")[0]
            dataDictionary[zoneSpecifier] = pd.read_csv(dataPath+file, parse_dates=True, index_col=["Date"])
    return dataDictionary