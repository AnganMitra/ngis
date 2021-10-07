import os
from typing import final
import pandas as pd
from functools import reduce
from datetime import timezone

# def getDataDictionary(dataPath = "./BKDataCleaned/"): # keys are 
def getSpatialGroupData(dataPath = "./BKDataCleaned/", start_index=0, end_index =100, floors = [4,5,6,7]): # keys are     
    dataDictionary={}
    for file in os.listdir(dataPath):
        if file.endswith(".csv") and file.startswith("Floor"):
            if int(file[5]) not in floors: continue  ## lexical parsing FloorX
            zoneSpecifier = file.strip().split(".")[0]
            try:
                dataDictionary[zoneSpecifier] = pd.read_csv(dataPath+file, parse_dates=True, index_col=["Date"]).iloc[start_index: end_index]
            except:
                import pdb;pdb.set_trace()
            finally:
                print (f"Data read from {zoneSpecifier} with {len(dataDictionary[zoneSpecifier])} samples...")
    
    encodingLabels = {i:v.columns for i,v in dataDictionary.items()}
    print (encodingLabels)
    return dataDictionary, encodingLabels


def getDomainGroupData(dataPath = "./BKDataCleaned/", start_index=0, end_index =100, floors = [4,5,6,7]): # keys are     
    dataDictionary,_ = getSpatialGroupData(dataPath,start_index,end_index, floors )
    sensorGroups =[]
    # import pdb; pdb.set_trace()
    for zone,v in dataDictionary.items():
        sensorGroups += [j for j in v.columns] #for i,v in dataDictionary.items()] )

    sensorGroups=list(set(sensorGroups))
    sensorDictionary = {}
    for sens in sensorGroups:
        sensorDictionary[sens] = []
        for zone,v in dataDictionary.items(): # iterate zone wise
            sensorDictionary[sens].append(v[sens].rename(zone))
            
    for sens in sensorGroups:   
        horizontaldf = pd.DataFrame(sensorDictionary[sens][0]).resample("5T").max()
        # horizontaldf.index = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in horizontaldf.index]  
        
        for index,frame in enumerate(sensorDictionary[sens][1:]): 
            try:
                # if index > 8 : import pdb; pdb.set_trace()
                horizontaldf = horizontaldf.join(frame.resample("5T").max(), how="outer")
                # print (index, "frame===>", frame, "hdf==========>",horizontaldf)

            except:
                # import pdb; pdb.set_trace()
                print ("Error in grouping ")
            # finally:
        # import pdb;pdb.set_trace()
        horizontaldf = horizontaldf.dropna(axis=0)
        print (f"Grouped by {sens} with {len(horizontaldf)} samples ....")
        
        sensorDictionary[sens] = horizontaldf

    encodingLabels = {i:v.columns for i,v in sensorDictionary.items()}
    print (encodingLabels)
    
    return sensorDictionary, encodingLabels

def getRandomGroupData(dataPath = "./BKDataCleaned/", start_index=0, end_index =100, floors = [4,5,6,7]): # keys are     
    return getSpatialGroupData(dataPath,start_index,end_index, floors )


# getDomainGroupData(dataPath = "./BKDataCleaned/")