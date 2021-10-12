import pandas as pd
import numpy as np
sensorTypes = ["lux", "temperature", "humidity", "ACPower", "appPower", "lightPower"]

filePath="./paperAnalysis/"
expGroup="234567f"
groupType="domain"
fp = f"{filePath}{expGroup}-{groupType}/" 

resDict = {}
for sensor in sensorTypes:
    resDict[sensor]={}
    df = pd.read_csv(fp+sensor+".csv")
    df[df<0]=0
    resDict[sensor].update ( { i : np.round(df[i].mean(),2)for i in df.columns}) 

resdf = pd.DataFrame.from_dict(resDict)
resdf.to_csv(fp+"vsfAccuracy.txt")