import pandas as pd
import numpy as np
zones = [ 'Floor4Z5', 'Floor4Z4', 'Floor2Z2', 'Floor6Z4',
       'Floor6Z5', 'Floor2Z1', 'Floor6Z1', 'Floor4Z2', 'Floor2Z4', 'Floor6Z2',
       'Floor4Z1', 'Floor7Z4', 'Floor3Z1', 'Floor7Z5', 'Floor5Z5', 'Floor3Z2',
       'Floor5Z4', 'Floor7Z2', 'Floor5Z1', 'Floor7Z1', 'Floor3Z5', 'Floor3Z4',
       'Floor5Z2']

sensorTypes = ["lux", "temperature", "humidity", "ACPower", "appPower", "lightPower"]

filePath="./paperAnalysis/"
expGroup="234567f"
groupType="zone"
fp = f"{filePath}{expGroup}-{groupType}/" 

resDict = {}
for zone in zones:
    resDict[zone]={}
    df = pd.read_csv(fp+zone+".csv")
    df[df<0]=0
    resDict[zone].update ( { i : np.round(df[i].mean(),2)for i in df.columns}) 

resdf = pd.DataFrame.from_dict(resDict).T
resdf.to_csv(fp+"vsfAccuracy.txt")