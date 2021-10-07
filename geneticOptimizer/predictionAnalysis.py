import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import plotly.express as px



rdf = {}
filePath = "./paperAnalysis/"
for file in os.listdir(filePath):
    if file.endswith(".csv"):
        key = file.split(".")[0]
        rdf[key]= pd.read_csv(filePath+file, )
        rdf[key][rdf[key]<0] = 0
        rdf[key].index = rdf[key].columns[1:]
        rdf[key]=rdf[key].drop('Unnamed: 0', axis = 1)
        # import pdb; pdb.set_trace()

def plotPredPerfbyKey(key):
    try:
        # rdf[key].plot(marker=11)
        fig = px.imshow(rdf[key])
        fig.write_image(f"paperAnalysis/{key}.pdf")
        # fig.show()
        # plt.show()
    except:
        import pdb; pdb.set_trace()
        pass
    finally:
        plt.clf()

for key in rdf.keys(): plotPredPerfbyKey(key)