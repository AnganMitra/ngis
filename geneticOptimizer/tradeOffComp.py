import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from addCluster import fbcluster
# fig, ax1 = plt.subplots()

def plotDoubleAxisFig(t,data1,data2, xlabel ="", ylabel = "", y2label = ""):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color)
    ax1.scatter(t, data1, color=color, marker="o", label="Forward MSE")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y2label, color=color)  # we already handled the x-label with ax1
    ax2.scatter(t, data2, color=color, marker="+", label="Backward MSE")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()



filePath = "./paperAnalysis/"
expMode = "234567f"
folderOptions = [f"{expMode}-domain",]  # f"{expMode}-domain-run0-identicalConf"
chromosomes={}
tradeOff = {}
for expOutput in folderOptions:
    chromosomes[expOutput]=pd.read_csv(filePath+expOutput+"/chromosomes.csv").T.iloc[1:,:]
    tradeOff[expOutput]=pd.read_csv(filePath+expOutput+"/tradeoff.csv")

filePath +="paperFigures/"


expOutput =folderOptions[0]
tradeOff[expOutput]=tradeOff[expOutput].sort_values(by='noOfSens')
try:
    # import pdb; pdb.set_trace()
    t = tradeOff[expOutput]["noOfSens"]
    plt.scatter(t, tradeOff[expOutput]["forward"], marker="+", label="")
    
    # tradeOff[expOutput]["forward"].plot()
    plt.tight_layout()  
    plt.xlabel="Number of Sensors"
    plt.ylabel="Forward (MSE) "
    plt.savefig(filePath+"tradeoffForwardSensor.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("tradeoffForwardSensor    .....")
except:
    print ("tradeoffForwardSensor    ..XXXXXXXX...")

try:
    

    plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = tradeOff[expOutput]["forward"] ,
            data2=tradeOff[expOutput]["backward"],
            xlabel="Number of Sensors",
            ylabel="Forward (MSE)",
            y2label="Backward (MSE)"
            )
    plt.ylabel="Backward (MSE)"
    plt.xlabel="Forward (MSE)"

    # plt.scatter(tradeOff[expOutput]["forward"], tradeOff[expOutput]["backward"], label=" NGIS", marker="+")
    # fwloss, bkloss = fbcluster(30, "kmeans")
    # plt.scatter(fwloss,bkloss, label=" Kmeans", marker="+")
    # fwloss, bkloss = fbcluster(30, "dbscan")
    # plt.scatter(fwloss,bkloss , label=" DBSCAN", marker="+")
    
    # # plt.scatter(t, bcluster(len(t), algo="dbscan"), label="backward DBSCAN", marker="+")
    # plt.ylabel="Backward (MSE)"
    # plt.xlabel="Forward (MSE) "
    # plt.gca().update(dict(title='Approximation Error', xlabel='Forward MSE', ylabel='Backward MSE', ))
    # plt.legend()
    plt.tight_layout()
    plt.savefig(filePath+"tradeoffForwardBackward.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("tradeoffForwardBackward    .....")
except:
    print ("tradeoffForwardBackward    ..XXXXXXXX...")

try:
    plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = tradeOff[expOutput]["forward"] ,
            data2=tradeOff[expOutput]["power"],
            xlabel="Number of Sensors",
            ylabel="Forward (MSE)",
            y2label="powerConsumed"
            )
    plt.savefig(filePath+"tradeoffForwardPower.png", dpi=200)
    plt.clf()
    print ("tradeoffForwardPower    .....")
except:
    print ("tradeoffForwardPower    ..XXXXXXXX...")


try:
    plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = tradeOff[expOutput]["installCost"] ,
            data2=tradeOff[expOutput]["power"],
            xlabel="Number of Sensors",
            ylabel="Purchase Cost ($)",
            y2label="powerConsumed"
            )
    plt.savefig(filePath+"tradeoffCostPower.png", dpi=200)
    plt.clf()
    print ("tradeoffCostPower    .....")
except:
    print ("tradeoffCostPower    ..XXXXXXXX...")

try:
    plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = tradeOff[expOutput]["forward"] ,
            data2=tradeOff[expOutput]["installCost"],
            xlabel="Number of Sensors",
            ylabel="Forward (MSE) ",
            y2label="installCost ($) "
            )
    # plt.show()
    plt.savefig(filePath+"tradeoffForwardCost.png", dpi=200)
    print ("tradeoffForwardCost    .....")
    plt.clf()
except:
    print ("tradeoffForwardCost    ..XXXXXXXX...")

# chromosomes[expOutput].plot()
chromosomes[expOutput].sum(axis=0).plot()  
plt.savefig(filePath+"noSensInPareto.png", dpi=200)
plt.clf()

# plt.show()
chromosomes[expOutput].sum(axis=1).plot()  
plt.savefig(filePath+"zoneRequirement.png", dpi=200)
# plt.show()
plt.clf()