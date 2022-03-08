from matplotlib import colors
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# fig, ax1 = plt.subplots()
filePath = "./paperAnalysis/"
expMode = "234567f"
# folderOptions = [f"{expMode}-domain-run{i}" for i in range(4)]  # f"{expMode}-domain-run0-identicalConf"
folderOptions=[
    "234567f-domain-run0" ,
    "234567f-domain-run1",
     "234567f-domain-run2", 
     "234567f-domain-run3"
     ]

labels = [
    "Cp/Camb = 2.5, Ep/Eamb=1",
    "Cp/Camb = 1, Ep/Eamb=1",
    "Cp/Camb = 1, Ep/Eamb=2",
    "Cp/Camb = 2, Ep/Eamb=2",
    
]
colorPlate = [
    'red',
    'green',
    'blue',
    'orange'
]
markerOptions =[
    "+",
    "o",
    "x",
    "^"

]

chromosomes={}
tradeOff = {}




def plotMultiColFig(t,data1, xlabel ="", ylabel = "", ):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color)
    for index, d in enumerate(data1):
        ax1.scatter(t, d, marker=markerOptions[index], label=labels[index], color=colorPlate[index])
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show() 



def plotMultiDoubleAxisFig(t,data1,data2, xlabel ="", ylabel = "", y2label = ""):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color)
    for index, d in enumerate(data1):
        ax1.scatter(t, d, marker=markerOptions[index], label=labels[index], color=colorPlate[index])
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y2label, color=color)  # we already handled the x-label with ax1
    for index, d in enumerate(data2):
        ax2.scatter(t, d, marker=markerOptions[index], label=labels[index])
    ax2.tick_params(axis='y', labelcolor=color)
    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show() 



for expOutput in folderOptions:
    chromosomes[expOutput]=pd.read_csv(filePath+expOutput+"/chromosomes.csv").T.iloc[1:,:]
    tradeOff[expOutput]=pd.read_csv(filePath+expOutput+"/tradeoff.csv").sort_values(by='noOfSens')

filePath +="paperFigures/"

try:
    plotMultiColFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["forward"] for expo in tradeOff.keys()] ,
            xlabel="Number of Sensors",
            ylabel="Forward (MSE) ",
            )
    plt.savefig(filePath+"forwardVar.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("forwardVar    .....")
except:
    print ("forwardVar    ..XXXXXXXX...")

try:
    plotMultiColFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["backward"] for expo in tradeOff.keys()] ,
            xlabel="Number of Sensors",
            ylabel="Backward (MSE) ",
            )
    plt.savefig(filePath+"backwardVar.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("backwardVar    .....")
except:
    print ("backwardVar    ..XXXXXXXX...")

try:
    plotMultiColFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["installCost"] for expo in tradeOff.keys()] ,
            xlabel="Number of Sensors",
            ylabel="Purchase Cost ($) ",
            )
    plt.savefig(filePath+"installCostVar.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("installCostVar    .....")
except:
    print ("installCostVar    ..XXXXXXXX...")

try:
    plotMultiColFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["power"] for expo in tradeOff.keys()] ,
            xlabel="Number of Sensors",
            ylabel="Power Consumption (kWh) ",
            )
    plt.savefig(filePath+"powerVar.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("powerVar    .....")
except:
    print ("powerVar    ..XXXXXXXX...")


# expOutput =folderOptions[0]
# tradeOff[expOutput]=tradeOff[expOutput].sort_values(by='noOfSens')
try:
    # import pdb; pdb.set_trace()
    tradeOff[expOutput]["forward"].plot()
    plt.xlabel="Number of Sensors"
    plt.ylabel="Forward (MSE) "
    plt.savefig(filePath+"tradeoffForwardSensor.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("tradeoffForwardSensor    .....")
except:
    print ("tradeoffForwardSensor    ..XXXXXXXX...")

try:
    plotMultiDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["forward"] for expo in tradeOff.keys()] ,
            data2=[tradeOff[expo]["backward"] for expo in tradeOff.keys()],
            xlabel="Number of Sensors",
            ylabel="Forward (MSE) ",
            y2label="Backward (MSE) "
            )
    plt.savefig(filePath+"tradeoffForwardBackward.png", dpi=200)
    # plt.show()
    plt.clf()
    print ("tradeoffForwardBackward    .....")
except:
    print ("tradeoffForwardBackward    ..XXXXXXXX...")

try:
    # import pdb; pdb.set_trace()
    plotMultiDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["forward"] for expo in tradeOff.keys()]  ,
            data2=[tradeOff[expo]["power"] for expo in tradeOff.keys()] ,
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
    plotMultiDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["installCost"] for expo in tradeOff.keys()]  ,
            data2=[tradeOff[expo]["power"] for expo in tradeOff.keys()] ,
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
    plotMultiDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
            data1 = [tradeOff[expo]["forward"] for expo in tradeOff.keys()]  ,
            data2=[tradeOff[expo]["installCost"] for expo in tradeOff.keys()] ,
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
for expOutput in folderOptions:chromosomes[expOutput].mean(axis="index").plot()  
plt.savefig(filePath+"noSensInPareto.png", dpi=200)
plt.clf()

# # plt.show()
# chromosomes[expOutput].sum(axis=1).plot()  
# plt.savefig(filePath+"zoneRequirement.png", dpi=200)
# # plt.show()
# plt.clf()