import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# fig, ax1 = plt.subplots()

def plotDoubleAxisFig(t,data1,data2, xlabel ="", ylabel = "", y2label = ""):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color)
    ax1.plot(t, data1, color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y2label, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color, marker="+")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()



filePath = "./paperAnalysis/"
expMode = "234567f"
folderOptions = [f"{expMode}-domain",f"{expMode}-zone",]
chromosomes={}
tradeOff = {}
for expOutput in folderOptions:
    chromosomes[expOutput]=pd.read_csv(filePath+expOutput+"/chromosomes.csv").T.iloc[1:,:]
    tradeOff[expOutput]=pd.read_csv(filePath+expOutput+"/tradeoff.csv")

filePath +="paperFigures/"
# t = np.arange(0.01, 10.0, 0.01)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)  "accuracy":accuracy, "powerConsumed":opCost, "installCost":installCost, "noOfSens"
# 
expOutput =folderOptions[0]
tradeOff[expOutput]=tradeOff[expOutput].sort_values(by='noOfSens')
plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
        data1 = tradeOff[expOutput]["forward"] ,
        data2=tradeOff[expOutput]["power"],
        xlabel="Number of Sensors",
        ylabel="Forward (MSE)",
        y2label="powerConsumed"
        )
plt.savefig(filePath+"tradeoffForwardPower.png", dpi=200)
plt.clf()
plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
        data1 = tradeOff[expOutput]["forward"] ,
        data2=tradeOff[expOutput]["backward"],
        xlabel="Number of Sensors",
        ylabel="Forward (MSE) ",
        y2label="Backward (MSE) "
        )
plt.savefig(filePath+"tradeoffForwardBackward.png", dpi=200)
# plt.show()
plt.clf()

plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
        data1 = tradeOff[expOutput]["forward"] ,
        data2=tradeOff[expOutput]["installCost"],
        xlabel="Number of Sensors",
        ylabel="Forward (MSE) ",
        y2label="installCost ($) "
        )
# plt.show()
plt.savefig(filePath+"tradeoffForwardCost.png", dpi=200)
plt.clf()

# chromosomes[expOutput].plot()
chromosomes[expOutput].sum(axis=0).plot()  
plt.savefig(filePath+"noSensInPareto.png", dpi=200)
plt.clf()

# plt.show()
chromosomes[expOutput].sum(axis=1).plot()  
plt.savefig(filePath+"zoneRequirement.png", dpi=200)
# plt.show()
plt.clf()