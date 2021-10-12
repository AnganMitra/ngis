import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# fig, ax1 = plt.subplots()

def plotDoubleAxisFig(t,data1,data2, xlabel ="", ylabel = "", y2label = ""):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y2label, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()



filePath = "./paperAnalysis/"
expMode = "234567f"
folderOptions = [f"{expMode}-domain",f"{expMode}-zone",]
chromosomes={}
tradeOff = {}
for expOutput in folderOptions:
    chromosomes[expOutput]=pd.read_csv(filePath+expOutput+"/chromosomes.csv")
    tradeOff[expOutput]=pd.read_csv(filePath+expOutput+"/tradeoff.csv")


# t = np.arange(0.01, 10.0, 0.01)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)  "accuracy":accuracy, "powerConsumed":opCost, "installCost":installCost, "noOfSens"
# 
expOutput =folderOptions[1]

plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
        data1 = tradeOff[expOutput]["accuracy"] ,
        data2=tradeOff[expOutput]["powerConsumed"],
        xlabel="Number of Sensors",
        ylabel="accuracy",
        y2label="powerConsumed"
        )
plt.show()
plt.clf()
plotDoubleAxisFig(t=tradeOff[expOutput]["noOfSens"],
        data1 = tradeOff[expOutput]["installCost"] ,
        data2=tradeOff[expOutput]["powerConsumed"],
        xlabel="Number of Sensors",
        ylabel="installCost",
        y2label="powerConsumed"
        )
plt.show()
