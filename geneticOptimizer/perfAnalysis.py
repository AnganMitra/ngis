columns =[ "ACPower", "lightPower", "appPower", "temperature", "humidity", "lux" ]
max_markers = ["8", "s", "p", "P", "h", "D"]
target = "Floor"
approxDict = {"ACPower" : {},
             "lightPower" : {},
             "appPower": {},
             "temperature": {},
             "humidity": {},
             "lux": {},}
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataLen=0
for fp in os.listdir("./perfDump/"):
    print (fp)
    if fp.startswith(target):
    # if True:
        if "-" in fp: continue
        f=open(f"./perfDump/{fp}", "r").read().strip("[").strip("]").strip().split("][")
        # support = fp.split(".")[0].split("|")[-1]
        key=fp.split(".")[0]
        support = key
        for index,cutterPerf in enumerate(f):
            try:
                approxDict[columns[index]][support] = [float(r) for r in cutterPerf.split(",")] 
            except:
                pass
                # import pdb; pdb.set_trace()
            
for sensor, payload in approxDict.items():
    # import pdb; pdb.set_trace()
    perfDict= pd.DataFrame.from_dict(payload).iloc[:,:] 
    for floorNo in range(2,8): 
        perfDict["max"] = perfDict[[i for i in perfDict.columns if i.startswith(f"Floor{floorNo}")]].max(axis=1)
        # perfDict["std"] = perfDict[[i for i in perfDict.columns if i.startswith(f"Floor{floorNo}")]].std(axis=1)
        maxPerf= (perfDict["max"])
        x = [i for i in range(len(maxPerf))]
        # maxPerf = [value/(len(maxPerf)-1*index +1) for index, value in enumerate(maxPerf)]
        maxPerf = [value/(1*index +1) for index, value in enumerate(maxPerf)]
        # plt.plot(x, maxPerf, label= f"Floor {floorNo} " )  

        plt.semilogy(x, maxPerf, marker= max_markers[floorNo-2], label= f"Floor {floorNo} " )  
        # plt.fill_between(x, maxPerf - perfDict["std"] , maxPerf + perfDict["std"] , alpha=0.2)
        # perfDict["min"] = perfDict[[i for i in perfDict.columns if i.startswith(f"Floor{floorNo}")]].min(axis=1)
        # plt.plot([i for i in range(dataLen)], perfDict["min"][25:125], marker=floorNo, label= f"Floor {floorNo} - min " )  
   

    plt.legend( loc="upper right",)
    plt.title(f"{sensor}")
    plt.xlabel("days")
    plt.ylabel("Reconstruction Loss")
    # plt.show()
    plt.savefig( f"{sensor}LL.png", dpi=300)
    plt.clf()


# for key, value in approxDict.items():
#     for support,cutterPerf in value.items():    
#         plt.plot([i for i in range((dataLen-10))], cutterPerf[:-10],  label= support )
#     plt.legend(bbox_to_anchor=(1,1), loc="upper left",)
#     plt.title(key+ " - " + target)
#     plt.xlabel("days")
#     plt.ylabel("RMSE")
#     plt.show()
        

        

