from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from config import sensorLibrary
import networkx as nx


class SensorMetadata:
    def __init__(self, groupby="zone") -> None:
        self.sensorLabels = [i for i in sensorLibrary.keys()]
        self.costVector = np.array([i["cost"] for i in sensorLibrary.values()]).reshape(1,-1)[0]
        self.powerVector = np.array([i["power"] for i in sensorLibrary.values()]).reshape(1,-1)[0]
        self.groupby = groupby
        # self.costPurchase  = np.array([0,5,0.5,0.5,0.5,0.5,0.5])
        # self.powerConsumption = np.array([200,100,200,30,30,30])

        pass


    def evaluateBusiness(self, chromosome, taskType="opCost"):
        metric = []
        # if taskType=="opCost":
        # import pdb; pdb.set_trace()
        if taskType == "network":
            metric.append(self.evaluateNonRandomness(chromosome))

        # stride = 0
        # stride = int(len(chromosome)/len(self.sensorLabels)) if self.groupby == "domain" else len(self.sensorLabels)

        # for start_index in range(0, len(chromosome), stride):
        #     end_index=start_index+stride
            
        #     try:
                
            #     if taskType=="installCost":
            #         opCost = -1000000

            #         # import pdb; pdb.set_trace()
            #         if self.groupby == "zone":
            #             opCost = np.dot(self.costVector,np.array(chromosome[start_index:end_index]))[0]
            #         elif self.groupby == "domain":
            #             opCost = sum(chromosome[start_index:end_index])*self.costVector[int(start_index/stride)] 
            #         # print ("OC===>     ", opCost, sum(chromosome[start_index:end_index]))
            #         metric.append(opCost)
            #     elif taskType == "power":
            #         power = 1000000
            #         if self.groupby == "zone":
            #             power = np.dot(self.powerVector,np.array(chromosome[start_index:end_index]))[0]
            #         elif self.groupby == "domain":
            #             power = sum(chromosome[start_index:end_index])*self.powerVector[int(start_index/stride)]
            #         # print ("PW===>     ", power, sum(chromosome[start_index:end_index]))
            #         metric.append(power)
            # except:
            #     import pdb; pdb.set_trace()
            #     pass
        # import pdb; pdb.set_trace()
        metric = sum(metric)
        # print (f"Business Eval  {taskType} {sum(chromosome)} {(metric)} ")
        return metric

    def generateTopology(self,chromosome):
        G = nx.Graph()
        colourMap =[]
        anchor = None
        stride = int(len(chromosome)/len(self.sensorLabels)) if self.groupby == "domain" else len(self.sensorLabels)
        for start_index in range(0, len(chromosome), stride):
            xg = [start_index + i for i,v in enumerate(chromosome[start_index:start_index+stride]) if v==True]
            yg = [start_index + i for i,v in enumerate(chromosome[start_index:start_index+stride]) if v==False]
            assert(len(xg)+len(yg)==stride)
            # import pdb;pdb.set_trace()
            for u in xg:
                for v in yg:
                    G.add_edge(u,v)
            if anchor == None: anchor = xg[0]
            else:
                G.add_edge(anchor ,xg[0])
                anchor=xg[0]
        # import pdb;pdb.set_trace()
        # plt.figure(figsize =(15, 15))  
        # nx.draw_networkx(G, with_labels = True)
        # plt.show()
        # plt.clf()
        return G
        

    def evaluateNonRandomness(self, chromosome):
        G = self.generateTopology(chromosome)
        # metric = nx.s_metric(G)
        # metric = nx.average_clustering(G)
        # metric =nx.wiener_index(G)
        # nr, nr_rd = nx.non_randomness(G,6)
        # metric = -nr_rd
        # metric = nr_nd
        metric = G.number_of_edges() /(69*137)
        return metric

    def plotDataCapacity(self):

 
        # creating an empty canvas
        fig = plt.figure()
        
        # defining the axes with the projection
        # as 3D so as to plot 3D graphs
        ax = plt.axes(projection="3d")
        
        # creating a wide range of points x,y,z
        # x=[0,1,2,3,4,5,6]  # zones 
        # y=[i for i in range(0, 10)] # sensors
        # z1=[ for z in ]
        x= []
        y =[]
        zp = []
        za=[]
        for zone in range(0, 25): # zones 
            for n in range(0,10): # sensors /zone
                x.append(zone)
                y.append(n)
                zp.append(zone*n*(n-1)*0.5)
                za.append((zone*n)*(zone*n-1)*0.5)
        
        # plotting a 3D line graph with X-coordinate,
        # Y-coordinate and Z-coordinate respectively
        ax.scatter3D(x, y, za, c=za, cmap='cividis' )
        ax.scatter3D(x, y, zp, c=zp, cmap='cividis')
        # plotting a scatter plot with X-coordinate,
        # Y-coordinate and Z-coordinate respectively
        # and defining the points color as cividis
        # and defining c as z which basically is a
        # defination of 2D array in which rows are RGB
        #or RGBA
        # ax.scatter3D(x, y, za, c=zp, cmap='cividis')
        
        # Showing the above plot
        plt.show()

if __name__=="__main__":
    SensorMetadataObject=SensorMetadata()
    SensorMetadataObject.plotDataCapacity()