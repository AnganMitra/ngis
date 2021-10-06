from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
class SensorMetadata:
    def __init__(self) -> None:
        self.sensorLabels = ["ACPower","lightPower","appPower","temperature","humidity","lux"]
        self.costPurchase  = np.array([0,5,0.5,0.5,0.5,0.5,0.5])
        self.powerConsumption = np.array([200,100,200,30,30,30])
        pass

    def loadDetails(self, dataPath):

        pass

    def evaluateBusiness(self, chromosome, taskType="opCost"):
        metric = 0
        for start_index in range(0, len(chromosome), len(self.sensorLabels)):
            end_index=start_index+len(self.sensorLabels)
            if taskType=="opCost":
                metric += self.powerConsumption*np.array(chromosome[start_index:end_index])
            elif taskType == "installCost":
                metric += self.powerConsumption*np.array(chromosome[start_index:end_index])
        
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