from matplotlib import markers
import pandas as pd
from sensorMetadata import SensorMetadata
from taskGenerator import TaskGenerator
import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.util import plotting
from pymoo.algorithms.moo.nsga2 import NSGA2
import plotly.graph_objects as go

SmartBuilingObject = None
SensorMetadataObject = None 
class sensorOptimizingProblem(Problem):

    def __init__(self, n_var=2, n_obj=1, n_constr=1, xl=0, xu=10, type_var=int):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, type_var=type_var,)

    def computeConstraintViolation(self,chromosome): # per chromosome
        coverageMiss = 0
        if sum(chromosome)==0: coverageMiss=1000
        stride = len(SmartBuilingObject.taskGenerator.sensorLabels)
        # print(stride)
        for index in range(0, len(chromosome),stride ):
            coverageMiss += 1 if sum( chromosome[index: index+ stride]) == 0 else 0

        return coverageMiss

    def _evaluate(self, x, out, *args, **kwargs):
        # evaluate the fitness function for the generation 
        resFitness = []
        cvError =[]
        for chromosome in x:
            # evaluate chromosome fitness
            fitness = multiObjectiveScore(chromosome)
            error = self.computeConstraintViolation(chromosome)
            resFitness.append(fitness)
            cvError.append(error)
        # import pdb; pdb.set_trace()
        out["F"] = np.array(resFitness)
        # print (out["F"])
        # shape = out["F"].shape
        out["G"] =  np.array(cvError)

        return super()._evaluate(x, out, *args, **kwargs)

def returnMethod(optimizationTypeBool=True, pop_size=20):
    method = None
    if not optimizationTypeBool:
        method = get_algorithm("nsga2",
                            pop_size=pop_size,
                            sampling=get_sampling("int_random"),
                            crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                            mutation=get_mutation("int_pm", eta=3.0),
                            eliminate_duplicates=True,
                            ) 
    else:
        method = get_algorithm("nsga2",
                        pop_size=pop_size,
                        sampling=get_sampling("bin_random"),
                        crossover=get_crossover("bin_hux"),
                        mutation=get_mutation("bin_bitflip"),
                        eliminate_duplicates=True
                        )
    
    return method


# evaluate forecasting power of a chromosome

def multiObjectiveScore(chromosome):
    scoreArray = [SmartBuilingObject.taskGenerator.evaluateAILoss(chromosome, taskType="prediction") , ## Forecasting channel 
                # SmartBuilingObject.taskGenerator.evaluateAILoss(chromosome, taskType="hf"), ## power to ambience
                # SmartBuilingObject.taskGenerator.evaluateAILoss(chromosome, taskType="hb"), ## ambience to power
                SensorMetadataObject.evaluateBusiness(chromosome, taskType="installCost"), ## Cost of sensors to be installed
                SensorMetadataObject.evaluateBusiness(chromosome, taskType="opCost"),  ## power needed to run the solution 
                # SensorMetadataObject.evaluateBusiness(chromosome, taskType="coverage"),  ## power needed to run the solution 
                
                ]

    return scoreArray

class SolverBuildings:

    def __init__(self, dataPath, start_index =0, end_index=100, floors=[4,5,6,7], groupBy="zone", output_path="./paperAnalysis/") -> None:
        self.taskGenerator = TaskGenerator(dataPath, start_index, end_index, floors, groupBy, output_path)
        self.n_obj = 2
        self.n_var = len(self.taskGenerator.sensorLabels)*len(self.taskGenerator.dataDictionary.keys())
        self.lower_limit =[0]*self.n_var
        self.upper_limit =[1]*self.n_var
        self.res=None
        self.problem=None
        self.algorithm=None
        self.output_path=output_path
        
        # self.sensorLabels = ["ACPower","lightPower","appPower","temperature","humidity","lux"]
        pass

    def initMemoryLearners(self):
        self.taskGenerator.initMemoryTable()

    def solveOptimization(self):
        self.problem = sensorOptimizingProblem(n_var=self.n_var, n_obj=self.n_obj, n_constr=1, xl=self.lower_limit, xu=self.upper_limit, type_var=bool, )
        self.algorithm = returnMethod(optimizationTypeBool=False)

        self.res = minimize(problem = self.problem,
                    algorithm = self.algorithm,
                    termination=('n_gen', 40),
                    seed=1,
                    save_history=True
                    )

        print("Best solution found: %s" % self.res.X)
        print("Function value: %s" % self.res.F)
        print("Constraint violation: %s" % self.res.CV)

    def plotParetoSolution(self):
        _X = np.row_stack([a.pop.get("X") for a in self.res.history])
        feasible = np.row_stack([a.pop.get("feasible") for a in self.res.history])[:, 0]

        plotting.plot(_X[feasible], _X[np.logical_not(feasible)], self.res.X[None,:]
                    , labels=["Feasible", "Infeasible", "Best"])

        plot = Scatter(title="Pareto Curve")
        plot.add(self.problem.pareto_front(use_cache=False, flatten=False), plot_type="line", color="black")
        plot.add(self.res.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
        plot.show()

    def plotSolution(self,):
        res_data=self.res.F.T
        # fig = go.Figure(data=go.Scatter(x=res_data[xIndex], y=res_data[yIndex], mode="markers"))
        # fig.show()
        
        accuracy = res_data[0] 
        # accuracy = (accuracy-min(accuracy))/(max(accuracy)- min(accuracy))
        opCost = res_data[2]
        # opCost = (opCost-min(opCost))/(max(opCost)- min(opCost))
        installCost =  res_data[1]
        sol = pd.DataFrame.from_dict({"accuracy":accuracy, "opCost":opCost, "installCost":installCost}).sort_values(by='installCost')
        sol.to_csv(self.output_path+"tradeoff.csv")
        chromosome = pd.DataFrame(self.res.X)
        chromosome.to_csv(self.output_path+"chromosomes.csv")
        # plt.plot(sol.installCost, sol.opCost, marker="+", label =   "operating power")
        # plt.plot(sol.installCost, sol.accuracy, marker="^", label = "aprroximation error")
        # plt.xlabel("Sensor Cost")
        # plt.legend()
        # import pdb; pdb.set_trace()
        # plt.savefig(self.output_path+"tradeoff.pdf", format="pdf", bbox_inches="tight")
        # plt.show()

    def zonalSolutionAnalysis(self):
        start_index = 0
        inferenceDict = []
        for key,value in self.taskGenerator.dataDictionary.items():
            end_index= start_index+len(value.columns)
            chrEncoding = self.res.X[0][start_index:end_index]
            numSensors = sum(chrEncoding)
            inferenceDict.append({
                "zone" : key,
                "numSensors" : numSensors,
                "sensorsSaved" : np.round(1 - numSensors/(end_index-start_index),2),
                "requiredSensors" : [self.taskGenerator.sensorLabels[key][i] for i,v in enumerate(chrEncoding) if v > 0],
                "approximatedSensors" : [self.taskGenerator.sensorLabels[key][i] for i,v in enumerate(chrEncoding) if v < 1]
            })
            start_index = end_index
        print (inferenceDict)
        print (f"Sensors used {sum(self.res.X[0])} out of {(len(self.res.X[0]))}" )
        return inferenceDict

def initVirtualSenseField(dataPath, start_index , end_index, floors, groupBy, output_path):
    global SensorMetadataObject
    global SmartBuilingObject
    SensorMetadataObject = SensorMetadata(groupBy)
    SmartBuilingObject = SolverBuildings(dataPath, start_index, end_index, floors, groupBy,output_path)

def createVirtualSenseField():
    SmartBuilingObject.initMemoryLearners()
    # SmartBuilingObject.solveOptimization()
    # SmartBuilingObject.zonalSolutionAnalysis()
    # SmartBuilingObject.plotSolution()

def optimizeVirtualSenseField():
    SmartBuilingObject.solveOptimization()
    pass

def zonalAnalysis():
    SmartBuilingObject.zonalSolutionAnalysis()
    # import pdb; pdb.set_trace()
    SmartBuilingObject.plotSolution()


def reloadResults():

    return