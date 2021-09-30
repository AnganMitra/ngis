from taskGenerator import TaskGenerator
import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.util import plotting
from pymoo.algorithms.moo.nsga2 import NSGA2

def compareChromosomes(chrA, chrB):
    distance = 0
    return distance

class sensorOptimizingProblem(Problem):

    def __init__(self, n_var=2, n_obj=1, n_constr=1, xl=0, xu=10, type_var=int):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, type_var=type_var,)

    def _evaluate(self, x, out, *args, **kwargs):
        # evaluate the fitness function for the generation 
        res = []
        for chromosome in x:
            # evaluate chromosome fitness
            fitness = multiObjectiveScore(chromosome)
            res.append(fitness)
        # import pdb; pdb.set_trace()
        out["F"] = np.array(res)
        out["G"] = -np.array(res)

        # out["F"] = - np.min(x * [3, 1], axis=1)
        # add constraints via specifying constraint(x) <=0 formula as below
        # out["G"] = x[:, 0] + x[:, 1] - 10 
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
    scoreArray = [offlineTaskGen.evaluateAILoss(chromosome, taskType="forecasting") , ## Forecasting error difference
                offlineTaskGen.evaluateAILoss(chromosome, taskType="control"), ## power
                # offlineTaskGen.evaluateBusiness(chromosome, taskType="installCost"), ## Cost of sensors to be installed
                # offlineTaskGen.evaluateBusiness(chromosome, taskType="opCost"),  ## power needed to run the solution 
                ]

    return scoreArray



offlineTaskGen = TaskGenerator(dataPath="./BKDataCleaned/")
offlineTaskGen.initMemoryTable()
n_var=len(offlineTaskGen.sensorLabels)*len(offlineTaskGen.dataDictionary.keys())
n_obj = 2
lower_limit =[0]*n_var
upper_limit =[1]*n_var

problem = sensorOptimizingProblem(n_var=n_var, n_obj=n_obj, n_constr=1, xl=lower_limit, xu=upper_limit, type_var=bool, )
algorithm = returnMethod(optimizationTypeBool=False)

res = minimize(problem = problem,
               algorithm = algorithm,
               termination=('n_gen', 40),
               seed=1,
               save_history=True
               )

print("Best solution found: %s" % res.X)
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)


_X = np.row_stack([a.pop.get("X") for a in res.history])
feasible = np.row_stack([a.pop.get("feasible") for a in res.history])[:, 0]

plotting.plot(_X[feasible], _X[np.logical_not(feasible)], res.X[None,:]
              , labels=["Feasible", "Infeasible", "Best"])

plot = Scatter(title="Pareto Curve")
plot.add(problem.pareto_front(use_cache=False, flatten=False), plot_type="line", color="black")
plot.add(res.F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
plot.show()


# import numpy as np
# from pymoo import algorithms
# from pymoo.optimize import minimize
# from pymoo.core.problem import Problem
# from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
# import pymoo.algorithms.moo.nsga2 as nsga2
# import plotly.graph_objects as go

# class ProblemWrapper(Problem):

#     def _evaluate(self, x, out, *args, **kwargs):
#         res = []
#         for design in x:
#             res.append

#         out["F"] = np.array(res)
#         return super()._evaluate(x, out, *args, **kwargs)

# sensorList = []


# numVar = len(sensorList)
# lower_limit =[0]*numVar
# upper_limit =[1]*numVar
# problem = ProblemWrapper(n_var=numVar, n_obj=2, xl=lower_limit, xu=upper_limit )

# algorithm = nsga2(pop_size=100)
# stop_criteria = ("n_gen",100)

# results = minimize(
#     problem=problem,
#     algorithm= algorithm,
#     termination= stop_criteria
# )

# res_data=results.F.T
# fig = go.Figure(data=go.Scatter(x=res_data[0], y=res_data[1], mode="markers"))
# fig.show()