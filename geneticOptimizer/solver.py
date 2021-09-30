from geneticOptimizer.taskGenerator import TaskGenerator
import numpy as np
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
from pymoo.util import plotting

offlineTaskGen = TaskGenerator(dataPath)

def compareChromosomes(chrA, chrB):
    distance = 0
    return distance

# evaluate forecasting power of a chromosome
def multiObjectiveScore(chromosome):
    scoreArray = [offlineTaskGen.evaluateForecastingLoss(chromosome) , ## Forecasting error difference
                offlineTaskGen.evaluateOperationalCost(chromosome) , ## Cost of sensors to be installed
                offlineTaskGen.evaluateOperationalPower(chromosome),  ## power needed to run the solution 
                offlineTaskGen.evaluateControlLoss(chromosome), ## power
                ]

    return scoreArray

class sensorOptimizingProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=0, xu=10, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # evaluate the fitness function for the generation 
        res = []
        for chromosome in x:
            # evaluate chromosome fitness
            res.append(multiObjectiveScore(chromosome))

        out["F"] = np.array(res)
        # out["F"] = - np.min(x * [3, 1], axis=1)
        # add constraints via specifying constraint(x) <=0 formula as below
        # out["G"] = x[:, 0] + x[:, 1] - 10 
        return super()._evaluate(x, out, *args, **kwargs)

def returnMethod(optimizationTypeBool=True):
    method = None
    if not optimizationTypeBool:
        method = get_algorithm("ga",
                            pop_size=20,
                            sampling=get_sampling("int_random"),
                            crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                            mutation=get_mutation("int_pm", eta=3.0),
                            eliminate_duplicates=True,
                            ) 
    else:
        method = get_algorithm("ga",
                        pop_size=200,
                        sampling=get_sampling("bin_random"),
                        crossover=get_crossover("bin_hux"),
                        mutation=get_mutation("bin_bitflip"),
                        eliminate_duplicates=True
                        )
    
    return method


problem = sensorOptimizingProblem()
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