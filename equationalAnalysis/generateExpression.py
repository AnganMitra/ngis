import numpy as np
import pysindy as ps
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore", r"/*" )

import json

dataPath = "./BKDataCleaned/" 
outputPath = "equationalDecomposition/"
floorID = 7
zoneID = 1
testPercentage = 0.9
resamplingFreq = "30S"
# start_index= int(sys.argv[1].strip()) # 150
# end_index = int(sys.argv[2].strip()) # 400
step_size = 36


plt.rcParams.update({  "text.usetex": True,  "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})

def deepEstimator(trainD, dt):

    x_train, x0_train, t_train = trainD
    estimator = ps.deeptime.SINDyEstimator(t_default=dt)
    estimator.fit(x_train)

    # Extract a model
    model = estimator.fetch_model()

    # Evolve the new initial condition in time with the SINDy model
    x_test_sim = model.simulate(x0_train, t_train)

    return x_test_sim, model

def plotFigure(x_sim, columns):
    plot_kws = dict(linewidth=2)

    fig, axs = plt.subplots(1, columns, figsize=(10, 4))
    t_plot = np.arange(0, x_sim.shape[0])
    for i in range(columns):
        axs[i].plot(t_plot, x_sim[:, i], "r", label=f"$x_{i}$", **plot_kws)
        axs[i].set(xlabel="t", ylabel=columnList[i])
        fig.show()
    plt.show()

def plotCompFigure(truth,predicted, columns, outputPath = "equationalDecomposition/"):
    plot_kws = dict(linewidth=2)
    # fig = plt.figure()
    # axs = [0]*(2*(columns))
    # for i in range(0,columns):
    #     axs[2*i] = fig.add_subplot(2, 1, 1)
    #     axs[2*i+1] = fig.add_subplot(2, 1, 2, sharex=axs[2*i])
    fig, axs = plt.subplots(1, columns-1, figsize=(30, 8))
    t_plot = np.arange(0, predicted.shape[0])
    # import pdb; pdb.set_trace()
    for i in range(0,columns):
        if columnList[i] == "dateP" : continue
        y_axis = [ 
           min(min(truth[:, i]), min(predicted[:, i])),
           max(max(truth[:, i]), max(predicted[:, i])),
        ]
        axs[i].plot(t_plot, truth[:, i], "g", label=f"$T-x_{i}$", **plot_kws)
        axs[i].plot(t_plot, predicted[:, i], "r", label=f"P-$x_{i}$", **plot_kws)
        axs[i].set(xlabel="t", ylabel=columnList[i])
        # axs[2*i+1].set(xlabel="t", ylabel=f"P-{columnList[i]}")
        axs[i].set_ylim(y_axis[0], y_axis[1])
        
        # axs[2*i+1].set_ylim(y_axis[0], y_axis[1])
        
    # plt.show()
    plt.savefig(f"{outputPath}GenX-{start_index}-{end_index}.png", dpi = 200)
    plt.clf()
    # fig.text(-0.8, 0.8, "Hello world\niiiii iii $\\frac{1}{2}$", fontsize=30, family='monospace')
    # plt.show()



def readData(columnList):
    filename = f"{dataPath}Floor{floorID}Z{zoneID}.csv"
    filename = f"{dataPath}F4Z5-working-hour.csv"
    df = pd.read_csv(filename, index_col=["Date"], parse_dates=True)

    # rdf = df.resample(resamplingFreq).agg("mean")
    # print (len(df))
    # rdf = df.resample(resamplingFreq).interpolate(method ='linear', limit_direction ='forward', limit = 10).dropna()[columnList]
    rdf = df.ffill().dropna()[columnList]
    # rdf = df.resample(resamplingFreq).pad().dropna()[columnList]
    # rdf = df.rolling(6).mean().dropna()[columnList]s
    
    rdf = (rdf - rdf.min()) / (rdf.max() - rdf.min())
    # print(len(rdf))
    

    return rdf

def splitData(rdf,  start_index, end_index):
    # import pdb; pdb.set_trace()
    pdf = rdf.iloc[start_index:end_index]
    
    x_train = pdf.to_numpy()
    # x_train = rdf.iloc[:int(len(rdf)*testPercentage),:].to_numpy()
    # x_train = x_train[start_index:end_index]
    x0_train = x_train[0]
    t_train = np.linspace(0, 1, int(len(x_train)))

    x_test = rdf.iloc[:,:].to_numpy()
    x0_test = rdf.iloc[0,:].to_numpy()
    """"
    check on train data
    """
    x0_test = x_train[0]
    t_test = np.linspace(0, 1, len(t_train))

    return (x_train, x0_train, t_train), (x_test, t_test, x0_test)


def crossValEquation(trainD, dt):
    x_train, x0_train, t_train = trainD
    model = ps.SINDy(t_default=dt)

    param_grid = {
        "optimizer__threshold": [1e-3, 1e-4, 1e-5, 5e-4 ],
        "optimizer__alpha": [5e-2, 5e-3, 5e-4,],
        "feature_library": [ ps.FourierLibrary(), ps.PolynomialLibrary() ],
        "differentiation_method__order": [1,2,3,4],
    
    }

    search = GridSearchCV(
        model,
        param_grid,
        cv=TimeSeriesSplit(n_splits=3)
    )

    search.fit(x_train)

    print("Best parameters:", search.best_params_)
    # search.best_estimator_.print()
    x_sim = search.best_estimator_.simulate(x0_train,t_train)
    return x_sim, search.best_estimator_

def customEq(x_train, dt ):
    x_train, x0_train, t_train = trainD
    fd = ps.FiniteDifference(drop_endpoints=True)
    fd = ps.SmoothedFiniteDifference(drop_endpoints=True)
    
    library_functions = [
        lambda x : np.exp(x),
        lambda x : 1./x,
        lambda x : x,
        lambda x,y : np.sin(x+y)
    ]
    library_function_names = [
        lambda x : 'exp(' + x + ')',
        lambda x : '1/' + x,
        lambda x : x,
        lambda x,y : 'sin(' + x + ',' + y + ')'
    ]
    custom_library = ps.CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )

    custom_library = ps.FourierLibrary()+ps.PolynomialLibrary()+ps.IdentityLibrary()

    model = ps.SINDy(feature_library=custom_library,
                        differentiation_method=fd,
                        discrete_time=True)

    model.fit(x_train, t=t_train, multiple_trajectories=True)
    model.print()
    x_sim = model.simulate(x0_train,t_train)
    return x_sim, model

def addDatePattern(rdf):
    # import pdb; pdb.set_trace()
    rdf["dateP"] = rdf.index.hour*60+rdf.index.minute
    return rdf

def dumpEquations(model, init, testPerf, lhs=None,  precision=3):
    eqns = model.equations(precision)
    stringModel = ""
    for i, eqn in enumerate(eqns):
        if model.discrete_time:
            strTok= (model.feature_names[i] + "[k+1] = " + eqn)
        elif lhs is None:
            strTok = (model.feature_names[i] + "' = " + eqn)
        else:
            strTok = (model[i] + " = " + eqn)
        stringModel +=strTok+"\n"
    payload = {
        "dynamics" :  stringModel,
        "x0_initial":  list(init), 
        "testPerformance " : testPerf,
        "variables" : columnList
    }
    # import pdb; pdb.set_trace()
    return payload

columnList = [ "temperature", "ACPower"]
print ([f"x{i} : {v}" for i,v in enumerate(columnList)])

rdf = readData(columnList)
rdf = addDatePattern(rdf); columnList.append("dateP")
lossEvolution = []

try:
    for start_index in range(0, int(len(rdf)*testPercentage), step_size):
        
        end_index = min(start_index+step_size, len(rdf)-1)

        # start_index,end_index = 1805, 1818
        trainD, testD = splitData(rdf, start_index, end_index)
        # plotFigure(trainD[0], columns= len(columnList))


        dt = 1/len(trainD[-1])
        # x_sim, model = deepEstimator(trainD, dt )
        x_sim, model = crossValEquation(trainD, dt)
        # plotFigure(x_sim, columns= len(columnList))

        x_test, t_test, x0_test = testD
        # if start_index > 1 : x0_test = x_sim[-1]
        # Compare SINDy-predicted derivatives with finite difference derivatives
        x_sim = model.simulate(x0_test, t_test)
        testPerf = model.score(testD[0], t=1/len(testD[0]))
        # import pdb; pdb.set_trace()
        # print ("-------------DYNAMICS----------------")
        # print (f"Start/End : {start_index}/{end_index}" )
        # print (f"Model Init : {trainD[1]}" )
        # print (f"Equations ", ); model.print()
        # print (f"Test Perf ", testPerf)
        # print ()
        lossEvolution.append(testPerf)
        payload = dumpEquations(model, trainD[1], testPerf, lhs=None,  precision=3)
        json.dump( payload,open(f"{outputPath}ModG-{start_index}-{end_index}.json", "w"))
        plotCompFigure(trainD[0],x_sim, columns= len(columnList))
        # exit()

except KeyboardInterrupt:
    # import pdb; pdb.set_trace()
    pass
finally:
    plt.clf()
    plt.plot(np.arange(0,len(lossEvolution)), lossEvolution, label="Modelling Loss")
    plt.savefig(f"{outputPath}Loss-{0}-{end_index}.png", dpi = 200)
    # plt.savefig()
    sys.exit()
