import numpy as np
import pysindy as ps
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt

dataPath = "./BKDataCleaned/"
floorID = 7
zoneID = 1
testPercentage = 0.1
resamplingFreq = "300s"
dt = 1 # np.round(1/12.0,3)


df = pd.read_csv(f"{dataPath}Floor{floorID}Z{zoneID}.csv", index_col=["Date"], parse_dates=True)
# rdf = df.resample(resamplingFreq).agg("mean")
rdf = df.resample("1s").ffill().dropna()[["ACPower", "temperature"]]
rdf = (rdf - rdf.min()) / (rdf.max() - rdf.min())

x_train = rdf.iloc[:int(len(rdf)*testPercentage),:].to_numpy()
t_train = np.arange(0, int(len(rdf)*testPercentage), dt)

x_test = rdf.iloc[:,:].to_numpy()
x0_test = rdf.iloc[0,:].to_numpy()
t_test = np.arange(0, len(rdf), dt)

def deepEstimator():


    estimator = ps.deeptime.SINDyEstimator(t_default=dt)
    estimator.fit(x_train)

    # Extract a model
    model = estimator.fetch_model()

    # Compare SINDy-predicted derivatives with finite difference derivatives
    print('Model score: %f' % model.score(x_test, t=dt))

    # Evolve the new initial condition in time with the SINDy model
    x_test_sim = model.simulate(x0_test, t_test)

    fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_test.shape[1]):
        axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
        axs[i].plot(t_test, x_test_sim[:, i], 'r--', label='model simulation')
        axs[i].legend()
        axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))
    fig.show()
    plt.show()

def crossValEquation():

    model = ps.SINDy(t_default=dt)

    param_grid = {
        "optimizer__threshold": [1e-3, 1e-4, 1e-5, 5e-4, ],
        "optimizer__alpha": [5e-2, 5e-3, 5e-4,],
        "feature_library": [ps.PolynomialLibrary(), ps.FourierLibrary(), ],
        "differentiation_method__order": [1,2,3,],
    
    }

    search = GridSearchCV(
        model,
        param_grid,
        cv=TimeSeriesSplit(n_splits=15)
    )



    search.fit(x_train)

    print("Best parameters:", search.best_params_)
    search.best_estimator_.print()



    x_sim = search.best_estimator_.simulate(x0_test,t_test)

    plot_kws = dict(linewidth=2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(t_train, x_train[:, 0], "r", label="$x_0$", **plot_kws)
    axs[0].plot(t_train, x_train[:, 1], "b", label="$x_1$", alpha=0.4, **plot_kws)
    axs[0].plot(t_test, x_sim[:, 0], "k--", label="model", **plot_kws)
    axs[0].plot(t_test, x_sim[:, 1], "k--")
    axs[0].legend()
    axs[0].set(xlabel="t", ylabel="$x_k$")

    axs[1].plot(x_train[:, 0], x_train[:, 1], "r", label="$x_k$", **plot_kws)
    axs[1].plot(x_sim[:, 0], x_sim[:, 1], "k--", label="model", **plot_kws)
    axs[1].legend()
    axs[1].set(xlabel="$x_1$", ylabel="$x_2$")
    fig.show()
    plt.show()

# crossValEquation()
deepEstimator()