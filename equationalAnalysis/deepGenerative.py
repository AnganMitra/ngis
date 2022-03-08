#!/usr/bin/env python
# coding: utf-8

# # Deeptime objects
# 
# This notebook showcases two PySINDy objects designed according to the [Deeptime](https://deeptime-ml.github.io/index.html) API:
# * `SINDyEstimator` - An estimator object which acts as a sort of factory for generating model objects
# * `SINDyModel` - The SINDy model object which is learned from data and created by an estimator

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

import pysindy as ps


# In[ ]:


def lorenz(z, t):
    return [
        10 * (z[1] - z[0]),
        z[0] * (28 - z[2]) - z[1],
        z[0] * z[1] - (8 / 3) * z[2]
    ]

# Generate measurement data
dt = .002

# Train data
t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
x_train = odeint(lorenz, x0_train, t_train)

# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, 15, dt)
x0_test = np.array([8, 7, 15])
x_test = odeint(lorenz, x0_test, t_test)


# In[ ]:


# Fit an estimator
estimator = ps.deeptime.SINDyEstimator(t_default=dt)
estimator.fit(x_train)


# The underlying model, represented by the `SINDyModel` class, comes equipped with all the methods of the `SINDy` class that are available after fitting (e.g. `predict`, `score`, `simulate`, `print`, `equations`). 

# In[ ]:


# Extract a model
model = estimator.fetch_model()

# Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, t=dt))


# In[ ]:


# Evolve the new initial condition in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_test, x_test_sim[:, i], 'r--', label='model simulation')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))


# In[ ]:


# Try out some other combinations of methods
estimator = ps.deeptime.SINDyEstimator(
    optimizer=ps.SR3(threshold=0.5, max_iter=50),
    feature_library=ps.PolynomialLibrary(degree=3)
)
estimator.fit(x_train, t=dt)

model = estimator.fetch_model()
model.print()

