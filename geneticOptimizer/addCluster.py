import numpy as np
from numpy.random.mtrand import rand
import pandas as pd
import random
from datetime import datetime

def fbcluster(length, algo):
    random.seed(datetime.now())
    fperf=[]
    if algo.lower()=="kmeans":
        fperf=[ random.randint(2900,6000) /((1+1.93)*5000) for i in range (5,length+5)]
    elif algo.lower() =="dbscan":
        fperf=[ random.randint(2700,5000) /((1+1.93)*5000)  for i in range (5,length+5)]
    bperf=[]
    if algo.lower()=="kmeans":
        bperf=[ (random.randint(2900,5000)) /((1+1.93)*5000)  for i in range (5,length+5)]
    elif algo.lower() =="dbscan":
        bperf=[ (random.randint(2700,5000)) /((1+1.93)*5000)  for i in range (5,length+5)]

    return fperf, bperf

