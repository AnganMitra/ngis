import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


filePath = "./paperAnalysis/"
expMode = "234567f"
folderOptions = [f"{expMode}-domain",]  # f"{expMode}-domain-run0-identicalConf"
chromosomes={}
functionalValChr = {}
for expOutput in folderOptions:
    chromosomes[expOutput]=pd.read_csv(filePath+expOutput+"/chromosomes.csv").T.iloc[1:,:]
    functionalValChr[expOutput]=pd.read_csv(filePath+expOutput+"/functionalValchromosomes.csv")

filePath +="paperFigures/"
for expOutput in folderOptions:
    chromosomes[expOutput]
