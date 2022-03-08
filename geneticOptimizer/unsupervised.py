
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN,KMeans
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.mixture import GaussianMixture
import yaml
plot2result =False
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import pickle
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
import sys
import dataLoader as ld
def engageModel (spacekey, model, X, task):
    
    if task == "train":
        model.fit(X)
        pickle.dump(model, open("models/{}".format( type(model).__name__), 'wb'))
        
    else:
        model = pickle.load(open("models/{}".format( type(model).__name__), 'rb'))

    y_predicted = model.predict(X)
    
    return y_predicted


algorithm_dict={
    'kmeans' : KMeans(n_clusters=23,init='k-means++', n_init = 40, max_iter = 100, random_state= 42) , 
    # 'agglomerativeclustering' : AgglomerativeClustering(n_clusters=2,affinity = 'euclidean', linkage = 'ward') ,
    # 'gmm': GaussianMixture(n_components=3, random_state=42),
    'dbscan' : DBSCAN(eps=10, min_samples=5) ,
    # 'optics' : optics(min_samples=50, xi=.05, min_cluster_size=5),
    
}

def run_unsupervisedLearning(spacekey,payload, output_filename, task, target=None):
    # import pdb; pdb.set_trace()
    algo_payload = payload.loc[:, payload.columns != target].copy(deep=True)
    for algoname, algorithm in algorithm_dict.items():
        try:
            # import pdb; pdb.set_trace()
            result = engageModel(spacekey,algorithm,algo_payload,task =task)
            result[result<2]=0
            result[result==2]=1
            payload["{}.{}".format(spacekey,algoname)] = result
            
        except:
            print ("Fail: {}".format(algoname))
        
    # import pdb; pdb.set_trace()
    keylist = algo_payload.columns
    cluster_center = runClusterInference( spacekey,payload,keylist)

    if not output_filename: 
        payload.to_csv("csv/uL-{}.csv".format(spacekey)) 
    else: 
        payload.to_csv(output_filename)
    return cluster_center


def runClusterInference( spacekey,payload, keylist):
    
    # import pdb; pdb.set_trace()
    cluster_center ={}

    for algoname, _ in algorithm_dict.items():
        try:
            # print ("{}.{}".format(spacekey,algoname))
            x = pd.concat([payload[payload[ "{}.{}".format(spacekey,algoname)] == i][keylist].agg(['mean'],axis = 0)  for i in set(payload[ "{}.{}".format(spacekey,algoname)] ) ], axis = 0)
            x = pd.DataFrame(x)
            x["count"] = [len(payload[payload[ "{}.{}".format(spacekey,algoname)] == i])  for i in set(payload[ "{}.{}".format(spacekey,algoname)] ) ]
            cluster_center["{}-{}".format(spacekey, algoname)] = x[x.index == "mean"]
            x.to_csv("paperAnalysis/unsupLearn/{}-{}.csv".format(spacekey,algoname))
        except:
            pass
    return cluster_center


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="path of the input dir")
    parser.add_argument("-o", "--output_dir", help="path of the output directory.")
    parser.add_argument("-s", "--spacekey", help="Provide spacekey")
    parser.add_argument("-lt", "--task", help= "Default: train, 'test' to test model, 'train' to build model")

    args = vars(parser.parse_args())
    input_file = None
    spacekey = None
    output_filename = None
    task = True
    try:
        input_dir = args["input_dir"]
    except:
        print ("Data file missing in arg")
        exit()

    try:
        output_dir = args["output_dir"]
    except:
        pass
        

    try:
        task = args["task"]
    except:
        task = "train"
    '''
    # to dump data
    sensorDictionary, encodingLabels= ld.getSpatialGroupData(dataPath = input_dir, 
                                        start_index=0, end_index =400000, 
                                        floors = [2,3,4,5,6,7])

    flatTable = []
    for k,v in sensorDictionary.items():
        flatTable.append(v)
        flatTable[-1].columns= [f"{k}-{j}" for j in flatTable[-1].columns]
    
    # allzoneallsensorData = pd.concat(flatTable, axis= 0).bfill().ffill().fillna(0)
    allzoneallsensorData= pd.concat(flatTable, axis= 0).ffill(axis="columns").bfill(axis="columns").dropna()
    allzoneallsensorData.to_csv("./paperAnalysis/unsupLearn/bigData.csv")
    '''    
    allzoneallsensorData = pd.read_csv(input_dir, index_col=["Date"], parse_dates=True)[:10000].T
    task="train"
    run_unsupervisedLearning(spacekey,payload=allzoneallsensorData, output_filename=output_dir, task= task, )
    