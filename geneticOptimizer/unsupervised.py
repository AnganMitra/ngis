
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.mixture import GaussianMixture
import yaml
plot2result =False
import pandas as pd
import argparse
import warnings
warnings.simplefilter(action='ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import pickle
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
import sys
sys.path.append("./learner/")
from datapreprocessing.dataPreprocessing import add_scalers, runInference, add_rows

def engageModel (spacekey, model, X, task):
    
    if task == "train":
        model.fit(X)
        pickle.dump(model, open("models/{}".format( type(model).__name__), 'wb'))
        
    else:
        model = pickle.load(open("models/{}".format( type(model).__name__), 'rb'))

    y_predicted = model.predict(X)
    
    return y_predicted


algorithm_dict={
    # 'kmeans' : kmeans(n_clusters=2,init='k-means++', n_init = 40, max_iter = 1000, random_state= 42) , 
    # 'agglomerativeclustering' : AgglomerativeClustering(n_clusters=2,affinity = 'euclidean', linkage = 'ward') ,
    'gmm': GaussianMixture(n_components=3, random_state=42),
    # 'dbscan' : dbscan(eps=10, min_samples=5) ,
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
            
            if plot2result:  plotResultFrame(points, algorithm, algorithms, "{}.png".format(algorithm))
        except:
            print ("Fail: {}".format(algoname))
        try:
            runInference(payload[target], result)
        except:
            pass
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
            x.to_csv("models/{}-{}".format(spacekey,algoname))
        except:
            pass
    return cluster_center


    
def learn(filename, spacekey,output_filename,task):
    df = pd.read_csv(filename, index_col=["time"], parse_dates=True)
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    # import pdb; pdb.set_trace()
    target = "babbage.presence"
    remove_columns = ["jacqaurd.motion", "babyfoot.motion"]
    payload = df[[x for x in df.columns if spacekey in x and x not in remove_columns]]
    run_unsupervisedLearning(spacekey,payload, output_filename, task= task, target=target)


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", help="path of the yaml file.")
    parser.add_argument("-o", "--output_filename", help="path of the output directory.")
    parser.add_argument("-s", "--spacekey", help="Provide spacekey")
    parser.add_argument("-lt", "--task", help= "Default: train, 'test' to test model, 'train' to build model")

    args = vars(parser.parse_args())
    input_file = None
    spacekey = None
    output_filename = None
    task = True
    try:
        input_file = args["input_file"]
    except:
        print ("Data file missing in arg")
        exit()

    try:
        output_filename = args["output_filename"]
    except:
        pass

    try:
        spacekey = args["spacekey"]
    except:
        pass

    try:
        task = args["task"]
    except:
        task = "train"
        

    learn(input_file, spacekey,output_filename,task)
    