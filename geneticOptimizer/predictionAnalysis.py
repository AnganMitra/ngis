import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import plotly.express as px
import argparse as agp
expMode = {
    "45f": [4,5],
    "67f" : [6,7],
    "456f" : [4,5,6],
    "567f" : [5,6,7],
    "4567f": [4,5,6,7],
    "4f" : [4],
    "6f" : [6],
    "5f" : [5],
    "7f" : [7]
}
rdf = {}
# filePath = "./paperAnalysis/"

def plotPredPerfbyKey(key, output_path):
    try:
        # rdf[key].plot(marker=11)
        fig = px.imshow(rdf[key])
        fig.write_image(f"{output_path}{key}.png",scale=2)
        
        # fig.show()
        # plt.show()
    except:
        import pdb; pdb.set_trace()
        pass
    finally:
        plt.clf()

if __name__=="__main__":


    parser = agp.ArgumentParser()

    parser.add_argument("-o", "--output_dir", help="path of the output data folder")
    parser.add_argument("-g", "--groupBy", help="Group sensors by -- random, zone, domain")
    parser.add_argument("-c", "--config", help="Experiment keys available -- "+str(expMode))

    args = vars(parser.parse_args())
    

    try:
        expInput = args['config']
        # print ("config : ",config)
        
    except:
        pass
    try:
        expInput = args['config']
        # print ("config : ",config)
        
    except:
        pass

    try:
        groupBy = args['groupBy'].strip()
        # print ("end_index : ", end_index)
    except:
        pass
    

    try:
        output_dir = args['output_dir']
        output_dir +=f"{expInput}-{groupBy}/"
        print ("output_dir : ", output_dir)
        os.mkdir(output_dir)
    except:
        pass
    for file in os.listdir(output_dir):
        if file.endswith(".csv"):
            key = file.split(".")[0]
            try:
                rdf[key]= pd.read_csv(output_dir+file, )
                # import pdb; pdb.set_trace()
                rdf[key][rdf[key]<0] = 0
                rdf[key].index = rdf[key].columns[1:]
                rdf[key]=rdf[key].drop('Unnamed: 0', axis = 1)
                rdf[key]=rdf[key][sorted(rdf[key].columns)].sort_index(key=lambda x: x.str.lower())
            except:
                print (file+" processing error.... ")
        # import pdb; pdb.set_trace()
    for key in rdf.keys(): plotPredPerfbyKey(key, output_path=output_dir)



