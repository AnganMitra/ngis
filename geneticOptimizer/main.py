import argparse as agp
import os
import solver as VirtualSensorField
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
taskType = {
    "optVsf" : "Get Optimal Sensor Locations", 
    "vsfGen": "Generate Virtual Sensor Field" ,
    "zonAnaly" : "Analyse accuracy, capex/opex"
}

if __name__=="__main__":


    parser = agp.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="path of the input data folder")
    parser.add_argument("-o", "--output_dir", help="path of the output data folder")
    parser.add_argument("-ts", "--start_index", help="Starting Index Req Integer")
    parser.add_argument("-te", "--end_index", help="End Index Req Integer")
    parser.add_argument("-g", "--groupBy", help="Group sensors by -- random, zone, domain")
    parser.add_argument("-c", "--config", help="Experiment keys available -- "+str(expMode))
    parser.add_argument("-tk", "--task", help="Task types available -- "+str(taskType ))
    args = vars(parser.parse_args())
    
    try:
        input_dir = args['input_dir']
        # print ("input_dir : ", input_dir)
    except:
        pass
    
    
    try:
        expInput = args['config']
        # print ("config : ",config)
        
    except:
        pass

    try:
        
        start_index = int(args['start_index'].strip())
        # print ("start_index : ", start_index)
    except:
        pass

    try:
        end_index = int(args['end_index'].strip())
        # print ("end_index : ", end_index)
    except:
        pass

    try:
        groupBy = args['groupBy'].strip()
        # print ("end_index : ", end_index)
    except:
        pass
    
    try:
        task = args['task']
        # print ("task : ",task)
    except:
        pass

    try:
        output_dir = args['output_dir']
        output_dir +=f"{expInput}-{groupBy}/"
        print ("output_dir : ", output_dir)
        os.mkdir(output_dir)
    except:
        pass
    VirtualSensorField.initVirtualSenseField(dataPath=input_dir, start_index =start_index, end_index=end_index, floors=expMode[expInput], groupBy=groupBy, output_path=output_dir)
    VirtualSensorField.reloadResults()
    if task == "vsfGen": 
        VirtualSensorField.createVirtualSenseField()
    else:
        VirtualSensorField.reloadResults()
    if task == "optVsf":
        VirtualSensorField.optimizeVirtualSenseField()
    elif task == "zonAnaly":
        VirtualSensorField.zonalAnalysis()