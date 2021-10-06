import argparse as agp
import solver as VirtualSensorField
expMode = {
    "45f": "floor45",
    "67f" : "floor67",
    "456f" : "floor456",
    "567f" : "floor567",
    "4567f": "floor4567",
    "4f" : "floor4",
    "6f" : "floor6",
    "5f" : "floor5",
    "7f" : "floor7"
}
taskType = {
    "sensLoc" : "Get Optimal Sensor Locations", 
    "vsfGen": "Generate Virtual Sensor Field" ,
    "zonAnaly" : "Analyse accuracy, capex/opex"
}

if __name__=="__main__":


    parser = agp.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="path of the input data folder")
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
    
    VirtualSensorField.initVirtualSenseField(input_dir)
    VirtualSensorField.reloadResults()
    if task == "vsfGen": 
        VirtualSensorField.updateVirtualSenseField()
    # elif task == "sensLoc":
    #     VirtualSensorField.resultAnalyse()
    elif task == "zonAnaly":
        VirtualSensorField.zonalAnalysis()