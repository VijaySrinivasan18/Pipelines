config={
# ["DT","RF","GLM"]
# ["KNN","GLM","SVC"]
    "model":"All",
    "parameters":{
    "GLM":{
        "penalty":"l2",
        "max_iter":3000
        
    },
    "KNN":{
        "n_neighbors":5
        },
    "DT":{},
    "RF":{},
    "SVC":{}
        
    },
    "validate":True,
    "model_build":True,
    "split":0.3,
    "predict":True,
    "cross_val":True,
    "cv":5,
    "version":[1],
    "store_results":True



}

