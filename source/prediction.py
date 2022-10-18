import pickle
import os


model_dict={"DT":"DecisionTreeClassifier","RF":"RandomForestClassifier","SVC":"SVC",
            "GLM":"LogisticRegression",
            "KNN":"KNeighborsClassifier"}


def model_prediction(versions=[1],model=None,x=None):
    prediction_dict={}
    for version in versions:

        if model is not None and x is not None:
            if model=="All":
                for model_name in model_dict:
                    if f"{model_name}_object_{version}.pickle" in os.listdir("artifacts"):
                        with open(f"artifacts/{model_name}_object_{version}.pickle","rb") as a:
                            model_obj=pickle.load(a)
                        prediction_dict[f"{model_name}_{version}_preds"]=model_obj.predict(x)
                        print("prediction done using:%s_%i"%(model_name,version))
                    else:
                        print(f"Model object of {model_name} of version {version} is not available to make predictions")
            if type(model)==list:
                for model_name in model:
                    if f"{model_name}_object_{version}.pickle" in os.listdir("artifacts"):
                        with open(f"artifacts/{model_name}_object_{version}.pickle","rb") as a:
                            model_obj=pickle.load(a)
                        prediction_dict[f"{model_name}_{version}_preds"]=model_obj.predict(x)
                        print("prediction done using:%s_%i"%(model_name,version))
                    else:
                        print(f"Model object of {model_name} of version {version} is not available to make predictions")

    return prediction_dict



