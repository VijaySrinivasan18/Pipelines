import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os


model_dict={"DT":DecisionTreeClassifier,"RF":RandomForestClassifier,"SVC":SVC,"GLM":LogisticRegression,
            "KNN":KNeighborsClassifier}

# def model_build(params,model=None,xdata=None,ydata=None):
#     if model is not None and xdata is not None and ydata is not None:
#         if model=="All":
#             for model_name,model_obj in model_dict.items():
#                 current_model=model_obj(**params[model_name]).fit(xdata,ydata.values.ravel())   #  Hyperparameter tuning pending
#                 print(current_model)
#                 with open(f"artifacts/{model_name}_object.pickle","wb") as s:
#                     pickle.dump(current_model,s)
#         elif type(model) is list:
#             for i in model:
#                 current_model=model_dict[i](**params[i]).fit(xdata,ydata.values.ravel())   #  Hyperparameter tuning pending
#                 print(current_model)
#                 with open(f"artifacts/{i}_object.pickle","wb") as s:
#                     pickle.dump(current_model,s)

    # print("Model Building completed")


def model_build(params,model=None,xdata=None,ydata=None,versions=[1]):
    
    for version in versions:
        if model is not None and xdata is not None and ydata is not None:
            if model=="All":
                for model_name,model_obj in model_dict.items():
                    model_already_avail=False
                    for _,_,files in os.walk("artifacts"):
                        for i in files:
                            if i==f"{model_name}_object_{version}.pickle":
                                model_already_avail=True
                                print(f"Version {version} of {model_name} is already available. Hence Skipping its model building")
                    
                    if model_already_avail == False:
                        current_model=model_obj(**params[model_name]).fit(xdata,ydata.values.ravel())   #  Hyperparameter tuning pending
                        
                        with open(f"artifacts/{model_name}_object_{version}.pickle","wb") as s:
                            pickle.dump(current_model,s)
            elif type(model) is list:
                for i in model:
                    
                    model_already_avail=False
                    for _,_,files in os.walk("artifacts"):
                        for j in files:
                            if j==f"{i}_object_{version}.pickle":
                                print(f"Version {version} of {i} is already available. Hence Skipping its model building")
                                model_already_avail=True
                    if model_already_avail==False:
                        current_model=model_dict[i](**params[i]).fit(xdata,ydata.values.ravel())   #  Hyperparameter tuning pending
                        
                        with open(f"artifacts/{i}_object_{version}.pickle","wb") as s:
                            pickle.dump(current_model,s)
    print("Model Building phase completed")