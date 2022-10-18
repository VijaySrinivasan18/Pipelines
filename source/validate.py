from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd

model_dict={"DT":DecisionTreeClassifier,"RF":RandomForestClassifier,"SVC":SVC,"GLM":LogisticRegression,
            "KNN":KNeighborsClassifier}


def cross_validation(params,model=None,x=None,y=None,cv_num=5):
    cv_scores={}
    if model is not None and x is not None and y is not None:
        if model=="All":
            for model_name,model_obj in model_dict.items():
                scores=cross_val_score(model_obj(**params[model_name]),x,y.values.ravel(),cv=cv_num)
                cv_scores[model_name]={"mean":scores.mean(),"std":scores.std()}
        if type(model)==list:
            for model_name in model:
                scores=cross_val_score(model_dict[model_name](**params[model_name]),x,y.values.ravel(),cv=cv_num)
                cv_scores[model_name]={"mean":scores.mean(),"std":scores.std()}
    return cv_scores


def validation(model=None,actual_target=None,predicted_vals=None,versions=None):
    validation_results={}
    
    if model is not None and actual_target is not None and predicted_vals is not None and versions is not None:
        for version in versions:
            if model=="All":
                for model_name,_ in model_dict.items():
                    if f"{model_name}_{version}_preds" in predicted_vals.keys():

                        accuracy=accuracy_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"])
                        precision=precision_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"],average=None)
                        recall=recall_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"],average=None)
                        f1score=f1_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"],average=None)
                        conf_matrix=confusion_matrix(actual_target,predicted_vals[f"{model_name}_{version}_preds"])
                        validation_results[f"{model_name}_{version}_preds"]={"accuracy":accuracy,"precision":precision,
                        "recall":recall,"f1_score":f1score,"confusion_matrix":conf_matrix}
                    else:
                        print(f"predictions values for version {version} of {model_name} is not available,hence skipping validation ")
                        

            if type(model)==list:
                for model_name in model:
                    if f"{model_name}_{version}_preds" in predicted_vals.keys():

                        accuracy=accuracy_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"])
                        precision=precision_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"],average=None)
                        recall=recall_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"],average=None)
                        f1score=f1_score(actual_target,predicted_vals[f"{model_name}_{version}_preds"],average=None)
                        conf_matrix=confusion_matrix(actual_target,predicted_vals[f"{model_name}_{version}_preds"])
                        validation_results[f"{model_name}_{version}_preds"]={"accuracy":accuracy,"precision":precision,
                        "recall":recall,"f1_score":f1score,"confusion_matrix":conf_matrix}  
                    else:
                        print(f"predictions values for version {version} of {model_name} is not available ,hence skipping validation")
                        
    return validation_results


def results_store(validation_results,ytrue,ypreds,cv_result):


    # for model,metric in validation_results.items():
    #     print(f"metrics of {model}:","\n", metric)
    file_pointer=pd.ExcelWriter("results/metrics.xlsx",mode="w")
    pd.DataFrame(cv_result).to_excel(excel_writer=file_pointer,sheet_name="Cross_val scores")
    for model,metric in validation_results.items():
        i=1
        
        for metric_name,metric_value in metric.items():  
            if metric_name!="confusion_matrix":
                pd.DataFrame([metric_value],index=[metric_name]).to_excel(file_pointer,sheet_name=model,startrow=i,startcol=4,header=False)
                i+=2
            else:
                pd.DataFrame(["Confusion Matrix"]).to_excel(file_pointer,sheet_name=model,startrow=i,startcol=4,header=False,index=False)
                i=i+1
                for j in metric_value:
                    pd.DataFrame([j]).to_excel(file_pointer,sheet_name=model,startrow=i,startcol=4,header=False,index=False)
                    i+=1
        for model,values in ypreds.items():
            pd.DataFrame(ytrue).to_excel(file_pointer,sheet_name=model,startrow=1,startcol=0,index=False)
            
            pd.DataFrame(values,columns=["predicted_values"]).to_excel(file_pointer,sheet_name=model,startrow=1,startcol=1,index=False)
    file_pointer.close()
    print("Results of models stored to excel file")



    