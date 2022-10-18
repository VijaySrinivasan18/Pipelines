
from source.model import model_build
from source.preprocess import *
from config import config 
from source.prediction import model_prediction
from source.validate import cross_validation,validation,results_store
import pandas as pd



df=datset_import("dataset/training.csv")
print("Traing Dataset Imported")
#  check for Missing values

check_missing_values(df)


xtrain,xtest,ytrain,ytest=dataset_splitting(df)
print("Dataset splitted into train and test")

#  Encoding  traing dataset
xtrain,ytrain=var_encode(xtrain,ytrain,is_train=True)
xtest,ytest=var_encode(xtest,ytest,is_train=False)

print("Categorical features encoded successfully")


cross_validation

if config["cross_val"]==True:
    cv_results=cross_validation(config["parameters"],config["model"],xtrain,ytrain,cv_num=config["cv"])

print(f"Cross Validation results of the models with folds of {config['cv']}")
print(pd.DataFrame(cv_results))



# Model Builing

if config["model_build"]==True:
    print("Starting to build model")
    model_build(config["parameters"],config["model"],xtrain,ytrain,versions=config["version"] )

#  Making Prediction

if config["predict"]==True:
   preds_results= model_prediction(versions=config["version"],model=config["model"],x=xtest)


#  Validating the predictions

if config["validate"]==True:
    validation_results=validation(config["model"],ytest,preds_results,versions=config["version"])


# for model,metric in validation_results.items():
#     print(f"metrics of {model}:","\n", metric)


if config["store_results"]==True:
    results_store(validation_results,ytest,preds_results,cv_results)
        
        
        


