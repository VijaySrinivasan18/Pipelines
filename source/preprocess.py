import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency,kendalltau,pointbiserialr,fisher_exact
from sklearn.model_selection import  train_test_split

def datset_import(path):
    df=pd.read_csv(path)
    df.columns=[x.lower() for x in df.columns]
    df["age"]=df["age"].astype(object)
    return df

def check_missing_values(x):
    if  x.isna().sum().sum()==0:
        print("Dataset do not have any missing values")
    else:
        print("Dataset has mission values")

def bp_encode(x):
    if x=="NORMAL":
        return 1
    elif x=="LOW":
        return 0
    elif x=="HIGH":
        return 2

def bp_decode(x):
    if x==1 :
        return "NORMAL"
    elif x==0:
        return "LOW"
    elif x==2:
        return "HIGH"

def drug_encode(x):
    if x=="drugA":
        return  0
    elif x=="drugB":
        return 1
    elif x=="drugC":
        return 2
    elif x=="drugX":
        return 3
    elif x=="DrugY":
        return 4

def drug_decode(x):
    if x==0:
        return  "drugA"
    elif x==1:
        return "drugB"
    elif x==2:
        return "drugC"
    elif x==3:
        return "drugX"
    elif x==4:
        return "DrugY"




def var_encode(x=None,y=None,is_train=True):
    if x is not None and y is not None :
        if is_train:
            bi_class=[]
            for i in x.columns:
                if len(x[i].unique())==2:
                    bi_class.append(i)

            for i in bi_class:
                label_encode_obj= LabelEncoder().fit(x[i])
                with open(f"artifacts/{i}_obj_encoder.pickle","wb") as s:
                    pickle.dump(label_encode_obj,s)
                # vars()[f"{i}_obj_encoder"]=label_encode_obj
                x[i]=label_encode_obj.transform(x[i])
            y["drug"]=y["drug"].apply(drug_encode)
            x["bp"]=x["bp"].apply(bp_encode)

        elif is_train==False:
            bi_class=[]
            for i in x.columns:
                if len(x[i].unique())==2:
                    bi_class.append(i)

            for i in bi_class:
                with open(f"artifacts/{i}_obj_encoder.pickle","rb") as s:
                    obj=pickle.load(s)
                x[i]=obj.transform(x[i])
            y["drug"]=y["drug"].apply(drug_encode)
            x["bp"]=x["bp"].apply(bp_encode)
        return x,y
    else:
        print("Either data passed for encoding is None")

def dataset_splitting(df,split=0.3):
    x=df.drop("drug",axis=1)
    y=df.iloc[:,[-1]]
    preval_of_y=list (y.value_counts(normalize=True).sort_values())
    if preval_of_y[0]>0.25:
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=14,test_size=split)
        return xtrain,xtest,ytrain,ytest
    else:
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,stratify=y,random_state=14,test_size=split)    
        return xtrain,xtest,ytrain,ytest   

